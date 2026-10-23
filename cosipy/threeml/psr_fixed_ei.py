import copy
import gc
from typing import Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
from astromodels import DiracDelta, PointSource
from astromodels.functions.function import CompositeFunction, Function
from astromodels.sources import Source
from astropy import units as u
from astropy.time import Time
from astropy.units import Quantity

from cosipy import SpacecraftHistory
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.interfaces import UnbinnedThreeMLSourceResponseInterface, EventInterface
from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import TimeTagEmCDSEventInSCFrameInterface
from cosipy.response.broadened_irf_cache import BroadenedIRFCache
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized
from cosipy.util.iterables import asarray


class UnbinnedThreeMLPointSourceResponseTrapz(UnbinnedThreeMLSourceResponseInterface):

    def __init__(self,
                 data: TimeTagEmCDSEventDataInSCFrameInterface,
                 irf: IRFRelativeHistUnpolarized,
                 sc_history: SpacecraftHistory,
                 cache: BroadenedIRFCache,
                 cells: Sequence[Tuple[int, int]],
                 mono_nodes: Optional[Quantity] = None,
                 batch_size: int = 1_000_000,
                 offset: Optional[float] = 1e-12,
                 attitude_step: Quantity = 1 * u.s):
        """
        Folds a point source spectrum with the IRF using the same energy nodes, and the
        same spacecraft attitude samples, for the expectation density of the events and
        for the total expected counts.

        The spectrum is ``F = w g``, with the weight ``w`` of the ``cache``, and ``g`` is
        interpolated linearly in ``ln E`` between the nodes ``E_n``, the edges of the
        ``cells`` (see :mod:`cosipy.response.broadened_irf_cache`). With ``K_n`` the
        broadened kernels of the nodes, the density of an event ``x`` and the expected
        counts are

            s(x) = r o sum_n g(E_n) K_n(x)
            S = sum_j livetime_j o_j sum_n g(E_n) Aeff_n,j

        where ``Aeff_n,j`` is the selected effective area of the kernel at the attitude
        of interval ``j``, ``r`` is the livetime fraction and ``o`` is zero if the source
        is occulted by the Earth. The attitude is sampled at the midpoint of each interval
        of ``sc_history`` (or of each sub-interval of at most ``attitude_step``), for ``S`` as well
        as for the events in that interval. Each
        piece of a kernel is normalized by the IRF over the selected data space to its
        selected effective area, so the density integrates to exactly ``S`` for any
        spectrum and any grid of cells: the accuracy of the interpolation of ``g`` changes
        the shape of the model, not its normalization. See :meth:`cells_for` to choose
        the cells for a spectrum.

        Lines that are narrower than a cell should be ``DiracDelta`` components, given in
        ``mono_nodes`` (see :meth:`mono_nodes_of`). They are evaluated with the original
        ``irf``, which is consistent on its own.

        The kernels are computed (or read from the cache file) for the ``NuLambda``
        pixels that the source track needs, one cell at a time, and only when the
        source location changes.

        Parameters
        ----------
        data : TimeTagEmCDSEventDataInSCFrameInterface
            Events.
        irf : IRFRelativeHistUnpolarized
            Instrument response, with the same energy selections as the data. The kernels
            are evaluated with the same selections.
        sc_history : SpacecraftHistory
            Spacecraft orientation and livetime.
        cache : BroadenedIRFCache
            Broadened kernels of ``irf``.
        cells : sequence of (level, index)
            Cells of the cache that tile its energy range.
        mono_nodes : Quantity, optional
            Energies of monoenergetic (Dirac delta) components. The spectrum evaluated
            there is taken as the integrated line flux (ph/cm2/s), as returned by
            astromodels' ``DiracDelta``. Any continuum component at these exact energies
            would be misinterpreted as line flux.
        batch_size : int
            Maximum number of IRF evaluations per call.
        offset : float, optional
            Added to the expectation density of every event, so that events with zero
            response (e.g. Earth occulted) don't result in log(0).
        attitude_step : Quantity
            Maximum time between attitude samples. Each interval of ``sc_history`` is
            split in equal sub-intervals no longer than this, sampled at their midpoint.
            The source moves in the spacecraft frame within an interval (~1 deg in 15 s
            for the DC4 orientation, several deg during slews), which sampling only its
            midpoint would leave out of the model. The cost of finer steps is only in
            ``S``.
        """

        self._source = None

        self._data = data
        self._irf = irf
        self._sc_ori = sc_history
        self._cache = cache
        self._batch_size = batch_size
        self._offset = offset

        self._cells = self._sorted_tiling(cache, cells)
        self._nodes_keV = np.array([cache.cell_edges_keV(*cell)[0] for cell in self._cells]
                                   + [cache.cell_edges_keV(*self._cells[-1])[1]])
        self._weight_index = cache.weight_index

        self._selections = (None if irf._selection_ranges is None
                            else EnergySelector(irf._selection_ranges * u.keV))

        self._line_energies_keV = np.zeros(0) if mono_nodes is None else np.unique(mono_nodes.to_value(u.keV))

        # Events and their spacecraft history interval
        self._n_events = data.nevents
        self._energy_m_keV = asarray(data.energy_keV, dtype=float)
        self._phi_rad = asarray(data.scattering_angle_rad, dtype=float)
        self._lon_scatt = asarray(data.scattered_lon_rad_sc, dtype=float)
        self._lat_scatt = asarray(data.scattered_lat_rad_sc, dtype=float)

        # Attitude samples: each interval of the history is split in equal sub-intervals
        # of at most attitude_step, sampled at their midpoint, with the live time split
        # in proportion. Events use the sample of their sub-interval, S all of them.
        start = np.asarray(self._sc_ori.obstime[:-1].utc.unix, dtype=float)
        duration = np.asarray(self._sc_ori.intervals_duration.to_value(u.s), dtype=float)
        livetime = np.asarray(self._sc_ori.livetime.to_value(u.s), dtype=float)
        nsub = np.maximum(np.ceil(duration / attitude_step.to_value(u.s)).astype(int), 1)
        first = np.concatenate([[0], np.cumsum(nsub)[:-1]])

        interval = np.repeat(np.arange(nsub.size), nsub)
        position = np.arange(nsub.sum()) - first[interval]
        sample_time = start[interval] + (position + 0.5) / nsub[interval] * duration[interval]

        self._livetime_s = (livetime / nsub)[interval]
        self._livetime_ratio = (livetime / duration)[interval]
        self._sc_ori_mid = self._sc_ori.interp(Time(sample_time, format='unix', scale='utc'))

        event_time = asarray(data.time.utc.unix, dtype=float)
        event_interval = np.clip(np.searchsorted(start, event_time, side="right") - 1, 0, nsub.size - 1)
        event_position = np.clip(np.floor((event_time - start[event_interval]) / duration[event_interval]
                                          * nsub[event_interval]), 0, nsub[event_interval] - 1).astype(int)
        self._interval = first[event_interval] + event_position

        self.clear_cache()

    @staticmethod
    def _sorted_tiling(cache: BroadenedIRFCache, cells) -> List[Tuple[int, int]]:
        """The cells sorted by energy, checking that they tile the range of the cache."""

        cells = sorted(((int(level), int(index)) for level, index in cells),
                       key=lambda cell: cache.cell_edges_keV(*cell)[0])
        edges = np.array([cache.cell_edges_keV(*cell) for cell in cells])

        if len(cells) == 0 or not (np.allclose(edges[1:, 0], edges[:-1, 1])
                                   and np.isclose(edges[0, 0], cache.energy_range_keV[0])
                                   and np.isclose(edges[-1, 1], cache.energy_range_keV[1])):
            raise ValueError("The cells must tile the energy range of the cache, without gaps or overlaps")

        return cells

    @staticmethod
    def _continuum_and_lines(spectrum: Union[PointSource, Function]):
        """The components of a spectrum that are not ``DiracDelta``, and those that are."""

        shapes = ([component.shape for component in spectrum.components.values()]
                  if isinstance(spectrum, PointSource) else [spectrum])

        functions = []
        for shape in shapes:
            functions += list(shape.functions) if isinstance(shape, CompositeFunction) else [shape]

        return ([f for f in functions if not isinstance(f, DiracDelta)],
                [f for f in functions if isinstance(f, DiracDelta)])

    @staticmethod
    def mono_nodes_of(spectrum: Union[PointSource, Function]) -> Optional[Quantity]:
        """
        The ``zero_point`` of every ``DiracDelta`` component of a spectrum, for
        ``mono_nodes``, or None if there are none. They must be fixed.
        """

        _, lines = UnbinnedThreeMLPointSourceResponseTrapz._continuum_and_lines(spectrum)

        if any(f.zero_point.free for f in lines):
            raise ValueError("The zero_point of DiracDelta components must be fixed.")

        return np.array([f.zero_point.value for f in lines]) * u.keV if lines else None

    @staticmethod
    def cells_for(spectrum: Union[PointSource, Function],
                  cache: BroadenedIRFCache,
                  accuracy: float = 1e-3,
                  max_level: int = None,
                  points_per_finest_cell: int = 16) -> List[Tuple[int, int]]:
        """
        Cells for which the interpolation of ``g = F / w`` (linear in ``ln E``) is
        accurate to about ``accuracy``, for the continuum part of the spectrum at its
        current parameter values.

        Starting from level 0, each cell is split while
        ``(Delta^2 / 8) max|g''| / mean(g)`` is above ``accuracy``, with ``Delta`` the
        width of the cell in ``ln E`` and ``g''`` the second derivative in ``ln E``,
        evaluated on a grid of ``points_per_finest_cell`` points per cell of
        ``max_level``. The grid is fixed during the fit, so choose it for the range of
        spectra that the fit can reach (e.g. with the parameters at their expected
        values). Features narrower than the finest cells can't be resolved: make them
        ``DiracDelta`` components if they are lines.

        Returns
        -------
        list of (level, index)
        """

        if max_level is None:
            max_level = cache.max_level

        continuum, _ = UnbinnedThreeMLPointSourceResponseTrapz._continuum_and_lines(spectrum)

        emin, emax = cache.energy_range_keV
        npoints = cache.ncells(max_level) * points_per_finest_cell + 1
        log_e = np.linspace(np.log(emin), np.log(emax), npoints)
        energy = np.exp(log_e)

        g = np.sum([f(energy) for f in continuum], axis=0) / energy ** cache.weight_index
        g2 = np.abs(np.gradient(np.gradient(g, log_e), log_e))

        def needs_split(level, index):
            lo, hi = np.log(cache.cell_edges_keV(level, index))
            inside = (log_e >= lo) & (log_e <= hi)
            mean = np.mean(np.abs(g[inside]))
            return mean > 0 and (hi - lo) ** 2 / 8 * np.max(g2[inside]) / mean > accuracy

        cells = []
        pending = [(0, i) for i in range(cache.ncells(0))]
        while pending:
            level, index = pending.pop()
            if level < max_level and needs_split(level, index):
                pending += [(level + 1, 2 * index), (level + 1, 2 * index + 1)]
            else:
                cells.append((level, index))

        return sorted(cells, key=lambda cell: cache.cell_edges_keV(*cell)[0])

    @property
    def event_type(self) -> Type[EventInterface]:
        return TimeTagEmCDSEventInSCFrameInterface

    def set_source(self, source: Source):
        """
        The source is passed as a reference and it's parameters
        can change. Remember to check if it changed since the
        last time the user called expectation.
        """
        if not isinstance(source, PointSource):
            raise TypeError("I only know how to handle point sources!")

        self._source = source

    def clear_cache(self):

        self._last_convolved_source_dict = None
        self._last_convolved_source_skycoord = None
        self._exposure = None  # per node, cm2 s keV (w units)
        self._event_resp = None  # (events, nodes), cm2 keV / (keV rad sr) (w units)
        self._line_exposure = None
        self._line_resp = None
        self._nevents = None
        self._expectation_density = None

    def copy(self) -> "UnbinnedThreeMLPointSourceResponseTrapz":
        """
        This method is used to re-use the same object for multiple
        sources.
        It is expected to return a copy of itself, but deepcopying
        any necessary information such that when
        a new source is set, the expectation calculation
        are independent.

        psr1 = ThreeMLSourceResponse()
        psr2 = psr.copy()
        psr1.set_source(source1)
        psr2.set_source(source2)
        """

        new = copy.copy(self)
        new.clear_cache()
        new._source = None
        return new

    def init_cache(self):
        self._update_cache()

    def _area_and_response(self, irf, energy_keV, lon, lat, time_weights):
        """
        For a photon energy: the exposure, ``sum_j time_weights_j Aeff_j`` over the
        intervals, and the differential effective area of each event at the attitude
        of its interval, times its livetime fraction and visibility.
        """

        nintervals = lon.size
        exposure = 0.
        for start in range(0, nintervals, self._batch_size):
            stop = min(start + self._batch_size, nintervals)
            photons = PhotonListWithDirectionAndEnergyInSCFrame(lon[start:stop], lat[start:stop],
                                                                np.full(stop - start, energy_keV))
            exposure += time_weights[start:stop] @ asarray(irf.effective_area_cm2(photons), dtype=float)

        response = np.zeros(self._n_events)
        for start in range(0, self._n_events, self._batch_size):
            stop = min(start + self._batch_size, self._n_events)
            j = self._interval[start:stop]
            photons = PhotonListWithDirectionAndEnergyInSCFrame(lon[j], lat[j], np.full(stop - start, energy_keV))
            events = EmCDSEventDataInSCFrameFromArrays(self._energy_m_keV[start:stop],
                                                       self._lon_scatt[start:stop],
                                                       self._lat_scatt[start:stop],
                                                       self._phi_rad[start:stop])
            response[start:stop] = asarray(irf.differential_effective_area_cm2(photons, events), dtype=float)

        return exposure, response * (self._livetime_ratio * self._visible)[self._interval]

    def _reaches_selection(self, cell) -> bool:
        """Whether the measured energies that a cell can produce overlap the selection."""

        if self._selections is None:
            return True

        lo = self._cache.cell_edges_keV(*cell)[0]
        eps_edges = self._cache.epsilon_edges(*cell)
        em_min, em_max = lo * (1 + eps_edges[0]), lo * (1 + eps_edges[-1])

        return any(a < em_max and b > em_min for a, b in self._selections.energy_ranges_keV)

    def _compute_cache(self, coord):

        sc_coord = self._sc_ori_mid.get_target_in_sc_frame(coord)
        lon = np.atleast_1d(np.asarray(sc_coord.lon.rad, dtype=float))
        lat = np.atleast_1d(np.asarray(sc_coord.lat.rad, dtype=float))
        self._visible = ~np.atleast_1d(np.asarray(self._sc_ori_mid.get_earth_occ(coord), dtype=bool))
        time_weights = self._livetime_s * self._visible

        nnodes = self._nodes_keV.size
        self._exposure = np.zeros(nnodes)
        self._event_resp = np.zeros((self._n_events, nnodes))

        # The node at the lower edge of a cell gets its L piece, the one at the upper edge its R piece
        for n, cell in enumerate(self._cells):
            if not self._reaches_selection(cell):
                continue

            irf_r, irf_l = self._cache.piece_irfs(*cell, selections=self._selections)
            for piece_irf, node in ((irf_l, n), (irf_r, n + 1)):
                exposure, response = self._area_and_response(piece_irf, self._nodes_keV[n], lon, lat, time_weights)
                self._exposure[node] += exposure
                self._event_resp[:, node] += response

            # Histograms are in reference cycles (with their accessors), so the tables of
            # a cell would only be freed by the cyclic garbage collector, after several cells
            del irf_r, irf_l, piece_irf
            gc.collect()

        # Lines, with the original IRF
        nlines = self._line_energies_keV.size
        self._line_exposure = np.zeros(nlines)
        self._line_resp = np.zeros((self._n_events, nlines))
        for k, energy in enumerate(self._line_energies_keV):
            self._line_exposure[k], self._line_resp[:, k] = self._area_and_response(self._irf, energy, lon, lat,
                                                                                    time_weights)

    def _update_cache(self):
        """
        Performs all calculation as needed depending on the current source location
        """
        if self._source is None:
            raise RuntimeError("Call set_source() first.")

        source_dict = self._source.to_dict()
        coord = self._source.position.sky_coord

        if (self._nevents is not None) and self._last_convolved_source_dict == source_dict:
            # Nothing has changed
            return

        if self._exposure is None or coord != self._last_convolved_source_skycoord:
            # Updating the location is very cost intensive. Only do if necessary
            self._compute_cache(coord)
            self._last_convolved_source_skycoord = coord.copy()

        # g = F / w at the nodes. F in 1/cm2/s/keV (3ML default)
        g = self._source(self._nodes_keV) / self._nodes_keV ** self._weight_index

        self._nevents = float(self._exposure @ g)
        density = self._event_resp @ g  # 1/keV.s.rad.sr

        if self._line_energies_keV.size > 0:
            line_flux = self._source(self._line_energies_keV)  # 1/cm2/s
            self._nevents += float(self._line_exposure @ line_flux)
            density = density + self._line_resp @ line_flux

        if self._offset is not None:
            density = density + self._offset

        self._expectation_density = density

        self._last_convolved_source_dict = source_dict

    def expected_counts(self) -> float:
        """
        Total expected counts
        """

        self._update_cache()

        return self._nevents

    def expectation_density(self) -> Iterable[float]:
        """
        Expected number of counts density for each event
        """

        self._update_cache()

        return self._expectation_density
