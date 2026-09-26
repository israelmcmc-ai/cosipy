import copy
from typing import Optional, Iterable, Type, Union

import numpy as np
from astromodels import PointSource
from astromodels.sources import Source
from astropy.time import Time
from astropy.units import Quantity
from histpy import Axis

from cosipy import SpacecraftHistory
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.interfaces import UnbinnedThreeMLSourceResponseInterface, EventInterface
from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import TimeTagEmCDSEventInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.util.iterables import asarray

from astropy import units as u


class UnbinnedThreeMLPointSourceResponseTrapz(UnbinnedThreeMLSourceResponseInterface):

    def __init__(self,
                 data: TimeTagEmCDSEventDataInSCFrameInterface,
                 irf: FarFieldSpectralInstrumentResponseFunctionInterface,
                 sc_history: SpacecraftHistory,
                 energies: Quantity,
                 epsilon_axis: Union[Axis, np.ndarray],
                 line_energies: Optional[Quantity] = None,
                 batch_size: int = 1_000_000,
                 offset: Optional[float] = 1e-12):
        """
        Folds a point source spectrum with the IRF using the trapezoidal
        rule in Ei.

        For each event, the Ei integration nodes are placed at
        ``Ei = Em / (1 + Epsilon)`` for every ``Epsilon`` bin center plus
        the outermost ``Epsilon`` edges, so they follow the energy
        dispersion of the IRF, plus at every point of ``energies``, so
        the spectrum and the IRF's ``Ei`` dependence are also resolved
        where the ``Epsilon`` bins are wide. The integral is bounded by
        ``[min(energies), max(energies)]``.

        The total expected counts are integrated over the ``energies``
        grid alone, since there is no measured energy to anchor on.

        Earth occultation and the livetime fraction are accounted for.
        All IRF queries are cached and only recomputed when the source
        location changes.

        Parameters
        ----------
        data : TimeTagEmCDSEventDataInSCFrameInterface
            Events.
        irf : FarFieldSpectralInstrumentResponseFunctionInterface
            Instrument response.
        sc_history : SpacecraftHistory
            Spacecraft orientation and livetime.
        energies : Quantity
            Ei points where the spectrum is always sampled, both for the
            total expected counts and for each event (within its
            ``Epsilon`` range). Its range bounds the integral. Add
            points here to resolve narrow spectral features, e.g. a
            narrow Gaussian line.
        epsilon_axis : histpy.Axis or numpy.ndarray
            Axis (or bin edges) of the fractional energy dispersion
            ``Epsilon = (Em - Ei)/Ei``, e.g. ``irf.epsilon_axis`` for
            an ``IRFRelativeHistUnpolarized``. Events are assumed to
            have zero response outside of it.
        line_energies : Quantity, optional
            Energies of monoenergetic (Dirac delta) components. They
            are not integrated: the spectrum evaluated there is taken
            as the integrated line flux (ph/cm2/s), as returned by
            astromodels' ``DiracDelta``, and multiplied by the response
            at that energy. Any continuum component at these exact
            energies would be misinterpreted as line flux. Don't include
            them in ``energies``.
        batch_size : int
            Maximum number of IRF evaluations per call.
        offset : float, optional
            Added to the expectation density of every event, so that
            events with zero response (e.g. Earth occulted) don't
            result in log(0).
        """

        # Interface inputs
        self._source = None

        # Other implementation inputs
        self._data = data
        self._irf = irf
        self._sc_ori = sc_history
        self._batch_size = batch_size
        self._offset = offset

        self._energies_keV = np.unique(np.asarray(energies.to_value(u.keV), dtype=float))
        self._emin, self._emax = self._energies_keV[0], self._energies_keV[-1]
        self._line_energies_keV = np.zeros(0) if line_energies is None else np.unique(line_energies.to_value(u.keV))

        self._trapz_weights = self._trapz_weights_1d(self._energies_keV)

        if not isinstance(epsilon_axis, Axis):
            epsilon_axis = Axis(epsilon_axis)
        eps_edges = np.asarray(epsilon_axis.edges, dtype=float)
        self._eps_min, self._eps_max = eps_edges[0], eps_edges[-1]
        self._eps_nodes = np.concatenate([[eps_edges[0]], np.asarray(epsilon_axis.centers, dtype=float), [eps_edges[-1]]])

        # Event info
        self._n_events = data.nevents
        self._energy_m_keV = asarray(data.energy_keV, dtype=float)
        self._phi_rad = asarray(data.scattering_angle_rad, dtype=float)
        self._lon_scatt = asarray(data.scattered_lon_rad_sc, dtype=float)
        self._lat_scatt = asarray(data.scattered_lat_rad_sc, dtype=float)

        unique_unix, self._inv_idx = np.unique(data.time.utc.unix, return_inverse=True)
        self._sc_ori_events = self._sc_ori.interp(Time(unique_unix, format='unix', scale='utc'))

        livetime_ratio = self._sc_ori.livetime.to_value(u.s) / self._sc_ori.intervals_duration.to_value(u.s)
        bin_indices = np.searchsorted(self._sc_ori.obstime.utc.unix, unique_unix, side="right") - 1
        bin_indices = np.clip(bin_indices, 0, self._sc_ori.nintervals - 1)
        self._livetime_ratio = livetime_ratio[bin_indices][self._inv_idx]

        # Exposure uses the midpoint of each SC history interval
        obstime = self._sc_ori.obstime
        self._sc_ori_mid = self._sc_ori.interp(obstime[:-1] + (obstime[1:] - obstime[:-1]) / 2)
        self._livetime_s = self._sc_ori.livetime.to_value(u.s)

        # Caches

        # See this issue for the caveats of comparing models
        # https://github.com/threeML/threeML/issues/645
        self._last_convolved_source_dict = None

        # Source location cached separately since changing the response
        # for a given direction is expensive
        self._last_convolved_source_skycoord = None

        # int Aeff(t, Ei) dt, one per self._energies_keV / self._line_energies_keV. cm2*s
        self._exposure = None
        self._line_exposure = None

        # Flattened (event, Ei node) pairs with non-zero trapezoidal weight
        self._node_event_idx = None
        self._node_energy_keV = None
        self._node_weighted_resp = None  # cm2*keV/(keV.rad.sr), includes trapz weight, occultation and livetime ratio

        # axis 0: events. axis 1: line energies
        self._line_resp = None  # cm2/(keV.rad.sr)

        self._nevents = None
        self._expectation_density = None

    @staticmethod
    def _trapz_weights_1d(x):
        widths = np.diff(x)
        weights = np.zeros_like(x)
        weights[:-1] += widths / 2
        weights[1:] += widths / 2
        return weights

    def _event_nodes(self, energy_m_keV):
        """
        Trapezoidal nodes and weights in Ei for each event.

        Returns arrays of shape (nevents, nnodes). Nodes are clipped to
        each event's integration range, so collapsed nodes end up with
        zero weight.
        """

        one_plus_eps = 1 + self._eps_nodes

        with np.errstate(divide='ignore'):
            nodes = energy_m_keV[:, None] / one_plus_eps[None, :]
        nodes[:, one_plus_eps <= 0] = np.inf

        lo = np.maximum(self._emin, energy_m_keV / (1 + self._eps_max))
        if self._eps_min <= -1:
            hi = np.full_like(lo, self._emax)
        else:
            hi = np.minimum(self._emax, energy_m_keV / (1 + self._eps_min))
        hi = np.maximum(hi, lo)

        nodes = np.concatenate([nodes,
                                np.broadcast_to(self._energies_keV, (energy_m_keV.size, self._energies_keV.size)),
                                lo[:, None], hi[:, None]], axis=1)
        nodes = np.clip(nodes, lo[:, None], hi[:, None])
        nodes.sort(axis=1)

        widths = np.diff(nodes, axis=1)
        weights = np.zeros_like(nodes)
        weights[:, :-1] += widths / 2
        weights[:, 1:] += widths / 2

        return nodes, weights

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
        self._exposure = None
        self._line_exposure = None
        self._node_event_idx = None
        self._node_energy_keV = None
        self._node_weighted_resp = None
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

    def _compute_exposure(self, coord):
        """
        int Aeff(t, Ei) dt for each Ei in self._energies_keV and
        self._line_energies_keV.
        """

        sc_coord = self._sc_ori_mid.get_target_in_sc_frame(coord)
        lon = np.asarray(sc_coord.lon.rad, dtype=float)
        lat = np.asarray(sc_coord.lat.rad, dtype=float)

        time_weights = self._livetime_s * ~self._sc_ori_mid.get_earth_occ(coord)

        energies = np.concatenate([self._energies_keV, self._line_energies_keV])
        exposure = np.zeros(energies.size)

        batch_ntimes = max(1, self._batch_size // energies.size)

        for start in range(0, lon.size, batch_ntimes):
            stop = min(start + batch_ntimes, lon.size)
            n = stop - start

            photons = PhotonListWithDirectionAndEnergyInSCFrame(np.repeat(lon[start:stop], energies.size),
                                                                np.repeat(lat[start:stop], energies.size),
                                                                np.tile(energies, n))

            aeff = asarray(self._irf.effective_area_cm2(photons), dtype=float).reshape(n, energies.size)

            exposure += time_weights[start:stop] @ aeff

        self._exposure = exposure[:self._energies_keV.size]
        self._line_exposure = exposure[self._energies_keV.size:]

    def _differential_aeff(self, event_idx, lon, lat, energies_keV):
        photons = PhotonListWithDirectionAndEnergyInSCFrame(lon[event_idx], lat[event_idx], energies_keV)

        events = EmCDSEventDataInSCFrameFromArrays(self._energy_m_keV[event_idx],
                                                   self._lon_scatt[event_idx],
                                                   self._lat_scatt[event_idx],
                                                   self._phi_rad[event_idx])

        return asarray(self._irf.differential_effective_area_cm2(photons, events), dtype=float)

    def _compute_event_response(self, coord):

        sc_coord = self._sc_ori_events.get_target_in_sc_frame(coord)
        lon = np.asarray(sc_coord.lon.rad, dtype=float)[self._inv_idx]
        lat = np.asarray(sc_coord.lat.rad, dtype=float)[self._inv_idx]

        event_weight = self._livetime_ratio * ~self._sc_ori_events.get_earth_occ(coord)[self._inv_idx]

        # Continuum
        nnodes = self._eps_nodes.size + self._energies_keV.size + 2
        batch_nevents = max(1, self._batch_size // nnodes)

        node_event_idx = []
        node_energy = []
        node_resp = []

        for start in range(0, self._n_events, batch_nevents):
            stop = min(start + batch_nevents, self._n_events)

            nodes, weights = self._event_nodes(self._energy_m_keV[start:stop])

            nonzero = weights > 0
            event_idx = np.nonzero(nonzero)[0] + start
            nodes = nodes[nonzero]
            weights = weights[nonzero]

            resp = self._differential_aeff(event_idx, lon, lat, nodes)

            node_event_idx.append(event_idx)
            node_energy.append(nodes)
            node_resp.append(resp * weights * event_weight[event_idx])

        self._node_event_idx = np.concatenate(node_event_idx)
        self._node_energy_keV = np.concatenate(node_energy)
        self._node_weighted_resp = np.concatenate(node_resp)

        # Lines
        nlines = self._line_energies_keV.size
        self._line_resp = np.zeros((self._n_events, nlines))

        if nlines > 0:
            event_idx = np.repeat(np.arange(self._n_events), nlines)
            energies = np.tile(self._line_energies_keV, self._n_events)

            eps = self._energy_m_keV[event_idx] / energies - 1
            inside = (eps >= self._eps_min) & (eps <= self._eps_max)

            line_resp = np.zeros(event_idx.size)
            inside_idx = np.nonzero(inside)[0]

            for start in range(0, inside_idx.size, self._batch_size):
                sel = inside_idx[start:start + self._batch_size]
                line_resp[sel] = self._differential_aeff(event_idx[sel], lon, lat, energies[sel])

            self._line_resp = line_resp.reshape(self._n_events, nlines) * event_weight[:, None]

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

        if (self._exposure is None) or (self._node_weighted_resp is None) or coord != self._last_convolved_source_skycoord:
            # Updating the location is very cost intensive. Only do if necessary
            self._compute_exposure(coord)
            self._compute_event_response(coord)
            self._last_convolved_source_skycoord = coord.copy()

        flux = self._source(self._energies_keV)  # 1/cm2/s/keV (3ML default)
        self._nevents = np.sum(self._exposure * self._trapz_weights * flux)

        density = np.bincount(self._node_event_idx,
                              weights=self._node_weighted_resp * self._source(self._node_energy_keV),
                              minlength=self._n_events)  # 1/keV.s.rad.sr

        if self._line_energies_keV.size > 0:
            line_flux = self._source(self._line_energies_keV)  # 1/cm2/s
            self._nevents += np.sum(self._line_exposure * line_flux)
            density += self._line_resp @ line_flux

        if self._offset is not None:
            density += self._offset

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
