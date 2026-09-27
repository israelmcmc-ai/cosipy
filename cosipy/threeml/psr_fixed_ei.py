import copy
import itertools
from typing import Optional, Iterable, Type, Union

import numpy as np
from astromodels import PointSource, DiracDelta
from astromodels.functions.function import Function, CompositeFunction
from astromodels.sources import Source
from astropy.time import Time
from astropy.units import Quantity
from histpy import Axis
from scipy.stats import qmc

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
        ``Ei = Em / (1 + Epsilon)`` for every ``Epsilon`` node, so they
        follow the energy dispersion of the IRF, plus at every point of
        ``energies``, so the spectrum and the IRF's ``Ei`` dependence
        are also resolved where the ``Epsilon`` nodes are sparse. The
        integral is bounded by ``[min(energies), max(energies)]``.

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
            Fractional energy dispersion ``Epsilon = (Em - Ei)/Ei``. If
            an Axis (e.g. ``irf.epsilon_axis`` for an
            ``IRFRelativeHistUnpolarized``), its bin centers plus its
            outer edges are used as nodes. If an array, the nodes
            themselves (see ``integration_nodes()``). Events are assumed
            to have zero response outside of the nodes' range.
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

        if isinstance(epsilon_axis, Axis):
            eps_edges = np.asarray(epsilon_axis.edges, dtype=float)
            self._eps_nodes = np.concatenate([eps_edges[:1], np.asarray(epsilon_axis.centers, dtype=float), eps_edges[-1:]])
        else:
            self._eps_nodes = np.unique(np.asarray(epsilon_axis, dtype=float))
        self._eps_min, self._eps_max = self._eps_nodes[0], self._eps_nodes[-1]

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
    def integration_nodes(spectrum: Union[PointSource, Function],
                          irf: FarFieldSpectralInstrumentResponseFunctionInterface,
                          energy_range: Quantity,
                          accuracy: float = 0.01,
                          estimate_epsilon: bool = True,
                          epsilon_range: Optional[tuple] = None,
                          nsamples: int = 128,
                          npoints_dense: int = 100_001,
                          npoints_dense_epsilon: int = 20_001) -> dict:
        """
        Suggest the ``energies``, ``line_energies`` and ``epsilon_axis``
        arguments for a given spectrum and IRF, aiming at a relative
        accuracy of about ``accuracy`` for the expected counts and the
        expectation density of each event.

        Both ``energies`` and the ``Epsilon`` nodes are obtained by
        thinning out a dense grid such that the trapezoidal rule on each
        remaining interval agrees with the dense grid within
        ``accuracy``, relative to that interval's own integral (or to
        1e-3 of the total, for negligible tails). This resolves narrow
        features as long as the dense grid does.

        - ``line_energies``: the ``zero_point`` of every ``DiracDelta``
          component. They must be fixed.
        - ``energies``: the spectrum without the Dirac deltas, on a
          dense log-spaced grid. The union of the nodes needed for the
          current parameter values, for every corner of the
          ``[min_value, max_value]`` range of the free parameters, and
          for ``nsamples`` quasi-random points within that range. A
          missing bound is replaced by the current value. Parameters
          whose range spans more than a decade are sampled in log.
        - ``epsilon_axis``: if ``estimate_epsilon``, an array of
          ``Epsilon`` nodes, from the ``irf.event_probability()`` of
          probe events as a function of ``Epsilon``, for a few ``Ei``
          within ``energy_range``, off-axis angles and scattering
          angles, with the scattered direction on the Compton cone.
          Otherwise, ``irf.epsilon_axis`` as is (e.g. for
          ``IRFRelativeHistUnpolarized``).

        Parameters
        ----------
        spectrum : astromodels.PointSource or astromodels.Function
            Source, or its spectral shape.
        irf : FarFieldSpectralInstrumentResponseFunctionInterface
            Instrument response.
        energy_range : Quantity
            ``(min, max)`` Ei range of the integration.
        accuracy : float
            Target relative accuracy.
        estimate_epsilon : bool
            Estimate the ``Epsilon`` nodes from the IRF's event
            probability. If False, the IRF must have an
            ``epsilon_axis``.
        epsilon_range : tuple, optional
            ``(min, max)`` range of ``Epsilon`` with a non-zero
            response, if ``estimate_epsilon``. By default, the range of ``irf.epsilon_axis`` if
            the IRF has one (e.g. ``IRFRelativeHistUnpolarized``), and
            ``(-1, 1)`` otherwise.
        nsamples : int
            Number of quasi-random samples of the free parameters.
        npoints_dense : int
            Number of points of the dense ``Ei`` grid.
        npoints_dense_epsilon : int
            Number of points of the dense ``Epsilon`` grid.

        Returns
        -------
        dict
            Keyword arguments ``energies``, ``line_energies`` and
            ``epsilon_axis`` for this class' constructor.
        """

        if isinstance(spectrum, PointSource):
            shapes = [component.shape for component in spectrum.components.values()]
        else:
            shapes = [spectrum]

        functions = []
        for shape in shapes:
            functions += list(shape.functions) if isinstance(shape, CompositeFunction) else [shape]

        lines = [f for f in functions if isinstance(f, DiracDelta)]
        continuum = [f for f in functions if not isinstance(f, DiracDelta)]

        if any(f.zero_point.free for f in lines):
            raise ValueError("The zero_point of DiracDelta components must be fixed.")

        line_energies = [f.zero_point.value for f in lines]

        # Energies
        emin, emax = energy_range.to_value(u.keV)
        x = np.geomspace(emin, emax, npoints_dense)
        keep = np.zeros(x.size, dtype=bool)

        parameters = [par for f in continuum for par in f.parameters.values() if par.free]
        current = [par.value for par in parameters]

        try:
            for values in UnbinnedThreeMLPointSourceResponseTrapz._parameter_samples(parameters, nsamples):
                for par, value in zip(parameters, values):
                    par.value = value
                keep |= UnbinnedThreeMLPointSourceResponseTrapz._thin(x, np.sum([f(x) for f in continuum], axis=0), accuracy)
        finally:
            for par, value in zip(parameters, current):
                par.value = value

        # Epsilon
        epsilon_axis = getattr(irf, 'epsilon_axis', None)

        if estimate_epsilon:
            if epsilon_range is None:
                epsilon_range = (-1, 1) if epsilon_axis is None else (epsilon_axis.lo_lim, epsilon_axis.hi_lim)

            epsilon = UnbinnedThreeMLPointSourceResponseTrapz._epsilon_nodes(irf, emin, emax, epsilon_range, accuracy,
                                                                             npoints_dense_epsilon)
        elif epsilon_axis is None:
            raise ValueError(f"{type(irf).__name__} doesn't have an epsilon_axis. Use estimate_epsilon = True.")
        else:
            epsilon = epsilon_axis

        return {'energies': x[keep] * u.keV,
                'line_energies': np.array(line_energies) * u.keV if line_energies else None,
                'epsilon_axis': epsilon}

    @staticmethod
    def _thin(x, y, accuracy):
        """
        Mask of the points of a dense grid needed to integrate y(x) with
        the trapezoidal rule within ``accuracy``, relative to each
        interval's integral. Intervals with less than 1e-3 of the total
        integral are held to 1e-3 of the total instead, so negligible
        tails are not refined.
        """

        cum_integral = np.concatenate([[0], np.cumsum(np.diff(x) * (y[1:] + y[:-1]) / 2)])
        floor = 1e-3 * cum_integral[-1]

        keep = np.zeros(x.size, dtype=bool)
        keep[[0, -1]] = True
        intervals = [(0, x.size - 1)]

        while intervals:
            i, j = intervals.pop()

            if j - i < 2:
                continue

            coarse = (x[j] - x[i]) * (y[i] + y[j]) / 2
            fine = cum_integral[j] - cum_integral[i]

            if abs(coarse - fine) > accuracy * max(fine, floor):
                m = (i + j) // 2
                keep[m] = True
                intervals += [(i, m), (m, j)]

        return keep

    @staticmethod
    def _parameter_samples(parameters, nsamples):
        """
        Current values, corners of the [min_value, max_value] box and
        quasi-random samples within it.
        """

        lo = np.array([par.value if par.min_value is None else par.min_value for par in parameters], dtype=float)
        hi = np.array([par.value if par.max_value is None else par.max_value for par in parameters], dtype=float)
        bounds = (lo.copy(), hi.copy())
        log = (lo > 0) & (hi > 10 * lo)

        lo[log], hi[log] = np.log(lo[log]), np.log(hi[log])

        unit_samples = [np.array(corner, dtype=float) for corner in itertools.product([0, 1], repeat=len(parameters))]
        if parameters and nsamples > 0:
            unit_samples += list(qmc.Sobol(len(parameters), seed=0).random(nsamples))

        yield [par.value for par in parameters]

        for unit in unit_samples:
            values = lo + unit * (hi - lo)
            values[log] = np.exp(values[log])
            yield np.clip(values, *bounds)

    @staticmethod
    def _epsilon_nodes(irf, emin, emax, epsilon_range, accuracy, npoints_dense):
        """
        Epsilon nodes that resolve irf.event_probability() for a set of
        probe events.
        """

        eps_lo, eps_hi = epsilon_range
        eps = np.linspace(max(eps_lo, -1 + 1e-3), eps_hi, npoints_dense)

        # Photons along (theta, lon = 0), scattered direction on the Compton cone
        energy, theta, phi = [a.ravel() for a in np.meshgrid(np.geomspace(emin, emax, 4),
                                                             np.deg2rad([0, 30, 60]),
                                                             np.deg2rad([10, 30, 60, 90, 120]),
                                                             indexing='ij')]

        scatt_z = np.cos(phi) * np.cos(theta) - np.sin(phi) * np.sin(theta)
        scatt_x = np.cos(phi) * np.sin(theta) + np.sin(phi) * np.cos(theta)

        def repeat(a):
            return np.repeat(a, eps.size)

        photons = PhotonListWithDirectionAndEnergyInSCFrame(repeat(np.zeros_like(theta)), repeat(np.pi / 2 - theta),
                                                            repeat(energy))

        events = EmCDSEventDataInSCFrameFromArrays((energy[:, None] * (1 + eps)).ravel(),
                                                   repeat(np.where(scatt_x < 0, np.pi, 0)),
                                                   repeat(np.arcsin(np.clip(scatt_z, -1, 1))),
                                                   repeat(phi))

        prob = asarray(irf.event_probability(photons, events), dtype=float).reshape(energy.size, eps.size)

        keep = np.zeros(eps.size, dtype=bool)
        for p in prob:
            if np.any(p > 0):
                keep |= UnbinnedThreeMLPointSourceResponseTrapz._thin(eps, p, accuracy)

        nodes = eps[keep]
        nodes[[0, -1]] = eps_lo, eps_hi

        return nodes

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
