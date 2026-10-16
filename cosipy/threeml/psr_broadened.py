import copy
from typing import Optional, Iterable, Type, Sequence

import numpy as np
from astromodels import PointSource
from astromodels.sources import Source
from astropy import units as u
from astropy.time import Time
from astropy.units import Quantity

from cosipy import SpacecraftHistory
from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.interfaces import UnbinnedThreeMLSourceResponseInterface, EventInterface
from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import TimeTagEmCDSEventInSCFrameInterface
from cosipy.response.broadened_relative_irf_hist import hat_functions, hat_integrals_keV
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized
from cosipy.util.iterables import asarray


class UnbinnedThreeMLPointSourceResponseBroadened(UnbinnedThreeMLSourceResponseInterface):

    def __init__(self,
                 data: TimeTagEmCDSEventDataInSCFrameInterface,
                 irf: IRFRelativeHistUnpolarized,
                 sc_history: SpacecraftHistory,
                 fine_irf: Optional[IRFRelativeHistUnpolarized] = None,
                 fine_regions: Optional[Sequence[Quantity]] = None,
                 fine_nodes_per_decade: int = 200,
                 mono_nodes: Optional[Quantity] = None,
                 batch_size: int = 1_000_000,
                 offset: Optional[float] = 1e-12):
        """
        Folds a point source spectrum with a broadened IRF (see
        :mod:`cosipy.response.broadened_relative_irf_hist`), using the same
        ``Ei`` nodes, and the same spacecraft attitude samples, for the
        expectation density of the events and for the total expected counts.

        With ``F`` the spectrum and ``E_n`` the nodes (the ``Ei`` centers of
        ``irf``), the density of an event ``x`` and the expected counts are

            s(x) = r o sum_n Delta_n F(E_n) dAeff_n(x)
            S = sum_j livetime_j o_j sum_n Delta_n F(E_n) Aeff_n

        where ``Delta_n`` is the integral of the hat function of each node,
        ``dAeff_n`` and ``Aeff_n`` are the differential and total (selected)
        effective areas of ``irf`` at ``E_n``, ``r`` is the livetime fraction
        and ``o`` is zero if the source is occulted by the Earth. The
        spacecraft attitude is sampled at the midpoint of each interval of
        ``sc_history``, for ``S`` as well as for the events in that interval.
        The IRF normalization then makes the density integrate over the
        selected data space to exactly ``S``, for any spectrum.

        Narrow spectral features can't be resolved by the broadened nodes.
        Within each of the ``fine_regions`` the hats of the nodes inside the
        region are replaced by the original ``fine_irf``, evaluated on a
        global log grid of ``fine_nodes_per_decade`` (plus the region
        boundaries and the ``Ei`` centers of ``fine_irf``), weighted by the
        trapezoidal rule and by the sum of the replaced hats, so the hats
        that remain and the fine nodes still add up to the full integral.
        The same nodes are used for the density and for ``S``.

        Parameters
        ----------
        data : TimeTagEmCDSEventDataInSCFrameInterface
            Events.
        irf : IRFRelativeHistUnpolarized
            Broadened IRF, e.g. ``IRFRelativeHistUnpolarized.from_h5(filename,
            broadened=True, selections=...)``. Its ``Ei`` centers are the
            nodes, and their range bounds the integral.
        sc_history : SpacecraftHistory
            Spacecraft orientation and livetime.
        fine_irf : IRFRelativeHistUnpolarized, optional
            Original (not broadened) IRF, with the same selections. Required
            for ``fine_regions`` and ``mono_nodes``.
        fine_regions : sequence of Quantity, optional
            ``[lo, hi]`` energy ranges. The nodes inside each range (at least
            one) are replaced by the fine evaluation, which then extends to
            the neighboring nodes.
        fine_nodes_per_decade : int
            Density of the fine grid.
        mono_nodes : Quantity, optional
            Energies of monoenergetic (Dirac delta) components, evaluated
            with ``fine_irf``. The spectrum evaluated there is taken as the
            integrated line flux (ph/cm2/s), as returned by astromodels'
            ``DiracDelta``. Any continuum component at these exact energies
            would be misinterpreted as line flux.
        batch_size : int
            Maximum number of IRF evaluations per call.
        offset : float, optional
            Added to the expectation density of every event, so that
            events with zero response (e.g. Earth occulted) don't
            result in log(0).
        """

        self._source = None

        self._data = data
        self._irf = irf
        self._fine_irf = fine_irf
        self._sc_ori = sc_history
        self._batch_size = batch_size
        self._offset = offset

        self._nodes_keV = np.asarray(irf.axes['Ei'].centers, dtype=float)
        coarse_weights = hat_integrals_keV(self._nodes_keV)

        if (fine_regions is not None and len(fine_regions) > 0 or mono_nodes is not None) and fine_irf is None:
            raise ValueError("fine_regions and mono_nodes require fine_irf")

        fine_energies, fine_weights = [], []
        replaced = np.zeros(self._nodes_keV.size, dtype=bool)

        for lo, hi in ([] if fine_regions is None else fine_regions):
            lo, hi = u.Quantity([lo, hi]).to_value(u.keV)
            region = (self._nodes_keV >= lo) & (self._nodes_keV <= hi)

            if not region.any():
                raise ValueError(f"The fine region [{lo}, {hi}] keV doesn't contain any node")

            idx = np.nonzero(region)[0]
            e_lo = self._nodes_keV[max(idx[0] - 1, 0)]
            e_hi = self._nodes_keV[min(idx[-1] + 1, self._nodes_keV.size - 1)]
            energy = np.geomspace(e_lo, e_hi, max(int(np.ceil(np.log10(e_hi / e_lo) * fine_nodes_per_decade)), 1) + 1)
            kinks = np.asarray(fine_irf.axes['Ei'].centers, dtype=float)
            energy = np.unique(np.concatenate([energy, self._nodes_keV[idx], kinks[(kinks > e_lo) & (kinks < e_hi)]]))

            trapz = np.zeros(energy.size)
            trapz[:-1] += np.diff(energy) / 2
            trapz[1:] += np.diff(energy) / 2

            fine_energies.append(energy)
            fine_weights.append(trapz * hat_functions(self._nodes_keV, energy)[:, region].sum(axis=1))
            replaced |= region

        self._coarse_weights = np.where(replaced, 0., coarse_weights)
        self._fine_energies_keV = np.concatenate(fine_energies) if fine_energies else np.zeros(0)
        self._fine_weights = np.concatenate(fine_weights) if fine_weights else np.zeros(0)

        self._line_energies_keV = np.zeros(0) if mono_nodes is None else np.unique(mono_nodes.to_value(u.keV))

        # Events and their SC interval
        self._n_events = data.nevents
        self._energy_m_keV = asarray(data.energy_keV, dtype=float)
        self._phi_rad = asarray(data.scattering_angle_rad, dtype=float)
        self._lon_scatt = asarray(data.scattered_lon_rad_sc, dtype=float)
        self._lat_scatt = asarray(data.scattered_lat_rad_sc, dtype=float)

        obstime = self._sc_ori.obstime
        self._interval = np.clip(np.searchsorted(obstime.utc.unix, data.time.utc.unix, side="right") - 1,
                                 0, self._sc_ori.nintervals - 1)
        self._livetime_s = self._sc_ori.livetime.to_value(u.s)
        self._livetime_ratio = self._livetime_s / self._sc_ori.intervals_duration.to_value(u.s)
        self._sc_ori_mid = self._sc_ori.interp(obstime[:-1] + (obstime[1:] - obstime[:-1]) / 2)

        self.clear_cache()

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
        self._event_resp = None
        self._nevents = None
        self._expectation_density = None

    def copy(self) -> "UnbinnedThreeMLPointSourceResponseBroadened":
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

    def _energies(self):
        """Coarse nodes, fine nodes and lines, with their IRF and integration weight (1 for lines)."""

        return [(self._irf, self._nodes_keV, self._coarse_weights),
                (self._fine_irf, self._fine_energies_keV, self._fine_weights),
                (self._fine_irf, self._line_energies_keV, np.ones(self._line_energies_keV.size))]

    def _compute_cache(self, coord):

        sc_coord = self._sc_ori_mid.get_target_in_sc_frame(coord)
        lon = np.asarray(sc_coord.lon.rad, dtype=float)
        lat = np.asarray(sc_coord.lat.rad, dtype=float)
        visible = ~np.asarray(self._sc_ori_mid.get_earth_occ(coord))

        exposure, event_resp = [], []

        for irf, energies, weights in self._energies():
            nenergy = energies.size

            if nenergy == 0:
                exposure.append(np.zeros(0))
                event_resp.append(np.zeros((self._n_events, 0)))
                continue

            # Total, per interval
            aeff = np.zeros((lon.size, nenergy))
            batch = max(1, self._batch_size // nenergy)
            for start in range(0, lon.size, batch):
                stop = min(start + batch, lon.size)
                n = stop - start
                photons = PhotonListWithDirectionAndEnergyInSCFrame(np.repeat(lon[start:stop], nenergy),
                                                                    np.repeat(lat[start:stop], nenergy),
                                                                    np.tile(energies, n))
                aeff[start:stop] = asarray(irf.effective_area_cm2(photons), dtype=float).reshape(n, nenergy)

            exposure.append(((self._livetime_s * visible) @ aeff) * weights)

            # Events, at the attitude of their interval
            resp = np.zeros((self._n_events, nenergy))
            for start in range(0, self._n_events, batch):
                stop = min(start + batch, self._n_events)
                n = stop - start
                j = self._interval[start:stop]
                photons = PhotonListWithDirectionAndEnergyInSCFrame(np.repeat(lon[j], nenergy),
                                                                    np.repeat(lat[j], nenergy),
                                                                    np.tile(energies, n))
                events = EmCDSEventDataInSCFrameFromArrays(np.repeat(self._energy_m_keV[start:stop], nenergy),
                                                           np.repeat(self._lon_scatt[start:stop], nenergy),
                                                           np.repeat(self._lat_scatt[start:stop], nenergy),
                                                           np.repeat(self._phi_rad[start:stop], nenergy))
                diff = asarray(irf.differential_effective_area_cm2(photons, events), dtype=float).reshape(n, nenergy)
                resp[start:stop] = diff * (weights * (self._livetime_ratio[j] * visible[j])[:, None])

            event_resp.append(resp)

        self._exposure = np.concatenate(exposure)
        self._event_resp = np.concatenate(event_resp, axis=1)

    def _update_cache(self):
        """
        Performs all calculation as needed depending on the current source location
        """
        if self._source is None:
            raise RuntimeError("Call set_source() first.")

        source_dict = self._source.to_dict()
        coord = self._source.position.sky_coord

        if (self._nevents is not None) and self._last_convolved_source_dict == source_dict:
            return

        if self._exposure is None or coord != self._last_convolved_source_skycoord:
            self._compute_cache(coord)
            self._last_convolved_source_skycoord = coord.copy()

        energies = np.concatenate([e for _, e, _ in self._energies()])
        flux = self._source(energies)  # 1/cm2/s/keV (3ML default), or 1/cm2/s for lines

        self._nevents = float(self._exposure @ flux)

        density = self._event_resp @ flux  # 1/keV.s.rad.sr

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
