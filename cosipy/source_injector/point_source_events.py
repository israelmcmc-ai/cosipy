from typing import Tuple

import numpy as np
from astromodels import PointSource
from astropy import units as u
from astropy.time import Time

from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromArrays
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.spacecraftfile import SpacecraftHistory


def simulate_point_source_events(irf: FarFieldSpectralInstrumentResponseFunctionInterface,
                                 sc_history: SpacecraftHistory,
                                 source: PointSource,
                                 energy_range_keV: Tuple[float, float] = (100., 10_000.),
                                 nenergy: int = 4001,
                                 rng: np.random.Generator = None,
                                 batch_size: int = 1_000_000,
                                 direction_at: str = 'event') -> Tuple[TimeTagEmCDSEventDataInSCFrameFromArrays,
                                                                       np.ndarray]:
    """
    Events of a point source with a continuum spectrum, drawn from the IRF.

    For each interval ``j`` of ``sc_history``, with the source direction at its midpoint
    (and no events if the source is occulted by the Earth there), the number of
    detected photons is Poisson with mean ``livetime_j * int F(E) Aeff_j(E) dE`` over
    ``energy_range_keV``, with the effective area after the IRF's selections. Their
    energies follow ``F Aeff_j`` (trapezoidal on a log grid of ``nenergy`` points), their
    times are uniform within the interval, and their direction is the source direction
    at their time. Each photon gives one event drawn by ``irf.random_events``, which is
    conditional on the selections, so the events are as if all photons had been
    simulated and the selections applied.

    Parameters
    ----------
    irf : FarFieldSpectralInstrumentResponseFunctionInterface
        Instrument response, e.g. an :class:`IRFRelativeHistUnpolarized`. If it has an
        ``rng`` attribute, it is set to ``rng``.
    sc_history : SpacecraftHistory
        Spacecraft orientation and live time.
    source : astromodels.PointSource
        Source, with a spectrum in 1/cm2/s/keV.
    energy_range_keV : tuple of float
        Range of the photon energies.
    nenergy : int
        Points of the energy grid.
    rng : numpy.random.Generator, optional
    batch_size : int
        Maximum number of IRF evaluations per call.
    direction_at : {'event', 'midpoint'}
        Source direction of each photon: at its time (default, the physical one), or at
        the midpoint of its interval, which is what the folding assumes.

    Returns
    -------
    events : TimeTagEmCDSEventDataInSCFrameFromArrays
        Sorted by time.
    photon_energy_keV : numpy.ndarray
        True energy of the photon of each event.
    """

    if rng is None:
        rng = np.random.default_rng()
    if hasattr(irf, 'rng'):
        irf.rng = rng

    coord = source.position.sky_coord
    obstime = sc_history.obstime
    mid = sc_history.interp(obstime[:-1] + (obstime[1:] - obstime[:-1]) / 2)
    sc_coord = mid.get_target_in_sc_frame(coord)
    lon, lat = np.asarray(sc_coord.lon.rad, dtype=float), np.asarray(sc_coord.lat.rad, dtype=float)
    time_weights = sc_history.livetime.to_value(u.s) * ~np.asarray(mid.get_earth_occ(coord), dtype=bool)

    energy = np.geomspace(*energy_range_keV, nenergy)
    flux = np.asarray(source(energy), dtype=float)

    nintervals = lon.size
    aeff = np.zeros((nintervals, nenergy))
    batch = max(1, batch_size // nenergy)
    for start in range(0, nintervals, batch):
        stop = min(start + batch, nintervals)
        photons = PhotonListWithDirectionAndEnergyInSCFrame(np.repeat(lon[start:stop], nenergy),
                                                            np.repeat(lat[start:stop], nenergy),
                                                            np.tile(energy, stop - start))
        aeff[start:stop] = np.asarray(irf.effective_area_cm2(photons), dtype=float).reshape(stop - start, nenergy)

    # Trapezoidal integral of F Aeff in each energy step, per interval
    integrand = aeff * flux
    steps = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(energy)
    counts = rng.poisson(time_weights * steps.sum(axis=1))
    interval = np.repeat(np.arange(nintervals), counts)
    n = interval.size

    # Energy: a step with probability proportional to its integral, then the linear
    # integrand within it (inverse CDF of a trapezoid)
    cdf = np.cumsum(steps, axis=1)
    step = np.array([np.searchsorted(cdf[j], r * cdf[j, -1], side='right')
                     for j, r in zip(interval, rng.random(n))], dtype=int)
    step = np.minimum(step, nenergy - 2)
    y0, y1 = integrand[interval, step], integrand[interval, step + 1]
    r = rng.random(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(np.abs(y1 - y0) > 1e-12 * np.maximum(y0, y1),
                     (np.sqrt(y0 ** 2 + r * (y1 ** 2 - y0 ** 2)) - y0) / (y1 - y0), r)
    photon_energy = energy[step] + t * (energy[step + 1] - energy[step])

    start, stop = obstime[:-1].utc.unix, obstime[1:].utc.unix
    unix = start[interval] + rng.random(n) * (stop - start)[interval]
    order = np.argsort(unix)
    unix, photon_energy = unix[order], photon_energy[order]
    time = Time(unix, format='unix', scale='utc')

    if n == 0:
        empty = np.zeros(0)
        return TimeTagEmCDSEventDataInSCFrameFromArrays(time.jd1, time.jd2, empty, empty, empty, empty), empty

    if direction_at == 'event':
        event_coord = sc_history.interp(time).get_target_in_sc_frame(coord)
        event_lon, event_lat = np.asarray(event_coord.lon.rad, dtype=float), np.asarray(event_coord.lat.rad, dtype=float)
    elif direction_at == 'midpoint':
        event_lon, event_lat = lon[interval[order]], lat[interval[order]]
    else:
        raise ValueError(f"direction_at must be 'event' or 'midpoint', got {direction_at!r}")

    events = irf.random_events(PhotonListWithDirectionAndEnergyInSCFrame(event_lon, event_lat, photon_energy))

    data = TimeTagEmCDSEventDataInSCFrameFromArrays(time.jd1, time.jd2,
                                                    *(np.asarray(a, dtype=float) for a in
                                                      (events.energy_keV, events.scattered_lon_rad_sc,
                                                       events.scattered_lat_rad_sc, events.scattering_angle_rad)))

    return data, photon_energy
