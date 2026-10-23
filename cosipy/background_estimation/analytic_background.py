from typing import Dict, Iterable, Tuple, Type

import numpy as np
from astropy import units as u
from astropy.time import Time

from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromArrays
from cosipy.interfaces.background_interface import BackgroundDensityInterface
from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import EventInterface, TimeTagEmCDSEventInSCFrameInterface
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.util.iterables import asarray


class IsotropicPowerLawBackground(BackgroundDensityInterface):
    """
    An ideal background with an analytic density, mostly to test the source models:
    a power law in measured energy, uniform in the Compton scattering angle ``Phi`` in
    ``[0, pi]``, isotropic in the scattered direction ``PsiChi`` in the spacecraft frame,
    and a constant rate in live time. Its only parameter is the rate (``label``, in Hz).

    The density of an event, per unit time and data space (``1 / s / keV / rad / sr``),
    is ``rate * f_live * p(Em) / pi / (4 pi)``, where ``f_live`` is the live time fraction
    of the spacecraft history interval of the event and ``p(Em)`` the normalized power
    law, and the expected counts are ``rate * livetime``.
    :meth:`random_events` draws events from this same model.

    Parameters
    ----------
    data : TimeTagEmCDSEventDataInSCFrameInterface
        Events. Their times must be within the spacecraft history.
    sc_history : SpacecraftHistory
        Spacecraft orientation and live time.
    rate : Quantity
        Initial value of the rate.
    index : float
        Index of the power law in measured energy.
    energy_range_keV : tuple of float
        Range of the power law, e.g. the energy selection of the data.
    label : str
        Name of the rate parameter.
    """

    event_data_type = TimeTagEmCDSEventDataInSCFrameInterface

    def __init__(self,
                 data: TimeTagEmCDSEventDataInSCFrameInterface,
                 sc_history: SpacecraftHistory,
                 rate: u.Quantity,
                 index: float = -2.,
                 energy_range_keV: Tuple[float, float] = (200., 5000.),
                 label: str = 'bkg_rate'):

        time = np.asarray(data.time.utc.unix, dtype=float)
        energy = asarray(data.energy_keV, dtype=float)
        intervals = self._intervals(time, sc_history)
        live_fraction = (sc_history.livetime / sc_history.intervals_duration).to_value('')[intervals]

        self._livetime = sc_history.cumulative_livetime().to_value(u.s)

        # Per unit time, per keV, rad and sr
        self._density_per_rate = (live_fraction * self.energy_density(energy, index, energy_range_keV)
                                  / np.pi / (4 * np.pi))

        self._label = label
        self._rate = rate.to_value(u.Hz)

    @staticmethod
    def _intervals(time: np.ndarray, sc_history: SpacecraftHistory) -> np.ndarray:
        """Index of the spacecraft history interval of each time (UTC unix seconds)."""

        if np.any((time < sc_history.tstart.utc.unix) | (time > sc_history.tstop.utc.unix)):
            raise ValueError("Event times are outside the spacecraft history range")

        return np.minimum(np.searchsorted(sc_history.obstime.utc.unix, time, side='right') - 1,
                          sc_history.nintervals - 1)

    @staticmethod
    def energy_density(energy_keV: np.ndarray, index: float, energy_range_keV: Tuple[float, float]) -> np.ndarray:
        """The power law normalized over the energy range, 1/keV, zero outside it."""

        emin, emax = energy_range_keV
        if index == -1:
            norm = np.log(emax / emin)
        else:
            norm = (emax ** (index + 1) - emin ** (index + 1)) / (index + 1)

        energy_keV = np.asarray(energy_keV, dtype=float)
        return np.where((energy_keV >= emin) & (energy_keV < emax), energy_keV ** index / norm, 0.)

    @property
    def event_type(self) -> Type[EventInterface]:
        return TimeTagEmCDSEventInSCFrameInterface

    @property
    def parameters(self) -> Dict[str, u.Quantity]:
        return {self._label: u.Quantity(self._rate, u.Hz)}

    def set_parameters(self, **parameters: u.Quantity) -> None:
        self._rate = parameters[self._label].to_value(u.Hz)

    def expected_counts(self) -> float:
        return self._rate * self._livetime

    def expectation_density(self) -> Iterable[float]:
        return self._rate * self._density_per_rate

    @staticmethod
    def random_events(sc_history: SpacecraftHistory,
                      rate: u.Quantity,
                      index: float = -2.,
                      energy_range_keV: Tuple[float, float] = (200., 5000.),
                      rng: np.random.Generator = None) -> TimeTagEmCDSEventDataInSCFrameFromArrays:
        """
        Events drawn from the model: a Poisson number of them with mean
        ``rate * livetime``, in each interval of ``sc_history`` in proportion to its live
        time and uniformly within it, sorted by time.
        """

        if rng is None:
            rng = np.random.default_rng()

        livetime = sc_history.livetime.to_value(u.s)
        counts = rng.poisson(rate.to_value(u.Hz) * livetime)
        interval = np.repeat(np.arange(livetime.size), counts)
        n = interval.size

        start, stop = sc_history.obstime[:-1].utc.unix, sc_history.obstime[1:].utc.unix
        time = np.sort(start[interval] + rng.random(n) * (stop - start)[interval])

        emin, emax = energy_range_keV
        x = rng.random(n)
        if index == -1:
            energy = emin * (emax / emin) ** x
        else:
            a, b = emin ** (index + 1), emax ** (index + 1)
            energy = (a + x * (b - a)) ** (1 / (index + 1))

        phi = rng.random(n) * np.pi
        lon = rng.random(n) * 2 * np.pi
        lat = np.arcsin(2 * rng.random(n) - 1)

        time = Time(time, format='unix', scale='utc')
        return TimeTagEmCDSEventDataInSCFrameFromArrays(time.jd1, time.jd2, energy, lon, lat, phi)
