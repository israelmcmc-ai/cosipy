from pathlib import Path
from typing import Dict, Iterable, Optional, Type, Union

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from histpy import Axes, Axis, Histogram
from scipy.ndimage import gaussian_filter1d
from scoords import SpacecraftFrame

from cosipy import SpacecraftHistory
from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.interfaces.background_interface import BackgroundDensityInterface
from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import EventInterface, TimeTagEmCDSEventInSCFrameInterface
from cosipy.util.iterables import asarray

__all__ = ["rocking_angle", "HistBackgroundTemplate", "FreeNormHistBackgroundDensity"]


def rocking_angle(sc_history: SpacecraftHistory) -> u.Quantity:
    """
    Rocking angle of the spacecraft at the start of each history interval.

    This is the signed tilt of the spacecraft +z axis from the Earth zenith,
    positive toward the North, i.e. toward the celestial pole projected on
    the local horizon plane.

    Notes
    -----
    The survey modes are separated by the user with fixed Rocking edges. For
    the real mission, the attitude history may need to be validated first.

    Parameters
    ----------
    sc_history : SpacecraftHistory

    Returns
    -------
    astropy.units.Quantity
        One angle in (-180, 180] deg per interval of `sc_history`.
    """

    # TODO: the survey modes are separated with fixed Rocking edges, which works for DC4
    # (exactly +-22 deg plus 1% slews). For the real mission the attitude history might be
    # more complicated (different rocking profiles, slews, safe modes...), so check that
    # it is consistent with the expected survey modes and remove the outliers.
    z_axis = sc_history.attitude.transform_to('galactic').as_axes()[2]
    z = z_axis.cartesian.xyz.value[:, :-1]
    zenith = sc_history.earth_zenith.cartesian.xyz.value[:, :-1]

    pole = SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame='icrs').galactic
    pole = pole.cartesian.xyz.value[:, None]

    north = pole - np.sum(pole * zenith, axis=0) * zenith
    north /= np.linalg.norm(north, axis=0)

    angle = np.arctan2(np.sum(z * north, axis=0), np.sum(z * zenith, axis=0))

    return u.Quantity(angle, u.rad).to(u.deg)


def _unitless(axis: Axis, unit: u.Unit) -> Axis:
    """Axis converted to `unit` and stripped of it. Unitless axes are assumed to be in `unit`."""
    if axis.unit is None:
        return axis
    return axis.to(unit).to(None, copy=False, update=False)


def _axis_values(axis: Axis, values: np.ndarray, unit: u.Unit):
    """`values` in `unit`, as a Quantity if the axis has units, to fill or evaluate a histogram."""
    return values if axis.unit is None else u.Quantity(values, unit)


def _check_sc_frame(psichi_axis):
    if not isinstance(psichi_axis.coordsys, SpacecraftFrame):
        raise NotImplementedError("Only templates with the PsiChi axis in the SpacecraftFrame are supported")


def _event_intervals(time: np.ndarray, sc_history: SpacecraftHistory) -> np.ndarray:
    """Index of the SpacecraftHistory interval that contains each event time (UTC unix seconds)."""

    if np.any((time < sc_history.tstart.utc.unix) | (time > sc_history.tstop.utc.unix)):
        raise ValueError("Event times are outside the spacecraft history range")

    intervals = np.searchsorted(sc_history.obstime.utc.unix, time, side='right') - 1

    return np.minimum(intervals, sc_history.nintervals - 1)


def _event_arrays(data: TimeTagEmCDSEventDataInSCFrameInterface):
    time = np.asarray(data.time.utc.unix, dtype=float)
    energy = asarray(data.energy_keV, dtype=float)
    phi = asarray(data.scattering_angle_rad, dtype=float)
    psichi = SkyCoord(asarray(data.scattered_lon_rad_sc, dtype=float),
                      asarray(data.scattered_lat_rad_sc, dtype=float),
                      unit=u.rad, frame=SpacecraftFrame())
    return time, energy, phi, psichi


def _gaussian_filter(array: np.ndarray, sigma: float, **kwargs) -> np.ndarray:
    """Gaussian filter, doing nothing for a negligible `sigma`, which gaussian_filter1d may not accept."""
    if sigma < 0.01:
        return array
    return gaussian_filter1d(array, sigma, **kwargs)


def _normalized_density(counts: np.ndarray, widths) -> np.ndarray:
    """Normalize over the last axis and divide by the bin widths. Empty slices are 0."""
    total = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, total * widths,
                     out=np.zeros(counts.shape, dtype=float), where=total > 0)


class HistBackgroundTemplate:
    """
    Raw-count histograms of the background, built directly from data, together
    with the livetime they were accumulated over.

    The histograms are independent from each other, so each can have its own
    binning. Axes without units are assumed to be in keV (``Em``), rad (``Phi``),
    deg (``Rocking``) and UTC unix seconds (``Time``).

    Parameters
    ----------
    rate_counts : histpy.Histogram
        Counts with axes ``Time``, ``Em``.
    livetime : histpy.Histogram
        Livetime in seconds, with the same ``Time`` axis as `rate_counts`.
    phi_counts : histpy.Histogram
        Counts with axes ``Rocking``, ``Em``, ``Phi``.
    psichi_counts : histpy.Histogram
        Counts with axes ``Rocking``, ``Em``, ``Phi``, ``PsiChi``. The
        ``PsiChi`` axis is a HealpixAxis whose ``coordsys`` sets the frame.
        Only the `SpacecraftFrame` is currently supported.
    """

    _names = {'rate_counts': 'RATE_COUNTS',
              'livetime': 'LIVETIME',
              'phi_counts': 'PHI_COUNTS',
              'psichi_counts': 'PSICHI_COUNTS'}

    _labels = {'rate_counts': ['Time', 'Em'],
               'livetime': ['Time'],
               'phi_counts': ['Rocking', 'Em', 'Phi'],
               'psichi_counts': ['Rocking', 'Em', 'Phi', 'PsiChi']}

    def __init__(self,
                 rate_counts: Histogram,
                 livetime: Histogram,
                 phi_counts: Histogram,
                 psichi_counts: Histogram):

        self.rate_counts = rate_counts.to_dense(copy=False)
        self.livetime = livetime.to_dense(copy=False)
        self.phi_counts = phi_counts.to_dense(copy=False)
        self.psichi_counts = psichi_counts.to_dense(copy=False)

        for name, labels in self._labels.items():
            if list(getattr(self, name).axes.labels) != labels:
                raise ValueError(f"The axes of '{name}' must be {labels}")

        if self.rate_counts.axes['Time'] != self.livetime.axes['Time']:
            raise ValueError("'rate_counts' and 'livetime' must have the same Time axis")

        if self.phi_counts.axes['Rocking'] != self.psichi_counts.axes['Rocking']:
            raise ValueError("'phi_counts' and 'psichi_counts' must have the same Rocking axis")

    @classmethod
    def empty(cls, rate_axes: Axes, phi_axes: Axes, psichi_axes: Axes) -> 'HistBackgroundTemplate':
        """
        Template with zeroed histograms.

        Parameters
        ----------
        rate_axes : histpy.Axes
            ``Time``, ``Em`` axes of the rate histogram. The livetime uses the ``Time`` axis.
        phi_axes : histpy.Axes
            ``Rocking``, ``Em``, ``Phi`` axes.
        psichi_axes : histpy.Axes
            ``Rocking``, ``Em``, ``Phi``, ``PsiChi`` axes.
        """

        return cls(Histogram(rate_axes),
                   Histogram(Axes([rate_axes['Time']])),
                   Histogram(phi_axes),
                   Histogram(psichi_axes))

    def _histograms(self) -> Dict[str, Histogram]:
        return {name: getattr(self, name) for name in self._names}

    def fill(self, data: TimeTagEmCDSEventDataInSCFrameInterface, sc_history: SpacecraftHistory) -> None:
        """
        Accumulate events and livetime. Call it once per chunk of data with the
        matching, non-overlapping, `sc_history`.

        Events outside the range of any axis are not counted, but the
        livetime of `sc_history` within the ``Time`` axis range always is.

        Parameters
        ----------
        data : TimeTagEmCDSEventDataInSCFrameInterface
        sc_history : SpacecraftHistory
            Must contain all the events.

        Raises
        ------
        ValueError
            If any event is outside `sc_history`.
        NotImplementedError
            If the PsiChi axis is not in the SpacecraftFrame.
        """

        self._accumulate(data, sc_history, 1)

    def remove(self, data: TimeTagEmCDSEventDataInSCFrameInterface, sc_history: SpacecraftHistory) -> None:
        """
        Inverse of `fill`: subtract the events and the livetime of `sc_history`.

        For example, to exclude the on-time window of a GRB from a template that
        was built with it. The rate of the affected time bins then comes from the
        rest of the bin, and time smoothing interpolates it from the neighboring times.

        The events and `sc_history` must be a subset of what was filled, with the same
        selections applied to the events. Tiny negative livetimes from float round-off
        are set to 0.

        Parameters
        ----------
        data : TimeTagEmCDSEventDataInSCFrameInterface
        sc_history : SpacecraftHistory
            Must contain all the events.
        """

        self._accumulate(data, sc_history, -1)

        self.livetime[...] = np.clip(self.livetime.contents, 0, None)

    def _accumulate(self, data, sc_history: SpacecraftHistory, sign: int) -> None:
        """Add (`sign` = 1) or subtract (`sign` = -1) the events and livetime."""

        _check_sc_frame(self.psichi_counts.axes['PsiChi'])

        time, energy, phi, psichi = _event_arrays(data)
        rocking = rocking_angle(sc_history).to_value(u.deg)[_event_intervals(time, sc_history)]

        for hist in (self.rate_counts, self.phi_counts, self.psichi_counts):
            values = {'Time': time,
                      'Rocking': rocking,
                      'Em': energy,
                      'Phi': phi,
                      'PsiChi': psichi}
            units = {'Rocking': u.deg, 'Em': u.keV, 'Phi': u.rad}
            hist.fill(*(_axis_values(axis, values[axis.label], units.get(axis.label))
                        for axis in hist.axes), weight=sign, warn_overflow=False)

        edges = np.clip(self.livetime.axis.edges,
                        sc_history.tstart.utc.unix, sc_history.tstop.utc.unix)
        cumulative = sc_history.cumulative_livetime(Time(edges, format='unix')).to_value(u.s)
        self.livetime[...] = self.livetime.contents + sign * np.diff(cumulative)

    def smooth(self,
               psichi_fwhm: Optional[u.Quantity] = None,
               time_fwhm: Optional[u.Quantity] = None,
               phi_fwhm: Optional[u.Quantity] = None,
               time_counts: Optional[float] = None) -> 'HistBackgroundTemplate':
        """
        Smoothed copy of the template.

        Parameters
        ----------
        psichi_fwhm : astropy.units.Quantity, optional
            FWHM of a Gaussian on the sphere, applied to every
            (``Rocking``, ``Em``, ``Phi``) map of `psichi_counts`. Negative
            values are clipped to 0 and the total of each map is preserved.
        phi_fwhm : astropy.units.Quantity, optional
            FWHM of a Gaussian along the ``Phi`` axis of `psichi_counts`, i.e.
            across neighboring ``Phi`` bins. It is applied before `psichi_fwhm`.
            The ``Phi`` bins must be uniform.
        time_fwhm : astropy.units.Quantity, optional
            FWHM of a Gaussian along ``Time``, the same for all energies. It is
            applied to the rate and to `livetime`, such that the rate stays
            consistent across gaps. The ``Time`` bins must be uniform.
        time_counts : float, optional
            Alternative to `time_fwhm` that adapts to the statistics: the FWHM of
            each ``Em`` bin of `rate_counts` is `time_counts` divided by its mean count
            rate over the whole template, i.e. about `time_counts` counts per FWHM. High
            energies get long integration times, and low energies short ones.

        Notes
        -----
        The smoothed rate of an ``Em`` bin is the smoothed counts over the smoothed
        livetime, both with the kernel of that bin. The new livetime is smoothed with the
        shortest kernel, and the new counts are the smoothed rate times the new livetime.
        The rate of a bin without livetime is filled only if the kernel reaches livetime.

        Returns
        -------
        HistBackgroundTemplate
        """

        if time_fwhm is not None and time_counts is not None:
            raise ValueError("Use either time_fwhm or time_counts, not both")

        new = HistBackgroundTemplate(*(hist.copy() for hist in self._histograms().values()))

        sigma_per_fwhm = 1 / (2 * np.sqrt(2 * np.log(2)))

        if phi_fwhm is not None:
            widths = _unitless(new.psichi_counts.axes['Phi'], u.rad).widths
            if not np.allclose(widths, widths[0]):
                raise ValueError("Phi smoothing requires uniform Phi bins")

            sigma = u.Quantity(phi_fwhm).to_value(u.rad) * sigma_per_fwhm / widths[0]
            new.psichi_counts[...] = _gaussian_filter(new.psichi_counts.contents, sigma, axis=2, mode='nearest')

        if psichi_fwhm is not None:
            axis = new.psichi_counts.axes['PsiChi']
            counts = new.psichi_counts.contents
            fwhm = u.Quantity(psichi_fwhm).to_value(u.rad)

            for index in np.ndindex(counts.shape[:-1]):
                total = counts[index].sum()
                if total == 0:
                    continue

                pixels = hp.reorder(counts[index], n2r=True) if axis.is_nested else counts[index]
                smoothed = np.clip(hp.smoothing(pixels, fwhm=fwhm), 0, None)
                smoothed *= total / smoothed.sum()
                counts[index] = hp.reorder(smoothed, r2n=True) if axis.is_nested else smoothed

        if time_fwhm is not None or time_counts is not None:
            widths = new.livetime.axis.widths
            if not np.allclose(widths, widths[0]):
                raise ValueError("Time smoothing requires uniform Time bins")

            nem = new.rate_counts.shape[1]

            if time_fwhm is not None:
                fwhm = np.full(nem, u.Quantity(time_fwhm).to_value(u.s))
            else:
                mean_rate = new.rate_counts.contents.sum(axis=0) / new.livetime.contents.sum()
                with np.errstate(divide='ignore'):
                    fwhm = np.minimum(time_counts / mean_rate, widths.sum())

            new._smooth_time(fwhm * sigma_per_fwhm / widths[0])

        return new

    def _smooth_time(self, sigma: np.ndarray) -> None:
        """Smooth the rate and livetime in place, with `sigma` (in ``Time`` bins) for each ``Em`` bin."""

        counts = self.rate_counts.contents
        livetime = self.livetime.contents

        new_livetime = _gaussian_filter(livetime, sigma.min(), axis=0, mode='constant')
        new_counts = np.zeros_like(counts)

        for i, sigma_i in enumerate(sigma):
            smoothed_livetime = _gaussian_filter(livetime, sigma_i, axis=0, mode='constant')
            smoothed_counts = _gaussian_filter(counts[:, i], sigma_i, axis=0, mode='constant')
            rate = np.divide(smoothed_counts, smoothed_livetime,
                             out=np.zeros_like(smoothed_counts), where=smoothed_livetime > 0)
            new_counts[:, i] = rate * new_livetime

        self.rate_counts[...] = new_counts
        self.livetime[...] = new_livetime

    def __iadd__(self, other: 'HistBackgroundTemplate') -> 'HistBackgroundTemplate':
        for name, hist in self._histograms().items():
            if hist.axes != getattr(other, name).axes:
                raise ValueError(f"The axes of '{name}' differ")

        for name, hist in self._histograms().items():
            hist[...] = hist.contents + getattr(other, name).contents

        return self

    def __add__(self, other: 'HistBackgroundTemplate') -> 'HistBackgroundTemplate':
        new = HistBackgroundTemplate(*(hist.copy() for hist in self._histograms().values()))
        new += other
        return new

    def write(self, filename, overwrite: bool = False) -> None:
        """
        Write the template to a HDF5 file, one group per histogram.

        Parameters
        ----------
        filename : str or Path
        overwrite : bool
            Replace the file if it exists. Otherwise raise.
        """

        filename = Path(filename)

        if filename.exists():
            if not overwrite:
                raise FileExistsError(f"{filename} already exists. Use overwrite=True.")
            filename.unlink()

        for name, hist in self._histograms().items():
            hist.write(filename, name=self._names[name])

    @classmethod
    def open(cls, filename) -> 'HistBackgroundTemplate':
        """Read a template written with `write`."""

        return cls(*(Histogram.open(filename, name=name) for name in cls._names.values()))


class FreeNormHistBackgroundDensity(BackgroundDensityInterface):
    """
    Unbinned background density built from `HistBackgroundTemplate` objects,
    with a free normalization per component.

    For an event at time ``t`` with measured energy ``Em``, Compton angle
    ``phi`` and scattered direction ``PsiChi``, the density of a component is
    ``r(t, Em) * f_live(t) * p(phi | Em, k) * p(PsiChi | Em, phi, k)``, where
    ``k`` is the rocking bin at ``t``. The rate ``r`` is piecewise constant in
    time and interpolated in ``Em``, and the shapes ``p`` are interpolated.
    The density is normalized to the counts predicted by the template in the
    analysis window, such that each normalization is the average rate (Hz) of
    the selected events over the livetime of `sc_history`. Its default is the
    template prediction.

    Parameters
    ----------
    data : TimeTagEmCDSEventDataInSCFrameInterface
        Events, assumed to be already selected.
    sc_history : SpacecraftHistory
        Spacecraft history of the analysis window.
    template : HistBackgroundTemplate or dict of them
        A dictionary defines one component per label. A single template
        has the label ``bkg_norm``.
    energy_selection : EnergySelector, optional
        The selection applied to `data`, needed for the predicted counts.
        By default there is no cut.

    Raises
    ------
    NotImplementedError
        If the PsiChi axis of a template is not in the SpacecraftFrame.
    ValueError
        If the analysis window is outside the template time range, or has
        livetime in a template time bin without livetime, or if events are
        outside `sc_history` or the Rocking axis.
    """

    _default_label = 'bkg_norm'

    def __init__(self,
                 data: TimeTagEmCDSEventDataInSCFrameInterface,
                 sc_history: SpacecraftHistory,
                 template: Union[HistBackgroundTemplate, Dict[str, HistBackgroundTemplate]],
                 energy_selection: Optional[EnergySelector] = None):

        if isinstance(template, HistBackgroundTemplate):
            template = {self._default_label: template}

        if len(template) == 0:
            raise ValueError("You need to input at least one component")

        self._labels = tuple(template)

        if energy_selection is None:
            energy_selection = EnergySelector()

        time, energy, phi, psichi = _event_arrays(data)
        intervals = _event_intervals(time, sc_history)
        rocking = rocking_angle(sc_history).to_value(u.deg)[intervals]
        f_live = (sc_history.livetime / sc_history.intervals_duration).to_value(u.dimensionless_unscaled)[intervals]

        self._livetime = sc_history.cumulative_livetime().to_value(u.s)

        counts = np.empty(len(self._labels))
        self._prob = np.empty((len(self._labels), len(time)))

        for i, component in enumerate(template.values()):
            _check_sc_frame(component.psichi_counts.axes['PsiChi'])

            density = (self._rate(component, time, energy) * f_live
                       * self._phi_density(component, rocking, energy, phi)
                       * self._psichi_density(component, rocking, energy, phi, psichi))

            counts[i] = self._predicted_counts(component, sc_history, energy_selection)

            if counts[i] <= 0:
                raise ValueError("The template predicts no counts in the analysis window")

            self._prob[i] = density / counts[i]

        self._norms = counts / self._livetime

    @property
    def event_type(self) -> Type[EventInterface]:
        return TimeTagEmCDSEventInSCFrameInterface

    @property
    def labels(self):
        return self._labels

    @property
    def parameters(self) -> Dict[str, u.Quantity]:
        return {label: u.Quantity(norm, u.Hz) for label, norm in zip(self._labels, self._norms)}

    def set_parameters(self, **parameters: u.Quantity) -> None:
        for label in parameters:
            if label not in self._labels:
                raise ValueError(f"Parameter {label} not in {self._labels}")

        for label, norm in parameters.items():
            self._norms[self._labels.index(label)] = norm.to_value(u.Hz)

    def expected_counts(self) -> float:
        return np.sum(self._norms) * self._livetime

    def expectation_density(self) -> Iterable[float]:
        return np.tensordot(self._norms * self._livetime, self._prob, axes=(0, 0))

    @staticmethod
    def _rate_per_keV(component: HistBackgroundTemplate) -> np.ndarray:
        """Rate in counts/s/keV for each (Time, Em) bin, 0 where there is no livetime."""

        livetime = component.livetime.contents[:, None]
        widths = _unitless(component.rate_counts.axes['Em'], u.keV).widths[None, :]

        return np.divide(component.rate_counts.contents, livetime * widths,
                         out=np.zeros(component.rate_counts.shape), where=livetime > 0)

    @staticmethod
    def _time_bins(component: HistBackgroundTemplate, time: np.ndarray) -> np.ndarray:
        """Time bin of each time, raising if outside the axis range or without livetime."""

        edges = component.livetime.axis.edges

        if np.any((time < edges[0]) | (time > edges[-1])):
            raise ValueError("The analysis window is outside the template time range")

        bins = np.minimum(np.searchsorted(edges, time, side='right') - 1, len(edges) - 2)

        if np.any(component.livetime.contents[bins] <= 0):
            raise ValueError("The analysis window has livetime in a template time bin without livetime")

        return bins

    @classmethod
    def _rate(cls, component: HistBackgroundTemplate, time: np.ndarray, energy: np.ndarray) -> np.ndarray:
        """Rate in counts/s/keV at each time and energy"""

        rate = cls._rate_per_keV(component)
        bins, weights = _unitless(component.rate_counts.axes['Em'], u.keV).interp_weights(energy)
        time_bins = cls._time_bins(component, time)

        return rate[time_bins, bins[0]] * weights[0] + rate[time_bins, bins[1]] * weights[1]

    @staticmethod
    def _predicted_counts(component: HistBackgroundTemplate,
                          sc_history: SpacecraftHistory,
                          energy_selection: EnergySelector) -> float:

        livetime = sc_history.livetime.to_value(u.s)
        active = livetime > 0

        obstime = sc_history.obstime.utc.unix
        mid_time = (obstime[:-1] + obstime[1:]) / 2
        time_bins = FreeNormHistBackgroundDensity._time_bins(component, mid_time[active])

        edges = _unitless(component.rate_counts.axes['Em'], u.keV).edges
        ranges = energy_selection.energy_ranges_keV
        overlap = np.clip(np.minimum(edges[1:, None], ranges[None, :, 1])
                          - np.maximum(edges[:-1, None], ranges[None, :, 0]),
                          0, None).sum(axis=1)

        rate = FreeNormHistBackgroundDensity._rate_per_keV(component)

        return float(np.sum(livetime[active] * (rate[time_bins] @ overlap)))

    @staticmethod
    def _rocking_bins(component: HistBackgroundTemplate, rocking: np.ndarray) -> np.ndarray:
        edges = _unitless(component.phi_counts.axes['Rocking'], u.deg).edges

        if np.any((rocking < edges[0]) | (rocking > edges[-1])):
            raise ValueError("Some events have a rocking angle outside the template Rocking axis")

        return np.minimum(np.searchsorted(edges, rocking, side='right') - 1, len(edges) - 2)

    @classmethod
    def _phi_density(cls, component, rocking, energy, phi) -> np.ndarray:
        """p(phi | Em, k) in 1/rad"""

        axes = [_unitless(component.phi_counts.axes['Em'], u.keV),
                _unitless(component.phi_counts.axes['Phi'], u.rad)]

        pdf = _normalized_density(component.phi_counts.contents, axes[1].widths)
        rocking_bins = cls._rocking_bins(component, rocking)

        result = np.empty(len(energy))
        for k in range(pdf.shape[0]):
            sel = rocking_bins == k
            result[sel] = Histogram(axes, contents=pdf[k]).interp(energy[sel], phi[sel])

        return result

    @classmethod
    def _psichi_density(cls, component, rocking, energy, phi, psichi) -> np.ndarray:
        """p(PsiChi | Em, phi, k) in 1/sr"""

        psichi_axis = component.psichi_counts.axes['PsiChi']
        axes = [_unitless(component.psichi_counts.axes['Em'], u.keV),
                _unitless(component.psichi_counts.axes['Phi'], u.rad),
                psichi_axis]

        pdf = _normalized_density(component.psichi_counts.contents, psichi_axis.pixarea().to_value(u.sr))
        rocking_bins = cls._rocking_bins(component, rocking)

        result = np.empty(len(energy))
        for k in range(pdf.shape[0]):
            sel = rocking_bins == k
            result[sel] = Histogram(axes, contents=pdf[k]).interp(energy[sel], phi[sel], psichi[sel])

        return result
