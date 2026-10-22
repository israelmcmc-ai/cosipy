import functools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np
from astropy import units as u
from astropy.coordinates import spherical_to_cartesian, UnitSphericalRepresentation
from astropy.io.fits import update
from astropy.units import Quantity
from histpy import Histogram, HealpixAxis, Axis, Axes
from mhealpy.plot.axes import HealpyAxes

from cosipy.interfaces import EventDataInterface
from cosipy.interfaces.data_interface import EmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import EmCDSEventInSCFrameInterface
from cosipy.interfaces.event_selection import EventSelectorInterface
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.interfaces.photon_parameters import PhotonWithDirectionAndEnergyInSCFrameInterface, PhotonListInterface, \
    PhotonListWithDirectionInSCFrameInterface, PhotonListWithDirectionAndEnergyInSCFrameInterface

import h5py as h5

from scoords import SpacecraftFrame

from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.polarization import PolarizationAxis
from cosipy.response.relative_coordinates import RelativeCDSCoordinates
from cosipy.util.iterables import asarray


class _Hats:
    """
    The hat functions that ``histpy.Histogram.interp()`` weights the bin
    centers of a linear-scale axis with, as piecewise-linear functions on the
    knots ``[lower edge, centers..., upper edge]``.

    Between centers, the hat of center ``i`` is 1 at it and 0 at the others.
    Between the outer centers and the edges it depends on ``edges``:

    - ``'clamp'`` (the ``interp()`` default): the outer hats are 1, and so
      beyond the edges.
    - ``'zero'`` (``fill_value=0, extrapolate=True``): the outer hats ramp
      linearly to 0 at the edges, and are 0 beyond them.
    - ``'periodic'`` (``fill_value='periodic'``): the last center is followed
      by the first one, one period (the axis range) later.

    Parameters
    ----------
    axis : histpy.Axis
        A linear-scale axis.
    edges : {'clamp', 'zero', 'periodic'}
    """

    def __init__(self, axis: Axis, edges: str):

        centers = np.asarray(axis.centers, dtype=float)
        lo, hi = float(axis.edges[0]), float(axis.edges[-1])
        n = centers.size

        if edges == 'clamp':
            lo_values, hi_values = np.eye(n)[0], np.eye(n)[-1]
        elif edges == 'zero':
            lo_values = hi_values = np.zeros(n)
        elif edges == 'periodic':
            # The edges are between the last center, one period earlier, and the first one
            prev_last = centers[-1] - (hi - lo)
            t = (lo - prev_last) / (centers[0] - prev_last)
            lo_values = hi_values = np.zeros(n)
            lo_values[0] += t
            lo_values[-1] += 1 - t
        else:
            raise ValueError(f"edges must be 'clamp', 'zero' or 'periodic', got {edges!r}")

        self.edges = edges
        self.knots = np.concatenate(([lo], centers, [hi]))
        self.values = np.vstack((lo_values, np.eye(n), hi_values))  # (knot, hat)

        segment_integrals = 0.5 * (self.values[1:] + self.values[:-1]) * np.diff(self.knots)[:, None]
        self._cumulative_at_knots = np.vstack((np.zeros(n), np.cumsum(segment_integrals, axis=0)))

    def _segments(self, x):
        x = np.clip(np.asarray(x, dtype=float), self.knots[0], self.knots[-1])
        j = np.clip(np.searchsorted(self.knots, x, side='right') - 1, 0, self.knots.size - 2)
        return j, x - self.knots[j]

    def __call__(self, x) -> np.ndarray:
        """
        Value of each hat at ``x`` (within the axis range), shape ``x.shape + (ncenters,)``.
        """

        j, dx = self._segments(x)
        t = (dx / np.diff(self.knots)[j])[..., None]

        return (1 - t) * self.values[j] + t * self.values[j + 1]

    def integrals(self) -> np.ndarray:
        """
        Integral of each hat over the axis range, shape ``(ncenters,)``.
        """

        return self._cumulative_at_knots[-1]

    def cumulative(self, x) -> np.ndarray:
        """
        Integral of each hat from the lower edge to ``x`` (clipped to the axis
        range), shape ``x.shape + (ncenters,)``. Exact, since the hats are
        piecewise linear.
        """

        j, dx = self._segments(x)
        slope = np.diff(self.values, axis=0) / np.diff(self.knots)[:, None]
        dx = dx[..., None]

        return self._cumulative_at_knots[j] + self.values[j] * dx + 0.5 * slope[j] * dx ** 2


class IRFRelativeHistUnpolarized(FarFieldSpectralInstrumentResponseFunctionInterface):
    """
    Histogram-based far-field instrument response function parametrized in
    coordinates relative to the photon and linearly interpolated.

    The response is stored as a 6-dimensional :class:`histpy.Histogram`
    with axes, in order:

    - ``NuLambda`` (:class:`histpy.HealpixAxis`): incoming photon
      direction in the spacecraft frame.
    - ``Ei`` (:class:`histpy.Axis`, units of energy): true incident
      photon energy.
    - ``Epsilon`` (:class:`histpy.Axis`, unitless): fractional energy
      deviation ``(Em - Ei) / Ei`` between the measured energy ``Em``
      and the true energy ``Ei``.
    - ``Phi`` (:class:`histpy.Axis`, units of angle): Compton
      kinematics-derived scattering angle.
    - ``Theta`` (:class:`histpy.Axis`, units of angle): offset of the
      geometric scattering angle (NuLambda to PsiChi) relative to the kinematic one
      (``Theta = phi_geo - phi_kin``).
    - ``Zeta`` (:class:`cosipy.polarization.PolarizationAxis`):
      azimuthal angle of the scattered photon around the source
      direction, referenced to the polarization convention stored in
      the axis.

    On construction the input histogram is validated, its axis units are
    standardized (energies to keV, angles to radians, contents to cm^2), and
    it is converted to the event probability density ``_event_prob``: the
    probability of an event (``Em``, ``Phi``, ``PsiChi``) given the photon
    (``NuLambda``, ``Ei``). The total effective area of the photon (``aeff``,
    or the ``NuLambda``/``Ei`` projection of ``irf``) is kept separately, so
    the differential effective area is that area times the event probability.
    The polarization convention carried by the ``Zeta`` axis is preserved as
    :attr:`_pol_convention` so it can be re-applied when translating event
    directions into the relative coordinate system.

    The density is multilinearly interpolated between bin centers, and it is
    normalized so that the interpolated density, not the bin contents,
    integrates to one over the data space for any photon:

    - Each bin is divided by its phase space and each ``(NuLambda, Ei)`` slice
      by the integral of its interpolant, with the true measure of the data
      space ``sin(phi + theta) dphi dtheta dzeta dEpsilon``. ``Zeta`` and
      ``Epsilon`` are integrated exactly. ``Phi`` and ``Theta`` are split in
      ``_CDS_SUBDIVISIONS`` sub-bins, which brings the error to ~1e-4.
    - The table is a density per unit ``Epsilon`` and it is divided by the
      photon ``Ei`` when evaluated, since ``dEm = Ei dEpsilon``. The
      interpolation between ``Ei`` centers happens at fixed ``Epsilon``, so
      each ``Ei`` slice integrates to one and so does any interpolation between
      them.
    - Between the outer ``Epsilon`` centers and the edges, the density ramps
      linearly to zero, and it is zero beyond them (``fill_value=0``). ``Zeta``
      is periodic: values are wrapped, and the density is interpolated between
      the last and first bins (``fill_value='periodic'``). The other axes are
      clamped to the outer centers. Photons outside the range of the ``Ei``
      axis get zero area and probability.

    This class is labelled "Unpolarized" because it does not model a
    dependence of the response on the incident photon polarization
    angle; the ``Zeta`` axis is used purely as the scattered-photon
    azimuth, not as a polarization-input degree of freedom.

    Parameters
    ----------
    irf : histpy.Histogram, str or path-like
        A 6D histogram with the axes described above and contents in
        units equivalent to area (``cm^2``), or an HDF5 file with that
        histogram in the group ``group`` (default ``'IRF'``), in the dense
        format. From a file, each ``NuLambda`` pixel is read and normalized
        the first time an evaluation needs it (unless ``preload``), so only
        the pixels around the photon directions take memory. The file is
        opened for each read. With ``selections``, only the ``Ei`` bins that
        can produce a measured energy inside them are read (see
        :meth:`_open_h5_axes`); photons outside those bins get zero area and
        probability, which is what the selection gives them anyway.
    aeff : histpy.Histogram, optional
        A separate 2D histogram, with axes ``['NuLambda', 'Ei']`` and
        contents in units equivalent to area (``cm^2``), used as the
        total effective area instead of projecting it out of ``irf``.
        Its ``NuLambda``/``Ei`` binning does not need to match that of
        ``irf`` -- e.g. it can use a finer grid, since the total
        effective area is typically cheaper to compute/store at higher
        resolution than the full differential response. If not
        provided (the default), the total effective area is obtained
        by projecting ``irf`` onto its own ``NuLambda``/``Ei`` axes. The
        event probability does not depend on it, since ``irf`` is
        normalized.
    copy : bool, optional
        If True (default) the input histogram(s) are copied before
        their axes and contents are modified in place. Set to False to
        avoid the copy when the caller no longer needs the original(s).
    nthreads : int, optional
        Number of worker threads used to parallelize
        :meth:`effective_area_cm2`/:meth:`differential_effective_area_cm2`
        over large photon/event lists. Defaults to ``1`` (no thread pool,
        no behavior change from before this parameter existed). The
        underlying ``histpy.Histogram.interp()`` call is vectorized numpy
        (fancy indexing + reduction) that releases the GIL for large
        inputs, so values > 1 can give a real speedup on multi-core
        machines -- but only once there's enough work per thread to be
        worth the fan-out/synchronization cost, see
        ``npoints_parallel_thresh``.
        NOTE: the bottleneck is the searchsorted algorithm, which is limited by L3 memory.
        You'll get diminishing returns beyond 5-10 threads.
    npoints_parallel_thresh : int, optional
        Minimum number of points per thread for parallelization to be
        used. Below ``nthreads * npoints_parallel_thresh`` points,
        :meth:`effective_area_cm2`/:meth:`differential_effective_area_cm2`
        call ``interp()`` directly, single-threaded, regardless of
        ``nthreads`` -- fanning small workloads out across threads costs
        more than it saves. Defaults to ``1000``; only matters when
        ``nthreads`` > 1.
    selections : cosipy.event_selection.EnergySelector or tuple thereof, optional
        Measured-energy cut(s), so the effective area matches an equivalent
        cut already applied to the event data being fit (e.g. via the same
        ``EnergySelector``). The selected effective area of a photon is its
        unselected area times the fraction of its event probability with
        ``Em`` inside the selection. The fraction is computed per photon from
        the same interpolant as the density (same ``NuLambda`` and ``Ei``
        weights, and the exact integral of the ``Epsilon`` hat functions over
        the cut), so the selected area is the integral of the differential
        effective area over the selected ``Em`` range, for any cut width.
        The event probability is divided by the fraction to make it
        conditional on passing the cut; the differential effective area is
        not. Only ``EnergySelector`` is currently supported; anything else
        raises ``TypeError``. If a tuple is given, the cuts combine with OR
        semantics (unioned via ``EnergySelector.union``) -- i.e. each entry
        is treated as an acceptable energy window, and an event/bin passes
        if it falls in any of them. For AND semantics, pass a single
        pre-intersected ``EnergySelector`` (e.g. ``sel1.intersect(sel2)``)
        instead of a tuple. Defaults to ``None`` (no cut).
    preload : bool, optional
        Only when ``irf`` is a file: read and normalize all the pixels on
        construction instead of on demand, e.g. when most of the sky will be
        evaluated anyway. Defaults to False.
    group : str, optional
        Only when ``irf`` is a file: the HDF5 group with the histogram.
        Defaults to ``'IRF'``.
    """

    event_data_type = EmCDSEventDataInSCFrameInterface
    photon_list_type = PhotonListWithDirectionAndEnergyInSCFrameInterface

    _CDS_SUBDIVISIONS = 32

    # interp() fill_value per axis: NuLambda, Ei, Epsilon, Phi, Theta, Zeta
    _FILL_VALUE = [None, None, 0, None, None, 'periodic']

    def __init__(self,
                 irf: Union[Histogram, str, Path],
                 aeff: Histogram = None,
                 copy = True,
                 nthreads = 1,
                 npoints_parallel_thresh = 1000,
                 selections: Union[EventSelectorInterface, Tuple[EventSelectorInterface, ...]] = None,
                 preload: bool = False,
                 group: str = 'IRF'):
        """
        Validate the input histogram(s), standardize their axis units,
        and set up the total effective area and the normalized event
        probability used at evaluation time.

        See the class docstring for a description of the expected axes
        and units.

        A ``Histogram`` is normalized in place, one ``NuLambda`` pixel at a
        time, without full-size temporary arrays. From an HDF5 file, each
        pixel is read and normalized the same way the first time it is needed
        (or all of them here, with ``preload``).

        Parameters
        ----------
        irf : histpy.Histogram, str or path-like
            Input response histogram, or the HDF5 file that holds it.
        aeff : histpy.Histogram, optional
            Optional separate total effective area histogram. See the
            class docstring.
        copy : bool, optional
            Whether to copy ``irf``/``aeff`` before modifying them.
        nthreads : int, optional
            Number of worker threads for parallel interpolation. See the
            class docstring.
        npoints_parallel_thresh : int, optional
            Minimum points per thread before parallelizing. See the
            class docstring.
        selections : cosipy.event_selection.EnergySelector or tuple thereof, optional
            Measured-energy cut(s). See the class docstring.
        preload : bool, optional
            Only for an HDF5 file. See the class docstring.
        group : str, optional
            Only for an HDF5 file: the group that holds the histogram.

        Raises
        ------
        ValueError
            If the histogram contents are not area-equivalent, the
            axis labels do not match the expected sequence, or an
            axis has an unexpected type or units.
        """

        combined_selector = self._combine_selections(selections)

        if isinstance(irf, Histogram):
            if copy:
                irf = irf.copy()
            axes, unit = irf.axes, irf.unit
            self._source = None
        else:
            axes, unit, ei_bins = self._open_h5_axes(irf, group, combined_selector)
            self._source = (irf, group, ei_bins)

        self._standardize_axes(axes, unit)

        # Get the total effective area. Projected from irf, it is filled per pixel when loaded.
        if aeff is not None:
            self._tot_aeff = self._standardize_aeff(aeff, copy) # cm^2
        else:
            self._tot_aeff = Histogram(Axes([axes['NuLambda'], axes['Ei']]))

        self._aeff_from_irf = aeff is None

        self._epsilon_hats = _Hats(axes['Epsilon'], 'zero')

        # Phase space of the (Phi, Theta, Zeta) bins
        phi_edges_mesh, arm_edges_mesh, az_edges_mesh = np.meshgrid(axes['Phi'].edges,
                                                                    axes['Theta'].edges,
                                                                    axes['Zeta'].edges, indexing='ij')

        self._phase_space_cds = RelativeCDSCoordinates.get_relative_cds_phase_space(phi_edges_mesh[:-1, :-1, :-1],
                                                                                    phi_edges_mesh[1:, :-1, :-1],
                                                                                    arm_edges_mesh[:-1, :-1, :-1],
                                                                                    arm_edges_mesh[:-1, 1:, :-1],
                                                                                    az_edges_mesh[:-1, :-1, :-1],
                                                                                    az_edges_mesh[:-1, :-1, 1:])

        self._cds_weights = self._cds_marginal_weights(axes)

        npix = axes['NuLambda'].nbins
        self._epsilon_marginal = np.zeros((npix, axes['Ei'].nbins, axes['Epsilon'].nbins))
        self._loaded = np.zeros(npix, dtype=bool)

        if self._source is None:
            # In place, so _event_prob shares the contents with the input
            irf = irf.to(u.cm * u.cm, copy=False).to(None, copy=False, update=False)
            self._event_prob = irf

            for pixel in range(npix):
                self._normalize_pixel(pixel, irf.contents[pixel])
        else:
            # The pages are only committed to memory as pixels are loaded
            self._area_unit_factor = unit.to(u.cm * u.cm)
            self._event_prob = Histogram(axes, contents=np.zeros(axes.nbins), copy_contents=False)

            if preload:
                self._load_pixels(range(npix))

        self._selection_ranges = (None if combined_selector is None
                                  else np.asarray(combined_selector.energy_ranges_keV, dtype=float))

        # Extra params
        self._nthreads = nthreads
        self._npoints_parallel_thresh = npoints_parallel_thresh
        self._executor = ThreadPoolExecutor(max_workers=nthreads) if nthreads > 1 else None

    def _standardize_axes(self, axes: Axes, unit):
        """
        Validate the axes and contents unit of the response, and standardize the
        axes in place: energies in keV, angles in rad, without units.
        """

        if unit is None or not unit.is_equivalent('cm^2'):
            raise ValueError("IRF contents are expected to have units of area.")

        if not np.array_equal(axes.labels, ['NuLambda', 'Ei', 'Epsilon', 'Phi', 'Theta', 'Zeta']):
            raise ValueError("IRF axes label must be ['NuLambda', 'Ei', 'Epsilon', 'Phi', 'Theta', 'Zeta']")

        if not isinstance(axes['NuLambda'], HealpixAxis):
            raise ValueError("IRF NuLambda axis is expected to be of HealpixAxis type")

        if axes['Ei'].unit is None or not axes['Ei'].unit.is_equivalent('keV'):
            raise ValueError("Ei axis is expected to have units of energy.")

        if axes['Epsilon'].unit is not None and not axes['Epsilon'].unit.is_equivalent(''):
            raise ValueError("Epsilon axis is expected to be unitless")

        if axes['Phi'].unit is None or not axes['Phi'].unit.is_equivalent('deg'):
            raise ValueError("Phi axis is expected to have units of angle.")

        if axes['Theta'].unit is None or not axes['Theta'].unit.is_equivalent('deg'):
            raise ValueError("Theta axis is expected to have units of angle.")

        if not isinstance(axes['Zeta'], PolarizationAxis):
            raise ValueError("IRF Zeta axis is expected to be of PolarizationAxis type")

        if not isinstance(axes['Zeta'].convention.frame, SpacecraftFrame):
            raise ValueError("IRF Zeta axis polarization convention must be defined in the spacecraft "
                              "frame (e.g. MEGAlibRelativeX/Y/Z or StereographicConvention).")

        # Zeta is interpolated as periodic, with a period equal to the axis range
        zeta_edges_deg = axes['Zeta'].edges.angle.to_value(u.deg)
        if not (np.isclose(zeta_edges_deg[0], 0) and np.isclose(zeta_edges_deg[-1], 360)):
            raise ValueError("IRF Zeta axis is expected to span the full [0, 360) deg range, got "
                              f"[{zeta_edges_deg[0]}, {zeta_edges_deg[-1]}] deg")

        for label in ['Epsilon', 'Phi', 'Theta', 'Zeta']:
            if axes[label].axis_scale != 'linear':
                raise ValueError(f"IRF {label} axis is expected to have a linear scale")

        axes['Ei'] = axes['Ei'].to(u.keV, copy = False).to(None, update = False, copy = False)
        axes['Epsilon'] = axes['Epsilon'].to(None, update = False, copy = False)
        axes['Phi'] = axes['Phi'].to(u.rad, copy = False).to(None, update = False, copy = False)
        axes['Theta'] = axes['Theta'].to(u.rad, copy = False).to(None, update = False, copy = False)
        self._pol_convention = axes['Zeta'].convention
        axes['Zeta'] = Axis(axes['Zeta'].edges.angle.to(u.rad).value, label = 'Zeta')

    @staticmethod
    def _open_h5_axes(filename, group: str, selector: Union[EnergySelector, None]):
        """
        Axes and unit of the histogram in ``group`` of an HDF5 file, and the
        slice of ``Ei`` bins to load.

        With a ``selector``, only the ``Ei`` bins that can produce a measured
        energy inside it are needed: a bin ``[a, b)`` produces ``Em`` between
        ``a (1 + eps_min)`` and ``b (1 + eps_max)``, with ``eps_min``/``eps_max``
        the edges of the ``Epsilon`` axis. One more bin is kept on each side,
        so that the photons in the needed range are interpolated as with the
        full axis.

        Raises
        ------
        ValueError
            If the histogram is not in the dense format without overflow bins.
        """

        with h5.File(filename, 'r') as f:
            g = f[group]

            if g.attrs.get('format', 'dense') != 'dense':
                raise ValueError(f"Only the dense histogram format can be loaded from a file, "
                                 f"got '{g.attrs['format']}' in '{group}'")

            axes = Axes.open(g['axes'])

            if g['contents'].shape != tuple(axes.nbins):
                raise ValueError(f"Histograms with overflow bins can't be loaded from a file ('{group}')")

            unit = u.Unit(g.attrs['unit']) if 'unit' in g.attrs else None

        ei_axis = axes['Ei']
        ei_bins = slice(0, ei_axis.nbins)

        if selector is not None:
            ranges = selector.energy_ranges_keV
            eps_min, eps_max = u.Quantity(axes['Epsilon'].edges[[0, -1]]).to_value('')
            ei_min = ranges[:, 0].min() / (1 + eps_max)
            ei_max = ranges[:, 1].max() / (1 + eps_min) if eps_min > -1 else np.inf

            ei_edges = ei_axis.edges.to_value(u.keV)
            needed = np.flatnonzero((ei_edges[1:] > ei_min) & (ei_edges[:-1] < ei_max))

            if len(needed) == 0:
                raise ValueError(f"No Ei bin of '{group}' can produce events in the selection")

            ei_bins = slice(max(needed[0] - 1, 0), min(needed[-1] + 2, ei_axis.nbins))

            axes = Axes([Axis(ei_axis.edges[ei_bins.start:ei_bins.stop + 1], label='Ei', scale=ei_axis.axis_scale)
                         if axis.label == 'Ei' else axis
                         for axis in axes])

        return axes, unit, ei_bins

    def _load_pixels(self, pixels):
        """
        Read the contents of the ``NuLambda`` ``pixels`` from the file, directly
        into ``_event_prob``, and normalize them. The file is opened for each call.
        """

        filename, group, ei_bins = self._source

        with h5.File(filename, 'r') as f:
            dataset = f[group]['contents']

            for pixel in pixels:
                contents = self._event_prob.contents[pixel]
                dataset.read_direct(contents, np.s_[pixel, ei_bins])
                contents *= self._area_unit_factor
                self._normalize_pixel(pixel, contents)

    def _ensure_loaded(self, photon_dir):
        """
        Load the ``NuLambda`` pixels that the interpolation at ``photon_dir`` uses.
        Every evaluation goes through here before reading ``_event_prob``,
        ``_epsilon_marginal`` or a ``_tot_aeff`` projected from the response.
        """

        if self._loaded.all():
            return

        pixels, weights = self._event_prob.axes['NuLambda'].interp_weights(photon_dir)
        pixels = np.unique(np.asarray(pixels)[np.asarray(weights) != 0])
        missing = pixels[~self._loaded[pixels]]

        if len(missing):
            self._load_pixels(missing)

    def _normalize_pixel(self, pixel: int, contents: np.ndarray):
        """
        Turn the contents of one ``NuLambda`` pixel (cm^2 per bin, axes ``Ei``,
        ``Epsilon``, ``Phi``, ``Theta``, ``Zeta``) into the normalized event
        probability, in place, and fill the per-pixel tables.

        The contents are divided by the phase space of each bin and normalized
        per ``Ei`` by the integral of the interpolant (see :meth:`_cds_marginal_weights`).
        With non-uniform bins or near the edges, that is not the sum of the contents.
        """

        if self._aeff_from_irf:
            self._tot_aeff.contents[pixel] = contents.sum(axis=(1, 2, 3, 4))

        contents /= self._phase_space_cds
        contents /= self._event_prob.axes['Epsilon'].widths[:, None, None, None]

        # Bins in the unphysical region of the CDS reparametrization (Phi +
        # Theta outside [0, pi]) have zero phase space, so the divisions
        # above produce 0/0 = NaN there if the contents are exactly zero,
        # or +-inf if they are not (e.g. due to smoothing or interpolation
        # when the histogram was built). Replace both with the physically
        # correct value of zero density, so these bins
        # don't poison interpolation for nearby physical events. Note that
        # the default posinf/neginf of nan_to_num would instead turn inf
        # into the largest finite float, which overflows to inf/NaN when
        # interpolating.
        contents[~np.isfinite(contents)] = 0

        cds_marginal = np.einsum('eijkl,jkl->ei', contents, self._cds_weights)
        integral = cds_marginal @ self._epsilon_hats.integrals()
        inv_integral = np.divide(1., integral, out=np.zeros_like(integral), where=integral > 0)
        contents *= inv_integral[:, None, None, None, None]

        # Integral over Phi, Theta and Zeta of the normalized density, per Epsilon center
        self._epsilon_marginal[pixel] = cds_marginal * inv_integral[:, None]
        self._loaded[pixel] = True

    @property
    def axes(self) -> Axes:
        """
        Axes of the differential response: ``NuLambda``, ``Ei`` (keV),
        ``Epsilon``, ``Phi``, ``Theta`` and ``Zeta`` (rad), without
        units. The total effective area might use a finer ``Ei`` grid.
        """
        return self._event_prob.axes

    @staticmethod
    def _cds_marginal_weights(axes: Axes) -> np.ndarray:
        """
        Weights to integrate the density over ``Phi``, ``Theta`` and ``Zeta``,
        as evaluated by ``interp()``, with the true phase space
        ``sin(phi + theta) dphi dtheta dzeta``. Contracting them with the
        density per (``Phi``, ``Theta``, ``Zeta``) center gives the integral per
        ``Epsilon`` center.

        The interpolant is a sum of products of 1D hat functions, so the
        integral is a contraction of the density with one weight per
        (``Phi``, ``Theta``, ``Zeta``) center. The phase space is uniform in
        ``Zeta``, which then uses the exact integral of each hat. ``Phi`` and
        ``Theta`` are split into ``_CDS_SUBDIVISIONS`` sub-bins per bin, where
        the hats are evaluated at the sub-bin centers and weighted by the
        exact phase space of the sub-bin. The error decreases as the square of
        the number of sub-bins, and it is ~1e-4 for the default.

        Parameters
        ----------
        axes : histpy.Axes
            Standardized axes ``NuLambda, Ei, Epsilon, Phi, Theta, Zeta``.

        Returns
        -------
        numpy.ndarray, shape (nPhi, nTheta, nZeta)
        """

        nsub = IRFRelativeHistUnpolarized._CDS_SUBDIVISIONS

        def sub_bins(axis):
            edges = np.asarray(axis.edges, dtype=float)
            sub_edges = np.concatenate([np.linspace(lo, hi, nsub + 1)[:-1]
                                        for lo, hi in zip(edges[:-1], edges[1:])] + [edges[-1:]])
            hats = _Hats(axis, 'clamp')(0.5 * (sub_edges[1:] + sub_edges[:-1]))
            return sub_edges, hats

        phi_edges, phi_hats = sub_bins(axes['Phi'])
        theta_edges, theta_hats = sub_bins(axes['Theta'])
        phi_mesh, theta_mesh = np.meshgrid(phi_edges, theta_edges, indexing='ij')

        # Phase space per unit Zeta of each (Phi, Theta) sub-bin
        phase_space = RelativeCDSCoordinates.get_relative_cds_phase_space(
            phi_mesh[:-1, :-1], phi_mesh[1:, :-1], theta_mesh[:-1, :-1], theta_mesh[:-1, 1:],
            np.zeros_like(phi_mesh[:-1, :-1]), np.ones_like(phi_mesh[:-1, :-1]))
        phase_space = np.nan_to_num(np.asarray(phase_space, dtype=float))

        zeta_integrals = _Hats(axes['Zeta'], 'periodic').integrals()

        return np.einsum('ab,aj,bk->jk', phase_space, phi_hats, theta_hats)[:, :, None] * zeta_integrals

    def _selected_fraction(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> np.ndarray:
        """
        Fraction of the interpolated event probability of each photon with
        ``Em`` inside the selections, using the same ``NuLambda`` and ``Ei``
        interpolation weights as the density and the exact integral of the
        ``Epsilon`` hats over the cut, so it is consistent with the density.
        Zero for photons outside the ``Ei`` axis.
        """

        photon_lon_rad, photon_lat_rad, energy_keV = self._photon_list_to_raw_values(photons)
        energy_keV = np.clip(energy_keV, 1e-10, None)
        photon_dir = UnitSphericalRepresentation(lon=Quantity(photon_lon_rad, 'rad', copy=False),
                                                 lat=Quantity(photon_lat_rad, 'rad', copy=False))

        axes = self._event_prob.axes
        self._ensure_loaded(photon_dir)
        pix, wpix = axes['NuLambda'].interp_weights(photon_dir)
        ei_bins, wei = axes['Ei'].interp_weights(energy_keV)

        # Integral of each Epsilon hat over the cut. Em = Ei (1 + Epsilon).
        hat_cut = np.zeros((len(energy_keV), axes['Epsilon'].nbins))
        for lo, hi in self._selection_ranges:
            hat_cut += (self._epsilon_hats.cumulative(hi / energy_keV - 1)
                        - self._epsilon_hats.cumulative(lo / energy_keV - 1))

        fraction = np.zeros(len(energy_keV))
        for p, w_p in zip(pix, wpix):
            for k, w_k in zip(ei_bins, wei):
                fraction += w_p * w_k * np.einsum('ni,ni->n', self._epsilon_marginal[p, k], hat_cut)

        return np.where(self._in_ei_range(self._event_prob, energy_keV), fraction, 0.0)

    @staticmethod
    def _combine_selections(selections) -> Union[EnergySelector, None]:
        """
        Union of the ``selections`` (a single ``EnergySelector`` or a
        tuple of them), or None if there are none.

        Raises
        ------
        TypeError
            If any selection is not an ``EnergySelector``.
        """

        if selections is None:
            return None

        if not isinstance(selections, tuple):
            selections = (selections,)

        for sel in selections:
            if not isinstance(sel, EnergySelector):
                raise TypeError(f"IRFRelativeHistUnpolarized only supports EnergySelector "
                                 f"selections currently, got {type(sel)}")

        return functools.reduce(EnergySelector.union, selections) if selections else None

    def _parallel_interp(self, hist, build_args, raw_arrays, **interp_kwargs):
        """
        Interpolate ``hist`` at the points described by ``raw_arrays``,
        splitting the work across :attr:`_nthreads` worker threads when
        there's enough of it to be worth it.

        ``histpy.Histogram.interp()``'s bottleneck (fancy-indexing the
        contents array, then a stack/reduction, done once per axis-bin
        combination -- see ``Histogram._interp_multilinear``) is
        vectorized numpy that releases the GIL for large inputs, so
        splitting the input points into chunks and interpolating each in
        its own thread gives real wall-clock speedup once there are
        enough points per thread. Below that, the fixed cost of fanning
        work out across threads and gathering results back dominates and
        makes it slower than calling ``interp()`` directly -- hence the
        ``_npoints_parallel_thresh``-based threshold below.

        Parameters
        ----------
        hist : histpy.Histogram
            The histogram to interpolate (``self._tot_aeff`` or
            ``self._event_prob``).
        build_args : callable
            Takes the (possibly chunked) raw arrays and returns the
            positional arguments ``hist.interp()`` expects. This is
            where e.g. a raw (lon, lat) pair gets wrapped into an
            ``astropy.coordinates.UnitSphericalRepresentation`` for the
            ``NuLambda`` axis -- cheaply, and once per chunk rather than
            once for the whole input.
        raw_arrays : tuple of numpy.ndarray
            The flat arrays to (maybe) split across threads. All must
            have the same length.
        **interp_kwargs
            Passed to ``hist.interp()``.

        Returns
        -------
        numpy.ndarray
            The concatenated interpolated values, in the original point
            order.
        """

        n = len(raw_arrays[0])

        if self._executor is None or n < self._nthreads * self._npoints_parallel_thresh:
            return hist.interp(*build_args(*raw_arrays), **interp_kwargs)

        nchunks = min(self._nthreads, max(1, n // self._npoints_parallel_thresh))
        chunks = zip(*(np.array_split(a, nchunks) for a in raw_arrays))
        futures = [self._executor.submit(lambda c=c: hist.interp(*build_args(*c), **interp_kwargs))
                   for c in chunks]

        return np.concatenate([f.result() for f in futures])

    @staticmethod
    def _standardize_aeff(aeff: Histogram, copy: bool) -> Histogram:
        """
        Validate a standalone total-effective-area histogram (the
        ``aeff`` constructor argument) and standardize its axis units,
        the same way ``irf``'s ``NuLambda``/``Ei`` axes and contents
        are standardized in :meth:`__init__`.

        Parameters
        ----------
        aeff : histpy.Histogram
            2D histogram with axes ``['NuLambda', 'Ei']`` and contents
            in units equivalent to area.
        copy : bool
            Whether to copy ``aeff`` before modifying it.

        Returns
        -------
        histpy.Histogram
            ``aeff`` with its ``Ei`` axis in keV and its contents in
            cm^2, both unitless (implicit units), ready to be
            interpolated on directly.

        Raises
        ------
        ValueError
            If the histogram contents are not area-equivalent, the
            axis labels are not ``['NuLambda', 'Ei']``, or an axis has
            an unexpected type or units.
        """

        if copy:
            aeff = aeff.copy()

        if not aeff.unit.is_equivalent('cm^2'):
            raise ValueError("aeff contents are expected to have units of area.")

        axes = aeff.axes

        if not np.array_equal(axes.labels, ['NuLambda', 'Ei']):
            raise ValueError("aeff axes label must be ['NuLambda', 'Ei']")

        if not isinstance(axes['NuLambda'], HealpixAxis):
            raise ValueError("aeff NuLambda axis is expected to be of HealpixAxis type")

        if axes['Ei'].unit is None or not axes['Ei'].unit.is_equivalent('keV'):
            raise ValueError("aeff Ei axis is expected to have units of energy.")

        axes['Ei'] = axes['Ei'].to(u.keV, copy = False).to(None, update = False, copy = False)

        return aeff.to(u.cm * u.cm, copy=False).to(None, copy=False, update=False)

    @classmethod
    def from_h5(cls, filename, *args, **kwargs):
        """
        Construct an :class:`IRFRelativeHistUnpolarized` from an HDF5
        file that stores the response histogram under the group
        ``"IRF"``. The response is read on demand, one ``NuLambda`` pixel at
        a time, unless ``preload=True`` is passed (see the class docstring).

        If the file also has a group named ``"AEFF"``, it is read as
        well and passed to :meth:`__init__` as the ``aeff`` argument
        (see the class docstring), unless ``aeff`` was already provided
        via ``*args``/``**kwargs``, in which case that value takes
        precedence and the file is not checked for an ``"AEFF"``
        group.

        Parameters
        ----------
        filename : str or path-like
            Path to the HDF5 file containing the response histogram
            (and, optionally, the separate total effective area
            histogram).
        *args, **kwargs
            Extra arguments forwarded verbatim to
            :meth:`__init__` (e.g. ``aeff``, ``copy``, ``nthreads``,
            ``npoints_parallel_thresh``, ``selections`` or ``preload``).
            ``copy`` defaults to False here, since the histograms are read
            from disk and nobody else holds them.

        Returns
        -------
        IRFRelativeHistUnpolarized
            Initialized instance with the histogram(s) loaded from
            disk.
        """

        if 'aeff' not in kwargs and len(args) == 0:
            with h5.File(filename, 'r') as f:
                if 'AEFF' in f:
                    kwargs['aeff'] = Histogram.open(filename, 'AEFF')

        if len(args) < 2:
            kwargs.setdefault('copy', False)

        return cls(filename, *args, **kwargs)

    def _effective_area_cm2(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> Iterable[float]:
        """
        Total effective area, in cm^2, for each incident photon, after
        any energy ``selections``.

        Interpolates the total effective area (``_tot_aeff``) at the
        direction and energy of each photon in the list, times the
        selected fraction if there are selections. It is zero for
        photons outside the range of the ``Ei`` axis. See
        :meth:`_parallel_interp` for how this is (optionally)
        parallelized over threads.

        Parameters
        ----------
        photons : PhotonListWithDirectionAndEnergyInSCFrameInterface
            Photons to evaluate the effective area on.

        Returns
        -------
        Iterable[float]
            One effective-area value per photon, in cm^2.
        """

        unselected = self._effective_area_cm2_unselected(photons)

        if self._selection_ranges is None:
            return unselected

        return unselected * self._selected_fraction(photons)

    def _effective_area_cm2_unselected(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> np.ndarray:
        """
        Same as :meth:`_effective_area_cm2`, but before any energy
        ``selections``.
        """

        photon_lon_rad, photon_lat_rad, photon_energy_keV = self._photon_list_to_raw_values(photons)

        if self._aeff_from_irf:
            self._ensure_loaded(UnitSphericalRepresentation(lon=Quantity(photon_lon_rad, 'rad', copy=False),
                                                            lat=Quantity(photon_lat_rad, 'rad', copy=False)))

        def build_args(lon_rad, lat_rad, energy_keV):
            photon_dir = UnitSphericalRepresentation(lon=Quantity(lon_rad, 'rad', copy=False),
                                                     lat=Quantity(lat_rad, 'rad', copy=False))
            return photon_dir, energy_keV

        values = np.asarray(self._parallel_interp(self._tot_aeff, build_args,
                                                  (photon_lon_rad, photon_lat_rad, photon_energy_keV)))

        return np.where(self._in_ei_range(self._tot_aeff, photon_energy_keV), values, 0.0)

    @staticmethod
    def _in_ei_range(hist: Histogram, energy_keV: np.ndarray) -> np.ndarray:
        """
        Whether ``energy_keV`` is within the range of ``hist``'s ``Ei``
        axis. ``interp()`` clamps to the edge bins instead.
        """

        ei_edges = hist.axes['Ei'].edges

        return (energy_keV >= ei_edges[0]) & (energy_keV <= ei_edges[-1])

    @staticmethod
    def _photon_list_to_raw_values(photons:PhotonListWithDirectionAndEnergyInSCFrameInterface):
        """
        Extract the raw arrays required to evaluate the response from
        a photon list.

        Returned as plain arrays, rather than a prebuilt
        ``UnitSphericalRepresentation``, so callers can split them into
        chunks (e.g. for :meth:`_parallel_interp`) with ``np.array_split``
        before building coordinate objects per chunk.

        Parameters
        ----------
        photons : PhotonListWithDirectionAndEnergyInSCFrameInterface
            Photon list providing spacecraft-frame directions and
            energies.

        Returns
        -------
        photon_lon_rad : numpy.ndarray
            Photon longitudes in the spacecraft frame, in radians.
        photon_lat_rad : numpy.ndarray
            Photon latitudes in the spacecraft frame, in radians.
        photon_energy_keV : numpy.ndarray
            Photon energies in keV, as a float array.
        """

        photon_lon_rad = asarray(photons.direction_lon_rad_sc, float)
        # Clip away floating-point overshoot past the poles (e.g. from
        # float32 downcasting upstream), which Latitude validates strictly
        # against.
        photon_lat_rad = np.clip(asarray(photons.direction_lat_rad_sc, float), -np.pi / 2, np.pi / 2)

        photon_energy_keV = asarray(photons.energy_keV, float)

        return photon_lon_rad, photon_lat_rad, photon_energy_keV

    def _interp_event_prob(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface, events: EmCDSEventDataInSCFrameInterface) -> np.ndarray:
        """
        ``_event_prob`` interpolated at each (photon, event) pair, in
        ``1 / sr / rad / keV``, without any selection fraction applied.
        The stored table is per unit ``Epsilon``, so the interpolated value
        is divided by the photon energy.

        For each pair the incident photon direction and energy are
        combined with the event's scattered direction, kinematic
        scattering angle and measured energy to build the six
        relative-coordinate arguments consumed by the histogram:

        - ``NuLambda`` : photon direction in the spacecraft frame,
        - ``Ei`` : true photon energy in keV,
        - ``Epsilon`` : fractional energy deviation
          ``(Em - Ei) / Ei``,
        - ``Phi`` : kinematics-derived scattering angle in radians,
        - ``Theta`` : difference between the geometric and kinematic
          scattering angles in radians,
        - ``Zeta`` : azimuthal scattered-photon angle around the
          source direction, in radians, expressed in the stored
          polarization convention.

        It is zero where ``Ei`` is outside the range of its axis.
        """

        photon_lon_rad, photon_lat_rad, photon_energy_keV = self._photon_list_to_raw_values(photons)

        photon_dir = UnitSphericalRepresentation(lon=Quantity(photon_lon_rad, 'rad', copy=False),
                                                 lat=Quantity(photon_lat_rad, 'rad', copy=False))

        psichi_lon_rad = asarray(events.scattered_lon_rad_sc, float)

        # Clip away values outside the range due to floating-point errors since
        # UnitSphericalRepresentation validates strictly
        psichi_lat_rad = np.clip(asarray(events.scattered_lat_rad_sc, float), -np.pi / 2, np.pi / 2)

        psichi_dir = UnitSphericalRepresentation(lon = Quantity(psichi_lon_rad, 'rad', copy = False),
                                                 lat = Quantity(psichi_lat_rad, 'rad', copy = False))

        phi_kin_rad = asarray(events.scattering_angle_rad, float)
        measured_energy_keV = asarray(events.energy_keV, float)

        # Convert to relative coordinates
        epsilon = (measured_energy_keV - photon_energy_keV)/photon_energy_keV

        self._ensure_loaded(photon_dir)

        relcoords = RelativeCDSCoordinates(photon_dir.to_cartesian().xyz, pol_convention=self._pol_convention)
        phi_geo, zeta = relcoords.to_relative(psichi_dir.to_cartesian().xyz)

        phi_geo_rad = phi_geo.to_value(u.rad)
        zeta_rad = zeta.to_value(u.rad)

        theta_rad = phi_geo_rad - phi_kin_rad

        # The relative-coordinate math above is already fully vectorized
        # over the whole input and cheap relative to the interpolation
        # below, so only the interp() call itself is (optionally)
        # parallelized -- see _parallel_interp.
        def build_args(lon_rad, lat_rad, energy_keV, eps, phi, theta, zeta_r):
            photon_dir_chunk = UnitSphericalRepresentation(lon=Quantity(lon_rad, 'rad', copy=False),
                                                            lat=Quantity(lat_rad, 'rad', copy=False))
            return photon_dir_chunk, energy_keV, eps, phi, theta, zeta_r

        # The table is per unit Epsilon, and dEm = Ei * dEpsilon
        prob = np.asarray(self._parallel_interp(self._event_prob, build_args,
                                                (photon_lon_rad, photon_lat_rad, photon_energy_keV,
                                                 epsilon, phi_kin_rad, theta_rad, zeta_rad),
                                                fill_value=self._FILL_VALUE, extrapolate=True)) / photon_energy_keV

        return np.where(self._in_ei_range(self._event_prob, photon_energy_keV), prob, 0.0)

    def _differential_effective_area_cm2(self, photons:PhotonListWithDirectionAndEnergyInSCFrameInterface, events: EmCDSEventDataInSCFrameInterface) -> Iterable[float]:
        """
        Differential effective area, in cm^2 per unit phase-space
        for each (photon, event) pair.

        This is the total effective area of the photon before any
        energy ``selections`` (:meth:`_effective_area_cm2_unselected`)
        times the interpolated event probability density
        (``_event_prob``), so it integrates over the data space
        (``Em``, ``Phi``, ``PsiChi``) to that effective area. The
        selections scale the area but not the density, i.e. the result
        is not conditional on passing them. See
        :meth:`_interp_event_prob` for the relative coordinates used.

        Parameters
        ----------
        photons : PhotonListWithDirectionAndEnergyInSCFrameInterface
            True photon directions and energies, one per event.
        events : EmCDSEventDataInSCFrameInterface
            Reconstructed events providing the scattered direction,
            kinematics-derived scattering angle, and measured energy.

        Returns
        -------
        Iterable[float]
            Differential effective area at each (photon, event) pair,
            in ``cm^2 / sr / rad / keV``. It is zero where ``Epsilon``
            or ``Ei`` are outside the range of their axes.
        """

        return self._effective_area_cm2_unselected(photons) * self._interp_event_prob(photons, events)

    def _event_probability(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface, events: EmCDSEventDataInSCFrameInterface) -> Iterable[float]:
        """
        Probability density, in ``1 / sr / rad / keV``, of each event
        given its photon, normalized over the data space (``Em``,
        ``Phi``, ``PsiChi``).

        If energy ``selections`` were given, the probability is
        conditional on the event passing them: the interpolated
        ``_event_prob`` is divided by the selected fraction. Where that
        fraction is zero (e.g. photon energy outside the range of the
        histogram) the event is impossible, so the probability is zero
        instead of 0/0 = NaN or x/0 = inf.
        """

        prob = self._interp_event_prob(photons, events)

        if self._selection_ranges is None:
            return prob

        fraction = self._selected_fraction(photons)

        return np.divide(prob, fraction, out=np.zeros_like(prob), where=fraction > 0)

    def _random_events(self, photons: PhotonListWithDirectionInSCFrameInterface) -> EventDataInterface:
        """
        Not implemented yet; provided to satisfy the
        :class:`FarFieldSpectralInstrumentResponseFunctionInterface`
        contract.
        """
        raise NotImplementedError("random_events not implemented yet.")
