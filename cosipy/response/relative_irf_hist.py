import functools
from concurrent.futures import ThreadPoolExecutor
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

    On construction the input histogram is validated, its axis units
    are standardized (energies to keV, angles to radians, contents to
    cm^2), and it is divided by the phase-space volume of each bin and
    normalized, per ``(NuLambda, Ei)``, so that it
    integrates to one over ``Epsilon``, ``Phi``, ``Theta`` and
    ``Zeta``. The result is stored as ``_event_prob``, the probability
    density of an event (``Em``, ``Phi``, ``PsiChi``) given the photon
    (``NuLambda``, ``Ei``), with implicit units of
    ``1 / sr / rad / keV``. The total effective area of the photon
    (``aeff``, or the ``NuLambda``/``Ei`` projection of ``irf``) is
    stored separately, so that the differential effective area is that
    area times the event probability. Photons outside the range of the
    ``Ei`` axis get zero area and probability. The polarization
    convention carried by
    the ``Zeta`` axis is preserved as :attr:`_pol_convention` so it
    can be re-applied when translating event directions into the
    relative coordinate system.

    Two evaluation modes are available (``interpolation``). In the
    ``'linear'`` mode the density is multilinearly interpolated between
    bin centers. Its normalization is exact only at ``Ei`` bin centers
    (and for photons at ``NuLambda`` pixel centers), and assumes the phase
    space is uniform within each bin. In the ``'nearest'`` mode the density
    is piecewise constant, the event probability is exactly normalized
    for any photon, and the selected effective area is exactly consistent
    with it.

    This class is labelled "Unpolarized" because it does not model a
    dependence of the response on the incident photon polarization
    angle; the ``Zeta`` axis is used purely as the scattered-photon
    azimuth, not as a polarization-input degree of freedom.

    Parameters
    ----------
    irf : histpy.Histogram
        A 6D histogram with the axes described above and contents in
        units equivalent to area (``cm^2``).
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
        Measured-energy cut(s) to bake into the total effective area
        (:attr:`_tot_aeff`), so it matches an equivalent cut already
        applied to the event data being fit (e.g. via the same
        ``EnergySelector``). The total effective area before the cut is
        kept in ``_aeff_unselected``, and the event probability is divided by the
        selected fraction (``_tot_aeff / _aeff_unselected``) to make it conditional on passing the cut.
        The differential effective area is not scaled by it. Only ``EnergySelector`` is currently
        supported; anything else raises ``TypeError``. If a tuple is
        given, the cuts combine with OR semantics (unioned via
        ``EnergySelector.union``) -- i.e. each entry is treated as an
        acceptable energy window, and an event/bin passes if it falls
        in any of them. For AND semantics, pass a single
        pre-intersected ``EnergySelector`` (e.g.
        ``sel1.intersect(sel2)``) instead of a tuple.

        For each ``(NuLambda, Ei)`` bin, ``_tot_aeff`` is scaled by the
        fraction of that bin's effective area whose measured energy
        ``Em = Ei*(1 + Epsilon)`` falls inside the selection. If
        ``tot_aeff`` is on a different (typically finer) ``NuLambda``/
        ``Ei`` grid than ``irf`` (the ``aeff`` constructor parameter),
        ``irf``'s own ``(NuLambda, Ei, Epsilon)`` response is first
        interpolated (in ``NuLambda``/``Ei`` only, one ``Epsilon`` bin
        at a time) onto ``tot_aeff``'s grid, and the selection fraction
        is then evaluated at each of ``tot_aeff``'s own exact ``Ei``
        values -- rather than evaluating the fraction on ``irf``'s own
        (coarser) ``Ei`` grid and interpolating *that* onto ``tot_aeff``.

        To account for a narrow energy cut, the ``_tot_aeff``'s ``Ei`` grid is
        also refined automatically near the cut: the fraction varies
        with ``Ei`` on a scale of ~``Ei * dEpsilon``, often much finer
        than ``aeff``'s own ``Ei`` bins, and ``_tot_aeff`` is linearly
        interpolated in ``Ei`` at evaluation time. Extra ``Ei`` edges are
        added wherever a cut boundary ``E_cut`` maps onto an ``Epsilon``
        edge or center (``Ei = E_cut / (1 + Epsilon)``); every original
        ``aeff`` edge is kept. Without a separate ``aeff``, ``_tot_aeff``
        stays on ``irf``'s own ``Ei`` grid.

        Within a single ``Ei`` value, the fraction is computed by
        linearly interpolating between ``Epsilon`` bin centers.
        Defaults to ``None`` (no cut).
    interpolation : {'linear', 'nearest'}, optional
        How the event probability is evaluated between bin centers.
        ``'linear'`` (default) is described above. ``'nearest'`` uses the
        value of the bin each (photon, event) pair lands in, on all six
        axes, and zero outside the axes' ranges. In this mode the event
        probability integrates to one over the data space for any photon
        (at any ``Ei`` and direction, not only at bin centers), and the
        selected effective area is exactly the integral of the
        differential effective area over the selected ``Em`` range: it is
        the unselected area times the fraction of the photon's own
        ``(NuLambda, Ei)`` bin content that falls in the selection
        (``Epsilon`` bins partially inside a range count by their
        overlap), instead of the interpolated ``_tot_aeff``. Because the
        response is a step function of ``Ei``, an integrator over ``Ei``
        must resolve the step at every ``Ei`` bin edge (e.g. trapezoidal
        folding with nodes at the ``Ei`` and ``Epsilon`` bin edges).
    """

    event_data_type = EmCDSEventDataInSCFrameInterface
    photon_list_type = PhotonListWithDirectionAndEnergyInSCFrameInterface

    def __init__(self,
                 irf: Histogram,
                 aeff: Histogram = None,
                 copy = True,
                 nthreads = 1,
                 npoints_parallel_thresh = 1000,
                 selections: Union[EventSelectorInterface, Tuple[EventSelectorInterface, ...]] = None,
                 interpolation: str = 'linear'):
        """
        Validate the input histogram(s), standardize their axis units,
        and pre-compute the total and differential effective area used
        at evaluation time.

        See the class docstring for a description of the expected axes
        and units.

        ``irf`` is normalized in place (it is the ``_event_prob`` attribute
        afterwards), without full-size temporary arrays. Anything that
        needs its raw contents (the total effective area and the energy
        selection) is computed before.

        Parameters
        ----------
        irf : histpy.Histogram
            Input response histogram.
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
            Measured-energy cut(s) applied to ``_tot_aeff``. See the
            class docstring.
        interpolation : {'linear', 'nearest'}, optional
            Evaluation mode. See the class docstring.

        Raises
        ------
        ValueError
            If the histogram contents are not area-equivalent, the
            axis labels do not match the expected sequence, or an
            axis has an unexpected type or units, or ``interpolation``
            is not ``'linear'`` or ``'nearest'``.
        """

        if interpolation not in ('linear', 'nearest'):
            raise ValueError(f"interpolation must be 'linear' or 'nearest', got {interpolation!r}")

        self._interpolation = interpolation

        if copy:
            irf = irf.copy()

        # Checks
        if not irf.unit.is_equivalent('cm^2'):
            raise ValueError("IRF contents are expected to have units of area.")

        axes = irf.axes

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

        # Events are evaluated with Zeta wrapped into [0, 360) deg (see
        # _differential_effective_area_cm2), so the axis must span exactly
        # that range for the wrapped values to land on a valid bin.
        zeta_edges_deg = axes['Zeta'].edges.angle.to_value(u.deg)
        if not (np.isclose(zeta_edges_deg[0], 0) and np.isclose(zeta_edges_deg[-1], 360)):
            raise ValueError("IRF Zeta axis is expected to span the full [0, 360) deg range, got "
                              f"[{zeta_edges_deg[0]}, {zeta_edges_deg[-1]}] deg")

        # Standardize units
        axes['Ei'] = axes['Ei'].to(u.keV, copy = False).to(None, update = False, copy = False)
        axes['Epsilon'] = axes['Epsilon'].to(None, update = False, copy = False)
        axes['Phi'] = axes['Phi'].to(u.rad, copy = False).to(None, update = False, copy = False)
        axes['Theta'] = axes['Theta'].to(u.rad, copy = False).to(None, update = False, copy = False)
        self._pol_convention = axes['Zeta'].convention
        axes['Zeta'] = Axis(axes['Zeta'].edges.angle.to(u.rad).value, label = 'Zeta')

        irf = irf.to(u.cm * u.cm, copy=False).to(None, copy=False, update=False) # To cm2 and remove units

        # Get the total effective area
        if aeff is not None:
            self._tot_aeff = self._standardize_aeff(aeff, copy) # cm^2
        else:
            self._tot_aeff = irf.project('NuLambda','Ei') # cm^2

        # Total effective area before any energy selection
        self._aeff_unselected = self._tot_aeff

        combined_selector = self._combine_selections(selections)

        if combined_selector is not None:
            self._tot_aeff = self._apply_energy_selection(irf, self._tot_aeff, combined_selector,
                                                            same_grid = aeff is None)

        # The divisions below are in place, since the contents are large

        # Phase space
        # Final content units will be 1/sr/rad/keV
        phi_edges_mesh, arm_edges_mesh, az_edges_mesh = np.meshgrid(axes['Phi'].edges,
                                                                    axes['Theta'].edges,
                                                                    axes['Zeta'].edges, indexing='ij')

        phase_space_cds = RelativeCDSCoordinates.get_relative_cds_phase_space(phi_edges_mesh[:-1, :-1, :-1],
                                                                              phi_edges_mesh[1:, :-1, :-1],
                                                                              arm_edges_mesh[:-1, :-1, :-1],
                                                                              arm_edges_mesh[:-1, 1:, :-1],
                                                                              az_edges_mesh[:-1, :-1, :-1],
                                                                              az_edges_mesh[:-1, :-1, 1:])

        ei_centers_mesh, epsilon_widths_mesh = np.meshgrid(axes['Ei'].centers,
                                                      axes['Epsilon'].widths,
                                                      indexing='ij')

        phase_space_em = ei_centers_mesh * epsilon_widths_mesh

        irf /= axes.expand_dims(phase_space_cds, axes.label_to_index(['Phi', 'Theta', 'Zeta']))
        irf /= axes.expand_dims(phase_space_em, axes.label_to_index(['Ei', 'Epsilon']))

        # Bins in the unphysical region of the CDS reparametrization (Phi +
        # Theta outside [0, pi]) have zero phase space, so the divisions
        # above produce 0/0 = NaN there if the contents are exactly zero,
        # or +-inf if they are not (e.g. due to smoothing or interpolation
        # when the histogram was built). Replace both with the physically
        # correct value of zero density, so these bins
        # don't poison interpolation for nearby physical events. Note that
        # the default posinf/neginf of nan_to_num would instead turn inf
        # into the largest finite float, which overflows to inf/NaN when
        # interpolating. This is done in place with a single mask, since
        # nan_to_num would also allocate copies of the (large) contents.
        irf.contents[~np.isfinite(irf.contents)] = 0

        # The density is linearly interpolated between bin centers when evaluated. With
        # non-uniform bins that interpolant doesn't integrate to the bin contents (e.g. it
        # over-integrates the tails of a peaked Theta distribution), so normalize by its
        # actual integral instead.
        if interpolation == 'nearest':
            # Content per (NuLambda, Ei, Epsilon) bin, excluding the bins zeroed above. The
            # density is piecewise constant, so this is its exact integral.
            self._nearest_proj = np.array([np.einsum('eabcd,bcd->ea', d, phase_space_cds)
                                           for d in irf.contents]) * phase_space_em[None]
            integral = self._nearest_proj.sum(axis=2)
        else:
            integral = self._interpolant_integral(irf, phase_space_em, phase_space_cds)
        inv_integral = np.divide(1., integral, out=np.zeros_like(integral), where=integral > 0)
        np.multiply(irf.contents, inv_integral[:, :, None, None, None, None], out=irf.contents)

        self._event_prob = irf
        self._has_selection = combined_selector is not None
        self._selection_ranges = (None if combined_selector is None
                                  else np.asarray(combined_selector.energy_ranges_keV, dtype=float))

        # Extra params
        self._nthreads = nthreads
        self._npoints_parallel_thresh = npoints_parallel_thresh
        self._executor = ThreadPoolExecutor(max_workers=nthreads) if nthreads > 1 else None

    @property
    def axes(self) -> Axes:
        """
        Axes of the differential response: ``NuLambda``, ``Ei`` (keV),
        ``Epsilon``, ``Phi``, ``Theta`` and ``Zeta`` (rad), without
        units. The total effective area might use a finer ``Ei`` grid.
        """
        return self._event_prob.axes

    @staticmethod
    def _interpolant_integral(density: Histogram, phase_space_em: np.ndarray,
                              phase_space_cds: np.ndarray) -> np.ndarray:
        """
        Integral, over the data space within the axes' ranges, of the
        ``density`` as evaluated by ``interp()`` (multilinear between bin
        centers, flat beyond the first/last center), per (``NuLambda``,
        ``Ei``).

        The interpolant is a sum of products of 1D "hat" functions, so its
        integral over a bin factorizes: along each axis, ``M[b, i]`` is the
        integral over bin ``b`` of the hat function of center ``i``. The
        phase-space measure is taken as uniform within each bin.

        Parameters
        ----------
        density : histpy.Histogram
            Density with axes ``NuLambda, Ei, Epsilon, Phi, Theta, Zeta``.
        phase_space_em : numpy.ndarray, shape (nEi, nEpsilon)
            Phase space of each (``Ei``, ``Epsilon``) bin (``Ei_center * dEps``).
        phase_space_cds : numpy.ndarray, shape (nPhi, nTheta, nZeta)
            Phase space of each CDS bin.

        Returns
        -------
        numpy.ndarray, shape (nNuLambda, nEi)
        """

        axes = density.axes
        cds_labels = ['Epsilon', 'Phi', 'Theta', 'Zeta']

        def hat_integrals(axis):
            centers, edges = np.asarray(axis.centers), np.asarray(axis.edges)
            unit = np.eye(len(centers))
            return np.array([[IRFRelativeHistUnpolarized._integrate_piecewise_linear(unit[i], centers, lo, hi)
                              for i in range(len(centers))]
                             for lo, hi in zip(edges[:-1], edges[1:])])

        m_eps, m_phi, m_theta, m_zeta = (hat_integrals(axes[label]) for label in cds_labels)
        widths = [np.asarray(axes[label].widths) for label in cds_labels]

        # Phase space per unit coordinate volume of each bin
        measure_em = phase_space_em / widths[0][None, :]
        measure_cds = phase_space_cds / (widths[1][:, None, None] * widths[2][None, :, None] * widths[3][None, None, :])

        integral = np.empty(density.contents.shape[:2])
        for pix, d in enumerate(density.contents):
            d = np.einsum('ai,eijkl->eajkl', m_eps, d)
            d = np.einsum('bj,eajkl->eabkl', m_phi, d)
            d = np.einsum('ck,eabkl->eabcl', m_theta, d)
            d = np.einsum('dl,eabcl->eabcd', m_zeta, d)
            integral[pix] = np.einsum('eabcd,ea,bcd->e', d, measure_em, measure_cds)

        return integral

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

    def _parallel_interp(self, hist, build_args, raw_arrays):
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
            The histogram to interpolate (e.g. ``self._tot_aeff`` or
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

        Returns
        -------
        numpy.ndarray
            The concatenated interpolated values, in the original point
            order.
        """

        n = len(raw_arrays[0])

        if self._executor is None or n < self._nthreads * self._npoints_parallel_thresh:
            return hist.interp(*build_args(*raw_arrays))

        nchunks = min(self._nthreads, max(1, n // self._npoints_parallel_thresh))
        chunks = zip(*(np.array_split(a, nchunks) for a in raw_arrays))
        futures = [self._executor.submit(lambda c=c: hist.interp(*build_args(*c)))
                   for c in chunks]

        return np.concatenate([f.result() for f in futures])

    @staticmethod
    def _selection_fraction(content_vs_epsilon: np.ndarray, target_ei_keV: np.ndarray,
                            epsilon_centers: np.ndarray, epsilon_widths: np.ndarray,
                            epsilon_edges: np.ndarray, selector: EnergySelector) -> np.ndarray:
        """
        Fraction of ``content_vs_epsilon``'s total (summed over
        ``Epsilon``, per ``(NuLambda, Ei)``) whose measured energy
        ``Em = Ei*(1 + Epsilon)`` falls within ``selector``'s ranges,
        evaluated at each of ``target_ei_keV``'s *exact* values.

        Within a single ``Ei`` value, both the selected part and the
        total are integrals over ``Epsilon`` of the density linearly
        interpolated between bin centers (and flat up to the axis
        edges) -- see :meth:`_integrate_piecewise_linear` -- the same
        model used to evaluate the differential effective area, so the
        fraction is never above 1. With non-uniform ``Epsilon`` bins
        this total differs slightly from the plain sum of the contents.

        Parameters
        ----------
        content_vs_epsilon : numpy.ndarray, shape (npix, nEi, nEpsilon)
            Per-``Epsilon``-bin effective area (cm^2), already evaluated
            at each of ``target_ei_keV`` -- i.e. not itself interpolated
            in ``Ei`` here, so the ``Em -> Epsilon`` conversion below
            uses each target ``Ei`` exactly. See
            :meth:`_apply_energy_selection` for how this is built.
        target_ei_keV : numpy.ndarray, shape (nEi,)
            True energy of each ``Ei`` slice of ``content_vs_epsilon``.
        epsilon_centers, epsilon_widths, epsilon_edges : numpy.ndarray
            ``irf``'s own ``Epsilon`` axis centers/widths/edges.
        selector : EnergySelector
            The (possibly multi-range) combined energy selection.

        Returns
        -------
        numpy.ndarray, shape (npix, nEi)
            Selected fraction of the total, per ``(NuLambda, Ei)``.
        """

        npix, nEi, _ = content_vs_epsilon.shape
        fraction = np.ones((npix, nEi))

        for i, ei in enumerate(target_ei_keV):
            content_i = content_vs_epsilon[:, i, :]
            density_i = content_i / epsilon_widths[None, :]
            total_i = IRFRelativeHistUnpolarized._integrate_piecewise_linear(
                density_i, epsilon_centers, epsilon_edges[0], epsilon_edges[-1])

            selected_total = np.zeros(npix)

            for lo_keV, hi_keV in selector.energy_ranges_keV:
                eps_lo = np.clip(lo_keV / ei - 1, epsilon_edges[0], epsilon_edges[-1])
                eps_hi = np.clip(hi_keV / ei - 1, epsilon_edges[0], epsilon_edges[-1])

                if eps_lo <= epsilon_edges[0] and eps_hi >= epsilon_edges[-1]:
                    # This range alone covers the whole bin -- exact,
                    # and no other range can add anything more.
                    selected_total = total_i
                    break

                selected_total = selected_total + IRFRelativeHistUnpolarized._integrate_piecewise_linear(
                    density_i, epsilon_centers, eps_lo, eps_hi)

            with np.errstate(invalid='ignore', divide='ignore'):
                fraction[:, i] = np.nan_to_num(selected_total / total_i)

        return fraction

    @staticmethod
    def _apply_energy_selection(irf: Histogram, tot_aeff: Histogram,
                                 selector: EnergySelector, same_grid: bool) -> Histogram:
        """
        Scale ``tot_aeff`` down, per ``(NuLambda, Ei)``, by the fraction
        of ``irf``'s effective area whose measured energy
        ``Em = Ei*(1 + Epsilon)`` falls within ``selector``'s ranges.

        If ``tot_aeff`` shares ``irf``'s own ``(NuLambda, Ei)`` grid
        (``same_grid``), the fraction is evaluated directly there --
        that's the only place the measured-energy distribution is
        known, via :meth:`_selection_fraction`. Otherwise (the ``aeff``
        constructor parameter gave a separate, typically finer, grid),
        ``irf``'s own ``(NuLambda, Ei, Epsilon)`` response is first
        interpolated -- in ``NuLambda``/``Ei`` only, one ``Epsilon`` bin
        at a time, so ``Epsilon`` itself is not reinterpolated -- onto
        ``tot_aeff``'s grid, and the selection fraction is then
        evaluated at each of ``tot_aeff``'s own *exact* ``Ei`` values.

        This matters for a narrow energy cut: ``Em -> Epsilon``
        (``eps = Em/Ei - 1``) depends on ``Ei``, so the corresponding
        ``Epsilon`` window shifts and rescales with it, and the
        resulting *fraction* can vary sharply with ``Ei`` even where
        the underlying response varies smoothly.

        In that (separate ``aeff``) case, ``tot_aeff``'s ``Ei`` grid is
        first refined near the cut boundaries -- see
        :meth:`_refine_ei_edges` -- so the returned histogram can have
        more ``Ei`` bins than ``tot_aeff``.

        Parameters
        ----------
        irf : histpy.Histogram
            The 6D response histogram, still in area (cm^2) units --
            i.e. before the phase-space division that produces
            ``_diff_aeff``.
        tot_aeff : histpy.Histogram
            The (as yet unmodified) total effective area to scale.
        selector : EnergySelector
            The (possibly multi-range) combined energy selection.
        same_grid : bool
            True if ``tot_aeff`` shares ``irf``'s own ``NuLambda``/``Ei``
            grid (i.e. no separate ``aeff`` histogram was given), so the
            fraction can be evaluated directly without interpolation.

        Returns
        -------
        histpy.Histogram
            ``tot_aeff``, scaled by the selection fraction.
        """

        epsilon_axis = irf.axes['Epsilon']
        epsilon_centers = epsilon_axis.centers
        epsilon_widths = epsilon_axis.widths
        epsilon_edges = epsilon_axis.edges

        irf_vs_epsilon = irf.project('NuLambda', 'Ei', 'Epsilon')  # cm^2, irf's own grid

        if same_grid:
            target_ei_keV = irf.axes['Ei'].centers
            content_vs_epsilon = irf_vs_epsilon.contents

            fraction = IRFRelativeHistUnpolarized._selection_fraction(
                content_vs_epsilon, target_ei_keV, epsilon_centers, epsilon_widths, epsilon_edges, selector)

            fraction_hist = Histogram(tot_aeff.axes, contents=fraction)

            return tot_aeff * fraction_hist

        # A narrow cut makes the fraction vary with Ei on a scale set by
        # irf's Epsilon binning (~Ei * dEpsilon), typically much finer
        # than tot_aeff's own Ei grid. Refine that grid where needed, so
        # _tot_aeff (linearly interpolated in Ei at evaluation time)
        # resolves the cut.
        aeff_ei_axis = tot_aeff.axes['Ei']
        refined_ei_edges = IRFRelativeHistUnpolarized._refine_ei_edges(
            aeff_ei_axis.edges, epsilon_edges, epsilon_centers, selector)

        if len(refined_ei_edges) > len(aeff_ei_axis.edges):
            refined_ei_axis = Axis(refined_ei_edges, label='Ei', scale=aeff_ei_axis.axis_scale)
            tot_aeff = IRFRelativeHistUnpolarized._regrid_ei(tot_aeff, refined_ei_axis)

        nulambda_dir = tot_aeff.axes['NuLambda'].pix2skycoord(np.arange(tot_aeff.axes['NuLambda'].nbins))
        target_ei_keV = tot_aeff.axes['Ei'].centers

        lon_mesh, ei_mesh = np.meshgrid(nulambda_dir.lon.rad, target_ei_keV, indexing='ij')
        lat_mesh, _ = np.meshgrid(nulambda_dir.lat.rad, target_ei_keV, indexing='ij')

        photon_dir = UnitSphericalRepresentation(lon=Quantity(lon_mesh, 'rad', copy=False),
                                                 lat=Quantity(lat_mesh, 'rad', copy=False))

        # Interpolate irf's response onto tot_aeff's own (NuLambda, Ei)
        # grid one Epsilon bin center at a time, so the result stays on
        # irf's own native Epsilon binning.
        content_vs_epsilon = np.empty((len(nulambda_dir), len(target_ei_keV), len(epsilon_centers)))
        # In chunks of NuLambda, since interp()'s temporary arrays for the
        # whole grid at once take several GB
        for k, eps_k in enumerate(epsilon_centers):
            for rows in np.array_split(np.arange(len(nulambda_dir)), 16):
                eps_mesh = np.full_like(ei_mesh[rows], eps_k)
                content_vs_epsilon[rows, :, k] = irf_vs_epsilon.interp(photon_dir[rows], ei_mesh[rows], eps_mesh)

        fraction = IRFRelativeHistUnpolarized._selection_fraction(
            content_vs_epsilon, target_ei_keV, epsilon_centers, epsilon_widths, epsilon_edges, selector)

        return tot_aeff * fraction

    @staticmethod
    def _refine_ei_edges(ei_edges: np.ndarray, epsilon_edges: np.ndarray, epsilon_centers: np.ndarray,
                         selector: EnergySelector, rtol: float = 1e-6) -> np.ndarray:
        """
        ``ei_edges`` plus extra edges wherever a cut boundary ``E_cut``
        maps onto one of ``irf``'s ``Epsilon`` edges or centers, i.e. at
        ``Ei = E_cut / (1 + Epsilon)``.

        At fixed ``Ei`` the selection fraction integrates a density that
        is piecewise-linear between ``Epsilon`` centers (and clipped at
        the outer ``Epsilon`` edges), so as a function of ``Ei`` it has
        kinks exactly where a cut boundary crosses an ``Epsilon`` center
        or outer edge, and is smooth in between. Adding an edge at each
        of those, plus at each inner ``Epsilon`` edge (roughly halfway
        between consecutive kinks), gives about two ``Ei`` bins per
        ``Epsilon`` bin that the cut boundary sweeps through.

        Original edges are always kept, so ``Ei`` bins that no cut
        boundary maps into keep their exact edges (and centers).

        Parameters
        ----------
        ei_edges : numpy.ndarray
            The (unitless, keV) ``Ei`` edges to refine.
        epsilon_edges, epsilon_centers : numpy.ndarray
            ``irf``'s own ``Epsilon`` axis edges/centers.
        selector : EnergySelector
            The (possibly multi-range) combined energy selection.
        rtol : float, optional
            New edges closer than ``rtol`` (relative) to an existing edge,
            or to each other, are dropped, to avoid degenerate bins.

        Returns
        -------
        numpy.ndarray
            The refined, strictly increasing ``Ei`` edges.
        """

        ei_edges = np.asarray(ei_edges, dtype=float)

        cut_bounds = np.asarray(selector.energy_ranges_keV, dtype=float).ravel()
        cut_bounds = cut_bounds[np.isfinite(cut_bounds) & (cut_bounds > 0)]

        one_plus_eps = 1 + np.concatenate((epsilon_edges, epsilon_centers))
        one_plus_eps = one_plus_eps[one_plus_eps > 0]

        new = np.unique(cut_bounds[:, None] / one_plus_eps[None, :])
        new = new[(new > ei_edges[0]) & (new < ei_edges[-1])]

        # Drop new edges too close to an original one...
        idx = np.searchsorted(ei_edges, new)
        dist = np.minimum(np.abs(new - ei_edges[idx - 1]), np.abs(ei_edges[idx] - new))
        new = new[dist > rtol * new]

        # ... or to the previous kept new edge.
        if len(new):
            keep = [new[0]]
            for x in new[1:]:
                if x - keep[-1] > rtol * x:
                    keep.append(x)
            new = np.asarray(keep)

        return np.union1d(ei_edges, new)

    @staticmethod
    def _regrid_ei(tot_aeff: Histogram, new_ei_axis: Axis) -> Histogram:
        """
        ``tot_aeff`` (axes ``['NuLambda', 'Ei']``) re-sampled at
        ``new_ei_axis``'s centers, using the original ``Ei`` axis's own
        interpolation (``histpy.Axis.interp_weights``, i.e. log-domain for
        a log-scaled axis) -- the same one ``tot_aeff.interp()`` would use
        -- so bins whose center didn't move keep their value.
        """

        old_ei_axis = tot_aeff.axes['Ei']
        old_contents = np.asarray(tot_aeff.contents)

        if old_ei_axis.nbins == 1:
            contents = np.repeat(old_contents, new_ei_axis.nbins, axis=1)
        else:
            bins, weights = old_ei_axis.interp_weights(new_ei_axis.centers)
            contents = old_contents[:, bins[0]] * weights[0] + old_contents[:, bins[1]] * weights[1]

        return Histogram(Axes([tot_aeff.axes['NuLambda'], new_ei_axis]), contents=contents)

    @staticmethod
    def _integrate_piecewise_linear(values: np.ndarray, x_centers: np.ndarray,
                                     x_lo: float, x_hi: float) -> np.ndarray:
        """
        Definite integral, over the last axis, from ``x_lo`` to
        ``x_hi``, of the function that equals ``values`` at each of
        ``x_centers`` and linearly interpolates between consecutive
        centers -- the same model ``histpy.Histogram.interp()`` uses
        for a bin's stored value -- held flat (constant) beyond the
        first/last center. ``x_lo``/``x_hi`` may be anywhere; they are
        not assumed to already lie within ``[x_centers[0], x_centers[-1]]``.

        Parameters
        ----------
        values : numpy.ndarray
            Values sampled at ``x_centers``, along the last axis;
            leading axes are broadcast/vectorized over.
        x_centers : numpy.ndarray
            Strictly increasing sample points.
        x_lo, x_hi : float
            Integration bounds.

        Returns
        -------
        numpy.ndarray
            The definite integral, with ``values``'s leading shape.
        """

        if x_hi <= x_lo:
            return np.zeros(values.shape[:-1])

        total = np.zeros(values.shape[:-1])

        # Flat extrapolation below the first / above the last center.
        lo_flat_hi = min(x_hi, x_centers[0])
        if lo_flat_hi > x_lo:
            total = total + values[..., 0] * (lo_flat_hi - x_lo)

        hi_flat_lo = max(x_lo, x_centers[-1])
        if x_hi > hi_flat_lo:
            total = total + values[..., -1] * (x_hi - hi_flat_lo)

        # Interior: piecewise-linear interpolation between centers.
        interior_lo = max(x_lo, x_centers[0])
        interior_hi = min(x_hi, x_centers[-1])

        if interior_hi > interior_lo:
            interior_points = x_centers[(x_centers > interior_lo) & (x_centers < interior_hi)]
            breakpoints = np.concatenate(([interior_lo], interior_points, [interior_hi]))

            idx = np.clip(np.searchsorted(x_centers, breakpoints, side='right') - 1,
                          0, len(x_centers) - 2)
            x0, x1 = x_centers[idx], x_centers[idx + 1]
            w = (breakpoints - x0) / (x1 - x0)
            v = values[..., idx] * (1 - w) + values[..., idx + 1] * w

            total = total + np.trapezoid(v, breakpoints, axis=-1)

        return total

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

    @staticmethod
    def _open_h5_ei_range(filename, name: str, ei_min_keV: float, ei_max_keV: float) -> Histogram:
        """
        Read the histogram stored in group ``name`` of an HDF5 file,
        loading only the ``Ei`` bins that overlap
        ``[ei_min_keV, ei_max_keV]``, with an HDF5 hyperslab selection
        rather than reading the whole contents.

        Only the dense format (without overflow bins) is supported, and
        ``sumw2`` is not read.

        Raises
        ------
        ValueError
            If the group is not in that format, or no ``Ei`` bin
            overlaps the range.
        """

        with h5.File(filename, 'r') as f:
            group = f[name]

            if group.attrs.get('format', 'dense') != 'dense':
                raise ValueError(f"Only the dense histogram format is supported to load a range of Ei, "
                                 f"got '{group.attrs['format']}' in '{name}'")

            axes = Axes.open(group['axes'])
            ei_axis = axes['Ei']
            ei_idx = axes.label_to_index('Ei')

            if group['contents'].shape != tuple(axes.nbins):
                raise ValueError(f"Histograms with overflow bins are not supported to load a range of Ei ('{name}')")

            ei_edges = ei_axis.edges.to_value(u.keV)
            keep = np.flatnonzero((ei_edges[1:] > ei_min_keV) & (ei_edges[:-1] < ei_max_keV))

            if len(keep) == 0:
                raise ValueError(f"No Ei bin of '{name}' overlaps [{ei_min_keV}, {ei_max_keV}] keV")

            i0, i1 = keep[0], keep[-1] + 1

            index = [slice(None)] * axes.ndim
            index[ei_idx] = slice(i0, i1)
            contents = group['contents'][tuple(index)]

            unit = u.Unit(group.attrs['unit']) if 'unit' in group.attrs else None

        new_ei_axis = Axis(ei_axis.edges[i0:i1 + 1], label=ei_axis.label, scale=ei_axis.axis_scale)
        new_axes = Axes([new_ei_axis if i == ei_idx else axis for i, axis in enumerate(axes)])

        return Histogram(new_axes, contents=contents, unit=unit, copy_contents=False)

    @classmethod
    def from_h5(cls, filename, *args, **kwargs):
        """
        Construct an :class:`IRFRelativeHistUnpolarized` from an HDF5
        file that stores the response histogram under the group
        ``"IRF"``.

        If the file also has a group named ``"AEFF"``, it is read as
        well and passed to :meth:`__init__` as the ``aeff`` argument
        (see the class docstring), unless ``aeff`` was already provided
        via ``*args``/``**kwargs``, in which case that value takes
        precedence and the file is not checked for an ``"AEFF"``
        group.

        If ``selections`` is given (as a keyword argument), only the
        ``Ei`` bins that can produce a measured energy inside the
        selection are read from the file. A bin ``[a, b)`` can produce
        ``Em`` between ``a*(1 + eps_min)`` and ``b*(1 + eps_max)``, with
        ``eps_min``/``eps_max`` the edges of the ``Epsilon`` axis. The
        ``AEFF`` bins are read where they overlap the ``Ei`` range
        loaded for ``IRF``. Photons outside the loaded range have zero
        effective area and event probability, and those in the lowest and highest
        bins are interpolated with the neighboring bin missing, so
        they differ slightly from a full load. Only the dense
        format is supported in this case.

        Parameters
        ----------
        filename : str or path-like
            Path to the HDF5 file containing the response histogram
            (and, optionally, the separate total effective area
            histogram).
        *args, **kwargs
            Extra arguments forwarded verbatim to
            :meth:`__init__` (e.g. ``aeff``, ``copy``, ``nthreads``,
            ``npoints_parallel_thresh`` or ``selections``). ``copy``
            defaults to False here, since the histograms are read from
            disk and nobody else holds them.

        Returns
        -------
        IRFRelativeHistUnpolarized
            Initialized instance with the histogram(s) loaded from
            disk.
        """

        selector = cls._combine_selections(kwargs.get('selections'))

        if selector is None:
            def open_irf():
                return Histogram.open(filename, "IRF")

            def open_aeff(irf):
                return Histogram.open(filename, "AEFF")
        else:
            ranges = selector.energy_ranges_keV

            with h5.File(filename, 'r') as f:
                epsilon_edges = u.Quantity(Axes.open(f['IRF/axes'])['Epsilon'].edges).to_value('')

            # Em in [a*(1 + eps_min), b*(1 + eps_max)] for Ei in [a, b)
            ei_min = ranges[:, 0].min() / (1 + epsilon_edges[-1])
            ei_max = ranges[:, 1].max() / (1 + epsilon_edges[0]) if epsilon_edges[0] > -1 else np.inf

            def open_irf():
                return cls._open_h5_ei_range(filename, "IRF", ei_min, ei_max)

            def open_aeff(irf):
                ei_edges = irf.axes['Ei'].edges.to_value(u.keV)
                return cls._open_h5_ei_range(filename, "AEFF", ei_edges[0], ei_edges[-1])

        irf = open_irf()

        if 'aeff' not in kwargs and len(args) == 0:
            with h5.File(filename, 'r') as f:
                if 'AEFF' in f:
                    kwargs['aeff'] = open_aeff(irf)

        if len(args) < 2:
            kwargs.setdefault('copy', False)

        return cls(irf, *args, **kwargs)

    def _effective_area_cm2(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> Iterable[float]:
        """
        Total effective area, in cm^2, for each incident photon, after
        any energy ``selections``.

        Interpolates the total effective area (``_tot_aeff``) at the
        direction and energy of each photon in the list. It is zero for
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

        if self._interpolation == 'nearest' and self._has_selection:
            return self._effective_area_cm2_unselected(photons) * self._nearest_selected_fraction(photons)

        return self._interp_aeff(self._tot_aeff, photons)

    def _effective_area_cm2_unselected(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> Iterable[float]:
        """
        Same as :meth:`_effective_area_cm2`, but before any energy
        ``selections``.
        """

        return self._interp_aeff(self._aeff_unselected, photons)

    def _interp_aeff(self, aeff: Histogram, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> np.ndarray:
        """
        ``aeff`` (axes ``['NuLambda', 'Ei']``) interpolated at each
        photon, and zero outside the range of its ``Ei`` axis.
        """

        photon_lon_rad, photon_lat_rad, photon_energy_keV = self._photon_list_to_raw_values(photons)

        def build_args(lon_rad, lat_rad, energy_keV):
            photon_dir = UnitSphericalRepresentation(lon=Quantity(lon_rad, 'rad', copy=False),
                                                     lat=Quantity(lat_rad, 'rad', copy=False))
            return photon_dir, energy_keV

        values = np.asarray(self._parallel_interp(aeff, build_args,
                                                  (photon_lon_rad, photon_lat_rad, photon_energy_keV)))

        return np.where(self._in_ei_range(aeff, photon_energy_keV), values, 0.0)

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

        It is zero where ``Epsilon`` or ``Ei`` are outside the range of
        their axes, instead of extending the density of the edge bins.
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

        relcoords = RelativeCDSCoordinates(photon_dir.to_cartesian().xyz, pol_convention=self._pol_convention)
        phi_geo, zeta = relcoords.to_relative(psichi_dir.to_cartesian().xyz)

        phi_geo_rad = phi_geo.to_value(u.rad)

        # RelativeCDSCoordinates.to_relative() returns az in (-pi, pi], but
        # the Zeta axis spans [0, 2*pi) and, once converted to a plain
        # (non-circular) Axis in __init__, clamps rather than wraps
        # out-of-range values. Wrap into [0, 2*pi) so negative az values
        # land on their correct bin instead of being clamped to Zeta = 0.
        zeta_rad = zeta.to_value(u.rad) % (2 * np.pi)

        theta_rad = phi_geo_rad - phi_kin_rad

        # The relative-coordinate math above is already fully vectorized
        # over the whole input and cheap relative to the interpolation
        # below, so only the interp() call itself is (optionally)
        # parallelized -- see _parallel_interp.
        if self._interpolation == 'nearest':
            return self._nearest_lookup(photon_dir, photon_energy_keV, epsilon, phi_kin_rad, theta_rad, zeta_rad)

        def build_args(lon_rad, lat_rad, energy_keV, eps, phi, theta, zeta_r):
            photon_dir_chunk = UnitSphericalRepresentation(lon=Quantity(lon_rad, 'rad', copy=False),
                                                            lat=Quantity(lat_rad, 'rad', copy=False))
            return photon_dir_chunk, energy_keV, eps, phi, theta, zeta_r

        prob = np.asarray(self._parallel_interp(self._event_prob, build_args,
                                                (photon_lon_rad, photon_lat_rad, photon_energy_keV,
                                                 epsilon, phi_kin_rad, theta_rad, zeta_rad)))

        # interp() clamps to the edge bins, which would extend their density
        # indefinitely
        epsilon_edges = self._event_prob.axes['Epsilon'].edges
        in_range = ((epsilon >= epsilon_edges[0]) & (epsilon <= epsilon_edges[-1])
                    & self._in_ei_range(self._event_prob, photon_energy_keV))

        return np.where(in_range, prob, 0.0)

    @staticmethod
    def _nearest_bin(axis, values):
        """
        Index of the bin of ``axis`` that contains each of ``values`` (clipped to the
        first/last bin), and whether each value is within the axis' range.
        """

        edges = np.asarray(axis.edges, dtype=float)
        values = np.asarray(values, dtype=float)
        index = np.clip(np.searchsorted(edges, values, side='right') - 1, 0, len(edges) - 2)

        return index, (values >= edges[0]) & (values <= edges[-1])

    def _nearest_photon_bins(self, photon_dir, energy_keV):
        """
        ``NuLambda`` pixel and ``Ei`` bin of each photon, and whether its energy is within
        the range of the ``Ei`` axis.
        """

        axes = self._event_prob.axes
        pix = np.asarray(axes['NuLambda'].find_bin(photon_dir))
        iei, ok = self._nearest_bin(axes['Ei'], energy_keV)

        return pix, iei, ok

    def _nearest_lookup(self, photon_dir, energy_keV, epsilon, phi, theta, zeta):
        """
        Value of ``_event_prob`` in the bin that each (photon, event) pair lands in, zero
        outside the axes' ranges.

        Each ``Epsilon`` bin spans ``Ei * dEpsilon`` in ``Em``, so the value is
        rescaled by ``Ei_center / Ei`` to stay normalized at any ``Ei`` within the bin.
        """

        axes = self._event_prob.axes
        pix, iei, ok = self._nearest_photon_bins(photon_dir, energy_keV)
        index = [pix, iei]

        for label, values in zip(['Epsilon', 'Phi', 'Theta', 'Zeta'], [epsilon, phi, theta, zeta]):
            i, in_range = self._nearest_bin(axes[label], values)
            index.append(i)
            ok &= in_range

        prob = self._event_prob.contents[tuple(index)] * np.asarray(axes['Ei'].centers)[iei] / energy_keV

        return np.where(ok, prob, 0.0)

    def _nearest_selected_fraction(self, photons):
        """
        Fraction of each photon's effective area with ``Em`` inside the selections, from
        the content of its own ``(NuLambda, Ei)`` bin. ``Epsilon`` bins that are partially
        inside a range count by their overlap. Zero for photons outside the ``Ei`` axis.
        """

        photon_lon_rad, photon_lat_rad, energy_keV = self._photon_list_to_raw_values(photons)
        photon_dir = UnitSphericalRepresentation(lon=Quantity(photon_lon_rad, 'rad', copy=False),
                                                 lat=Quantity(photon_lat_rad, 'rad', copy=False))
        pix, iei, ok = self._nearest_photon_bins(photon_dir, energy_keV)

        eps_edges = np.asarray(self._event_prob.axes['Epsilon'].edges, dtype=float)
        energy_keV = np.asarray(energy_keV, dtype=float)

        weights = np.zeros((len(energy_keV), len(eps_edges) - 1))
        for lo, hi in self._selection_ranges:
            overlap = (np.minimum(hi / energy_keV[:, None] - 1, eps_edges[None, 1:])
                       - np.maximum(lo / energy_keV[:, None] - 1, eps_edges[None, :-1]))
            weights += np.clip(overlap, 0, None) / np.diff(eps_edges)[None, :]

        proj = self._nearest_proj[pix, iei]
        total = proj.sum(axis=1)
        fraction = np.divide((proj * weights).sum(axis=1), total, out=np.zeros_like(total), where=total > 0)

        return np.where(ok, fraction, 0.0)

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
        ``_event_prob`` is divided by the selected fraction of the
        photon's effective area (``_effective_area_cm2`` over
        ``_effective_area_cm2_unselected``). Where that fraction is zero
        (e.g. photon energy or direction outside the region covered by
        the histogram) the event is impossible, so the probability is
        zero instead of 0/0 = NaN or x/0 = inf.
        """

        prob = self._interp_event_prob(photons, events)

        if not self._has_selection:
            return prob

        selected = self._effective_area_cm2(photons)
        total = self._effective_area_cm2_unselected(photons)

        # prob / (selected / total)
        return np.divide(prob * total, selected, out=np.zeros_like(prob), where=selected > 0)

    def _random_events(self, photons: PhotonListWithDirectionInSCFrameInterface) -> EventDataInterface:
        """
        Not implemented yet; provided to satisfy the
        :class:`FarFieldSpectralInstrumentResponseFunctionInterface`
        contract.
        """
        raise NotImplementedError("random_events not implemented yet.")

