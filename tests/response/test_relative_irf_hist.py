from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import UnitSphericalRepresentation
from astropy.units import Quantity
from histpy import Axis, Axes, HealpixAxis, Histogram
from scoords import SpacecraftFrame

from cosipy.event_selection.distance_selection import DistanceSelector
from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.polarization import PolarizationAxis, StereographicConvention
from cosipy.response.relative_coordinates import RelativeCDSCoordinates
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized


def _make_irf_hist(nside=2, seed=0):
    rng = np.random.default_rng(seed)

    axes = Axes([
        HealpixAxis(nside=nside, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis(np.geomspace(100, 1000, 4) * u.keV, label='Ei', scale='log'),
        Axis(np.linspace(-0.5, 0.5, 5), label='Epsilon'),
        Axis(np.linspace(0, 180, 7) * u.deg, label='Phi'),
        Axis(np.linspace(-90, 90, 5) * u.deg, label='Theta'),
        PolarizationAxis(np.linspace(0, 360, 9) * u.deg, convention=StereographicConvention(), label='Zeta'),
    ])

    contents = rng.random(axes.nbins)

    return Histogram(axes, contents=contents, unit=u.cm * u.cm)


def _make_aeff_hist(nside=4, seed=1):
    rng = np.random.default_rng(seed)

    axes = Axes([
        HealpixAxis(nside=nside, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis(np.geomspace(50, 2000, 6) * u.keV, label='Ei', scale='log'),
    ])

    contents = rng.random(axes.nbins) * 100

    return Histogram(axes, contents=contents, unit=u.cm * u.cm)


class TestFromH5Aeff:

    def test_auto_detects_aeff_group(self, tmp_path):
        """If the file has an "AEFF" group, from_h5() should use it as the
        total effective area, even though its NuLambda/Ei grid differs
        from the irf's own."""

        path = tmp_path / "irf_with_aeff.h5"
        irf_hist = _make_irf_hist()
        aeff_hist = _make_aeff_hist()

        irf_hist.write(str(path), name="IRF", overwrite=True)
        aeff_hist.write(str(path), name="AEFF", overwrite=True)

        model = IRFRelativeHistUnpolarized.from_h5(path)

        assert model._tot_aeff.axes.labels.tolist() == ['NuLambda', 'Ei']
        assert model._tot_aeff.contents.shape == aeff_hist.contents.shape

    def test_falls_back_to_projection_without_aeff_group(self, tmp_path):
        """Without an "AEFF" group, from_h5() should behave as before:
        the total effective area is the irf's own NuLambda/Ei projection."""

        path = tmp_path / "irf_only.h5"
        irf_hist = _make_irf_hist()
        irf_hist.write(str(path), name="IRF", overwrite=True)

        model = IRFRelativeHistUnpolarized.from_h5(path)

        assert model._tot_aeff.contents.shape == (irf_hist.axes['NuLambda'].nbins, irf_hist.axes['Ei'].nbins)

    def test_explicit_aeff_kwarg_takes_precedence(self, tmp_path):
        """An aeff passed explicitly to from_h5() should win over any
        "AEFF" group present in the file."""

        path = tmp_path / "irf_with_aeff.h5"
        irf_hist = _make_irf_hist()
        file_aeff_hist = _make_aeff_hist(nside=4)
        irf_hist.write(str(path), name="IRF", overwrite=True)
        file_aeff_hist.write(str(path), name="AEFF", overwrite=True)

        override_aeff_hist = _make_aeff_hist(nside=8, seed=2)

        model = IRFRelativeHistUnpolarized.from_h5(path, aeff=override_aeff_hist)

        assert model._tot_aeff.contents.shape == override_aeff_hist.contents.shape
        assert model._tot_aeff.contents.shape != file_aeff_hist.contents.shape

    def test_explicit_positional_aeff_takes_precedence(self, tmp_path):
        """Same as above, but with aeff passed positionally."""

        path = tmp_path / "irf_with_aeff.h5"
        irf_hist = _make_irf_hist()
        file_aeff_hist = _make_aeff_hist(nside=4)
        irf_hist.write(str(path), name="IRF", overwrite=True)
        file_aeff_hist.write(str(path), name="AEFF", overwrite=True)

        override_aeff_hist = _make_aeff_hist(nside=8, seed=3)

        model = IRFRelativeHistUnpolarized.from_h5(path, override_aeff_hist)

        assert model._tot_aeff.contents.shape == override_aeff_hist.contents.shape


class _FakePhotonList:
    def __init__(self, lon_rad, lat_rad, energy_keV):
        self.direction_lon_rad_sc = lon_rad
        self.direction_lat_rad_sc = lat_rad
        self.energy_keV = energy_keV


class _FakeEventData:
    def __init__(self, lon_rad, lat_rad, phi_rad, energy_keV):
        self.scattered_lon_rad_sc = lon_rad
        self.scattered_lat_rad_sc = lat_rad
        self.scattering_angle_rad = phi_rad
        self.energy_keV = energy_keV


def _make_photons_and_events(n, seed=42):
    """Random, but physically-plausible-enough photon/event arrays: energies
    within the irf's Ei axis range, directions on the sphere, scattering
    angles in [0, pi]. Values don't need to be physically self-consistent
    (e.g. the event doesn't need to be a real Compton-scattered version of
    the photon) -- these tests only check that parallelizing the interp()
    call doesn't change its result, not that the result is physically
    correct (that's covered elsewhere by the class's other tests/usage)."""

    rng = np.random.default_rng(seed)

    photon_lon = rng.uniform(-np.pi, np.pi, n)
    photon_lat = rng.uniform(-np.pi / 2, np.pi / 2, n)
    photon_energy = rng.uniform(100, 1000, n)

    psichi_lon = rng.uniform(-np.pi, np.pi, n)
    psichi_lat = rng.uniform(-np.pi / 2, np.pi / 2, n)
    phi_kin = rng.uniform(0, np.pi, n)
    measured_energy = rng.uniform(100, 1000, n)

    photons = _FakePhotonList(photon_lon, photon_lat, photon_energy)
    events = _FakeEventData(psichi_lon, psichi_lat, phi_kin, measured_energy)

    return photons, events


class TestParallelInterp:
    """nthreads > 1 (with a small npoints_parallel_thresh so the parallel
    path actually triggers at small, fast-to-test N) must give the same
    results as
    nthreads=1 -- chunking the input and concatenating the per-chunk
    results should be numerically transparent, since each point's
    interpolation is independent of every other point's."""

    def test_executor_only_created_when_nthreads_greater_than_one(self):
        serial_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=1)
        assert serial_model._executor is None

        parallel_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=4)
        assert isinstance(parallel_model._executor, ThreadPoolExecutor)

    def test_effective_area_cm2_matches_serial(self):
        # Separate Histogram instances per model rather than sharing one
        # via .copy(): Histogram.copy() doesn't deep-copy its individual
        # Axis objects, and IRFRelativeHistUnpolarized.__init__ mutates
        # them in place (unit standardization) even when copy=True, so two
        # models built from copies of the *same* underlying histogram
        # would corrupt each other's axes.
        serial_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=1)
        # npoints_parallel_thresh=1 -> threshold is nthreads*thresh=4
        # points, so the parallel path triggers well below the N used here.
        parallel_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=4, npoints_parallel_thresh=1)

        # N=37: deliberately not a multiple of nthreads, to exercise
        # np.array_split's uneven-chunk behavior.
        photons, _ = _make_photons_and_events(37)

        serial_result = serial_model._effective_area_cm2(photons)
        parallel_result = parallel_model._effective_area_cm2(photons)

        np.testing.assert_allclose(parallel_result, serial_result)

    def test_differential_effective_area_cm2_matches_serial(self):
        serial_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=1)
        parallel_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=4, npoints_parallel_thresh=1)

        photons, events = _make_photons_and_events(37)

        serial_result = serial_model._differential_effective_area_cm2(photons, events)
        parallel_result = parallel_model._differential_effective_area_cm2(photons, events)

        np.testing.assert_allclose(parallel_result, serial_result)

    def test_matches_serial_below_npoints_parallel_thresh(self):
        """Same as above, but with N below nthreads*npoints_parallel_thresh,
        so the parallel model's _parallel_interp() takes its direct,
        single-threaded branch -- should trivially still match, and
        confirms that branch is exercised too."""

        serial_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=1)
        parallel_model = IRFRelativeHistUnpolarized(_make_irf_hist(), nthreads=4, npoints_parallel_thresh=1000)

        photons, events = _make_photons_and_events(5)

        np.testing.assert_allclose(
            parallel_model._effective_area_cm2(photons),
            serial_model._effective_area_cm2(photons))
        np.testing.assert_allclose(
            parallel_model._differential_effective_area_cm2(photons, events),
            serial_model._differential_effective_area_cm2(photons, events))


def _make_uniform_irf_hist(nside=2, n_eps=8, eps_range=(-0.8, 0.8)):
    """A single-Ei-bin irf histogram with all contents equal to 1 cm^2
    and a uniform Epsilon axis, so the selection fraction for any cut
    has an exact, hand-computable closed form (constant density ->
    the piecewise-linear integration model is exact, not just an
    approximation)."""

    axes = Axes([
        HealpixAxis(nside=nside, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis([100., 1000.] * u.keV, label='Ei'),
        Axis(np.linspace(*eps_range, n_eps + 1), label='Epsilon'),
        Axis(np.linspace(0, 180, 7) * u.deg, label='Phi'),
        Axis(np.linspace(-90, 90, 5) * u.deg, label='Theta'),
        PolarizationAxis(np.linspace(0, 360, 9) * u.deg, convention=StereographicConvention(), label='Zeta'),
    ])

    return Histogram(axes, contents=np.ones(axes.nbins), unit=u.cm * u.cm)


def _make_peaked_irf_hist(nside=1, ei_edges_keV=(100., 178., 316., 562., 1000.),
                          eps_sigma=0.01, n_eps=100, eps_range=(-0.5, 0.5)):
    """A multi-Ei-bin irf histogram whose Epsilon distribution is a narrow
    Gaussian around eps=0, identical (by construction) at every Ei bin --
    i.e. the energy resolution itself does not depend on Ei at all. Any
    Ei-dependence of a measured-energy selection's *fraction* therefore
    comes purely from the Em -> Epsilon = Em/Ei - 1 conversion, not from a
    genuinely varying response, which is what lets
    test_narrow_cut_matches_exact_fraction_at_target_ei below compute an
    exact expected answer independent of Ei interpolation."""

    eps_edges = np.linspace(*eps_range, n_eps + 1)
    eps_centers = 0.5 * (eps_edges[:-1] + eps_edges[1:])
    eps_profile = np.exp(-0.5 * (eps_centers / eps_sigma) ** 2)
    eps_profile /= eps_profile.sum()  # each Ei/pixel/Phi/Theta/Zeta slice sums to 1

    axes = Axes([
        HealpixAxis(nside=nside, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis(np.asarray(ei_edges_keV) * u.keV, label='Ei', scale='log'),
        Axis(eps_edges, label='Epsilon'),
        Axis([0., 180.] * u.deg, label='Phi'),
        Axis([-90., 90.] * u.deg, label='Theta'),
        PolarizationAxis(np.linspace(0, 360, 5) * u.deg, convention=StereographicConvention(), label='Zeta'),
    ])

    contents = np.broadcast_to(
        eps_profile[None, None, :, None, None, None],
        axes.nbins).copy()

    return Histogram(axes, contents=contents, unit=u.cm * u.cm)


def _make_fine_flat_aeff_hist(nside=1, n_ei=201, ei_range_keV=(100., 1000.)):
    """A total-effective-area histogram, flat in Ei and much finer than
    _make_peaked_irf_hist's own (coarse) Ei grid."""

    axes = Axes([
        HealpixAxis(nside=nside, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis(np.geomspace(*ei_range_keV, n_ei + 1) * u.keV, label='Ei', scale='log'),
    ])

    return Histogram(axes, contents=np.ones(axes.nbins), unit=u.cm * u.cm)


class TestEnergySelections:
    """selections=... on IRFRelativeHistUnpolarized should scale
    _tot_aeff, per (NuLambda, Ei), by the fraction of effective area
    whose measured energy falls inside the selection -- and never
    touch _diff_aeff."""

    def test_non_energy_selector_raises_type_error(self):
        with pytest.raises(TypeError):
            IRFRelativeHistUnpolarized(_make_irf_hist(), selections=DistanceSelector())

    def test_selections_none_matches_omitting_the_parameter(self):
        baseline = IRFRelativeHistUnpolarized(_make_irf_hist(seed=1))
        explicit_none = IRFRelativeHistUnpolarized(_make_irf_hist(seed=1), selections=None)

        np.testing.assert_array_equal(explicit_none._tot_aeff.contents, baseline._tot_aeff.contents)

    def test_full_range_cut_leaves_tot_aeff_unchanged(self):
        """A cut wide enough to enclose every Ei bin's full measured-energy
        range should hit the exact fast path -- _tot_aeff byte-for-byte
        unchanged, not just approximately."""

        baseline = IRFRelativeHistUnpolarized(_make_irf_hist(seed=2))
        full_range = EnergySelector(u.Quantity([0., 1e9], u.keV))
        cut = IRFRelativeHistUnpolarized(_make_irf_hist(seed=2), selections=full_range)

        np.testing.assert_array_equal(cut._tot_aeff.contents, baseline._tot_aeff.contents)

    def test_uniform_binning_exact_fraction_bin_aligned_cut(self):
        """Ei center is 550 keV; Epsilon edges are spaced by 0.2, so a
        [330, 550) keV cut is exactly eps in [-0.4, 0.0) -- two whole
        bins, no boundary approximation."""

        n_phi, n_theta, n_zeta, n_eps = 6, 4, 8, 8
        model = IRFRelativeHistUnpolarized(
            _make_uniform_irf_hist(n_eps=n_eps), copy=False,
            selections=EnergySelector(u.Quantity([330., 550.], u.keV)))

        expected = 2 * n_phi * n_theta * n_zeta
        np.testing.assert_allclose(model._tot_aeff.contents, expected)

    def test_uniform_binning_exact_fraction_sub_bin_cut(self):
        """[385, 550) keV -> eps in [-0.3, 0.0), i.e. half of the
        [-0.4,-0.2) bin plus the whole [-0.2,0) bin. Density is uniform
        (content=1 in every bin), so the exact answer is just
        density * width, regardless of the boundary falling mid-bin."""

        n_phi, n_theta, n_zeta = 6, 4, 8
        model = IRFRelativeHistUnpolarized(
            _make_uniform_irf_hist(), copy=False,
            selections=EnergySelector(u.Quantity([385., 550.], u.keV)))

        density_per_solid_angle_bin = 1 / 0.2  # content=1 per Epsilon bin of width 0.2
        expected = density_per_solid_angle_bin * 0.3 * n_phi * n_theta * n_zeta
        np.testing.assert_allclose(model._tot_aeff.contents, expected)

    def test_tuple_of_selectors_combines_via_union(self):
        """A tuple of EnergySelectors passed to selections= is OR'd
        together (each entry is an acceptable window), matching
        EnergySelector.union -- not intersected/AND'd."""

        a = EnergySelector(u.Quantity([100., 500.], u.keV))
        b = EnergySelector(u.Quantity([400., 900.], u.keV))  # overlaps a

        via_tuple = IRFRelativeHistUnpolarized(_make_irf_hist(seed=3), selections=(a, b))
        via_union = IRFRelativeHistUnpolarized(_make_irf_hist(seed=3), selections=a.union(b))
        via_intersect = IRFRelativeHistUnpolarized(_make_irf_hist(seed=3), selections=a.intersect(b))

        np.testing.assert_allclose(via_tuple._tot_aeff.contents, via_union._tot_aeff.contents)
        assert not np.allclose(via_tuple._tot_aeff.contents, via_intersect._tot_aeff.contents)

    def test_multi_range_selector_sums_disjoint_windows(self):
        """A single EnergySelector with two disjoint ranges should give
        the same total as summing each range's own single-range cut
        (they don't overlap, so no double-counting)."""

        n_phi, n_theta, n_zeta = 6, 4, 8

        range_a = EnergySelector(u.Quantity([330., 470.], u.keV))  # eps in [-0.4, -0.145...)
        range_b = EnergySelector(u.Quantity([600., 750.], u.keV))  # eps in [0.0909..., 0.3636...)
        both = range_a.union(range_b)

        model_a = IRFRelativeHistUnpolarized(_make_uniform_irf_hist(), copy=False, selections=range_a)
        model_b = IRFRelativeHistUnpolarized(_make_uniform_irf_hist(), copy=False, selections=range_b)
        model_both = IRFRelativeHistUnpolarized(_make_uniform_irf_hist(), copy=False, selections=both)

        np.testing.assert_allclose(model_both._tot_aeff.contents,
                                    model_a._tot_aeff.contents + model_b._tot_aeff.contents)

    def test_aeff_and_selections_together_interpolates_onto_aeff_grid(self):
        no_cut = IRFRelativeHistUnpolarized(_make_irf_hist(seed=4), aeff=_make_aeff_hist(seed=5))
        cut = IRFRelativeHistUnpolarized(
            _make_irf_hist(seed=4), aeff=_make_aeff_hist(seed=5),
            selections=EnergySelector(u.Quantity([150., 400.], u.keV)))

        # The cut refines the Ei grid near its boundaries, but never
        # drops an original edge.
        cut_edges = np.asarray(cut._tot_aeff.axes['Ei'].edges)
        no_cut_edges = np.asarray(no_cut._tot_aeff.axes['Ei'].edges)
        assert len(cut_edges) > len(no_cut_edges)
        assert np.all(np.isin(no_cut_edges, cut_edges))

        no_cut_on_cut_grid = IRFRelativeHistUnpolarized._regrid_ei(no_cut._tot_aeff, cut._tot_aeff.axes['Ei'])
        assert np.all(cut._tot_aeff.contents <= no_cut_on_cut_grid.contents + 1e-9)
        assert cut._tot_aeff.contents.sum() < no_cut_on_cut_grid.contents.sum()

    def test_diff_aeff_unaffected_by_selections(self):
        no_cut = IRFRelativeHistUnpolarized(_make_irf_hist(seed=6))
        cut = IRFRelativeHistUnpolarized(
            _make_irf_hist(seed=6),
            selections=EnergySelector(u.Quantity([150., 400.], u.keV)))

        np.testing.assert_array_equal(cut._event_prob.contents, no_cut._event_prob.contents)

    def test_narrow_cut_matches_exact_fraction_at_target_ei(self):
        """A narrow measured-energy cut, applied with a separate `aeff`
        on a much finer Ei grid than irf's own (coarse) one, must use
        each of aeff's own *exact* Ei values for the Em -> Epsilon
        conversion -- not irf's own (coarser) Ei grid, whose derived
        fraction-vs-Ei curve is a poor thing to linearly interpolate for
        a narrow cut, since Em/Ei - 1 makes the corresponding Epsilon
        window shift and rescale rapidly with Ei even when (as here) the
        underlying response doesn't vary with Ei at all."""

        eps_sigma = 0.01
        irf_hist = _make_peaked_irf_hist(eps_sigma=eps_sigma)
        aeff_hist = _make_fine_flat_aeff_hist()

        model = IRFRelativeHistUnpolarized(
            irf_hist, aeff=aeff_hist,
            selections=EnergySelector(u.Quantity([495., 505.], u.keV)))

        eps_edges = np.asarray(irf_hist.axes['Epsilon'].edges)
        eps_centers = 0.5 * (eps_edges[:-1] + eps_edges[1:])
        eps_widths = np.diff(eps_edges)
        eps_profile = np.exp(-0.5 * (eps_centers / eps_sigma) ** 2)
        eps_profile /= eps_profile.sum()
        density = eps_profile / eps_widths

        def exact_fraction(ei_keV):
            eps_lo = np.clip(495. / ei_keV - 1, eps_edges[0], eps_edges[-1])
            eps_hi = np.clip(505. / ei_keV - 1, eps_edges[0], eps_edges[-1])
            selected = IRFRelativeHistUnpolarized._integrate_piecewise_linear(
                density, eps_centers, eps_lo, eps_hi)
            return selected  # eps_profile already sums to 1, so this IS the fraction

        # _tot_aeff's Ei grid is aeff's, refined near the cut.
        target_ei_keV = np.asarray(model._tot_aeff.axes['Ei'].centers)
        expected_fraction = np.array([exact_fraction(ei) for ei in target_ei_keV])

        # aeff_hist is flat (all ones), so _tot_aeff post-cut equals the
        # fraction directly, for every NuLambda pixel.
        for pix in range(model._tot_aeff.contents.shape[0]):
            np.testing.assert_allclose(model._tot_aeff.contents[pix], expected_fraction, atol=1e-6)

    @pytest.mark.parametrize('cut_keV', [(495., 505.), (400., 600.)])
    def test_coarse_aeff_grid_is_refined_to_resolve_cut(self, cut_keV):
        """On an aeff Ei grid much coarser than the fraction's own
        variation (~Ei * dEpsilon), sampling the fraction only at aeff's
        bin centers and linearly interpolating _tot_aeff between them
        misrepresents a narrow cut badly -- for (495, 505) keV on this
        grid, its integral over Ei comes out ~2x too large. The grid must
        be refined so the interpolated _tot_aeff tracks the exact fraction
        everywhere, not just at the grid points."""

        lo, hi = cut_keV
        eps_sigma = 0.01
        irf_hist = _make_peaked_irf_hist(eps_sigma=eps_sigma)
        aeff_hist = _make_fine_flat_aeff_hist(n_ei=40, ei_range_keV=(50., 10000.))

        model = IRFRelativeHistUnpolarized(irf_hist, aeff=aeff_hist,
                                           selections=EnergySelector(u.Quantity([lo, hi], u.keV)))

        eps_edges = np.asarray(irf_hist.axes['Epsilon'].edges)
        eps_centers = 0.5 * (eps_edges[:-1] + eps_edges[1:])
        eps_profile = np.exp(-0.5 * (eps_centers / eps_sigma) ** 2)
        eps_profile /= eps_profile.sum()
        density = eps_profile / np.diff(eps_edges)

        def exact_fraction(ei_keV):
            return IRFRelativeHistUnpolarized._integrate_piecewise_linear(
                density, eps_centers,
                np.clip(lo / ei_keV - 1, eps_edges[0], eps_edges[-1]),
                np.clip(hi / ei_keV - 1, eps_edges[0], eps_edges[-1]))

        ei_axis = model._tot_aeff.axes['Ei']
        fine_ei = np.geomspace(300., 800., 5001)
        bins, weights = ei_axis.interp_weights(fine_ei)
        interpolated = (model._tot_aeff.contents[0][bins[0]] * weights[0]
                        + model._tot_aeff.contents[0][bins[1]] * weights[1])
        expected = np.array([exact_fraction(ei) for ei in fine_ei])

        np.testing.assert_allclose(interpolated, expected, atol=0.02)
        np.testing.assert_allclose(np.trapezoid(interpolated, fine_ei),
                                   np.trapezoid(expected, fine_ei), rtol=0.01)

    def test_refinement_only_splits_bins_the_cut_can_reach(self):
        """New edges only go where a cut boundary maps into Ei (Ei =
        E_cut / (1 + Epsilon)); every original edge is kept, so bins
        outside that range keep their exact edges and centers."""

        irf_hist = _make_peaked_irf_hist()  # Epsilon in [-0.5, 0.5]
        aeff_hist = _make_fine_flat_aeff_hist(n_ei=40, ei_range_keV=(50., 10000.))
        orig_edges = aeff_hist.axes['Ei'].edges.to_value(u.keV)
        orig_centers = aeff_hist.axes['Ei'].centers.to_value(u.keV)

        model = IRFRelativeHistUnpolarized(irf_hist, aeff=aeff_hist,
                                           selections=EnergySelector(u.Quantity([495., 505.], u.keV)))

        new_edges = np.asarray(model._tot_aeff.axes['Ei'].edges)
        new_centers = np.asarray(model._tot_aeff.axes['Ei'].centers)
        assert np.all(np.isin(orig_edges, new_edges))

        # Epsilon in [-0.5, 0.5] -> Em in [495, 505] keV only comes from
        # Ei in [330, 1010] keV.
        added = np.setdiff1d(new_edges, orig_edges)
        assert len(added) > 0
        assert added.min() >= 495. / 1.5 and added.max() <= 505. / 0.5

        untouched = (orig_edges[1:] <= added.min()) | (orig_edges[:-1] >= added.max())
        assert np.all(np.isin(orig_centers[untouched], new_centers))

    def test_full_range_cut_with_aeff_leaves_grid_unchanged(self):
        """A cut whose boundaries can't map into aeff's Ei range adds no
        edges."""

        irf_hist = _make_peaked_irf_hist()
        aeff_hist = _make_fine_flat_aeff_hist(n_ei=40, ei_range_keV=(50., 10000.))
        orig_edges = np.asarray(aeff_hist.axes['Ei'].edges.to_value(u.keV))

        model = IRFRelativeHistUnpolarized(irf_hist, aeff=aeff_hist,
                                           selections=EnergySelector(u.Quantity([0., 1e9], u.keV)))

        np.testing.assert_array_equal(np.asarray(model._tot_aeff.axes['Ei'].edges), orig_edges)
        np.testing.assert_allclose(model._tot_aeff.contents, 1.)


class TestUnphysicalBins:
    """Bins in the unphysical region of the CDS reparametrization (Phi + Theta
    outside [0, pi]) have zero phase space. Non-zero contents there (e.g. from
    smoothing) must not end up as huge/infinite differential effective area,
    which overflows to inf/NaN during interpolation."""

    def test_nonzero_contents_in_zero_phase_space_bins_are_zeroed(self):
        model = IRFRelativeHistUnpolarized(_make_irf_hist())

        diff_aeff = model._event_prob.contents

        assert np.all(np.isfinite(diff_aeff))
        # Random contents in [0, 1) divided by finite phase spaces stay moderate
        assert diff_aeff.max() < 1e6

    def test_differential_effective_area_is_finite(self):
        model = IRFRelativeHistUnpolarized(_make_irf_hist())

        photons, events = _make_photons_and_events(500)

        assert np.all(np.isfinite(model._differential_effective_area_cm2(photons, events)))


class TestZeroEffectiveArea:
    """Where the total effective area is zero the event probability
    (diff. area / area) must be zero, not NaN/inf, otherwise a few NaNs in
    the per-event density break the likelihood fit."""

    def test_differential_area_is_zero_where_effective_area_is_zero(self):
        aeff = _make_aeff_hist()
        aeff = Histogram(aeff.axes, contents=np.zeros(aeff.axes.nbins), unit=u.cm * u.cm)
        model = IRFRelativeHistUnpolarized(_make_irf_hist(), aeff=aeff)

        photons, events = _make_photons_and_events(200)

        diff = np.asarray(model._differential_effective_area_cm2(photons, events), dtype=float)

        assert np.all(diff == 0)

    def test_conditional_probability_is_zero_where_selected_area_is_zero(self):
        """A cut that nothing passes: no NaN/inf, and zero probability."""

        model = IRFRelativeHistUnpolarized(_make_irf_hist(),
                                           selections=EnergySelector(u.Quantity([5000., 6000.], u.keV)))

        photons, events = _make_photons_and_events(200)

        prob = np.asarray(model._event_probability(photons, events), dtype=float)

        assert np.all(np.isfinite(prob))
        assert np.all(prob == 0)

    def test_event_probability_matches_ratio_where_effective_area_is_positive(self):
        model = IRFRelativeHistUnpolarized(_make_irf_hist())

        photons, events = _make_photons_and_events(200)

        prob = np.asarray(model._event_probability(photons, events), dtype=float)
        expected = (np.asarray(model._differential_effective_area_cm2(photons, events))
                    / np.asarray(model._effective_area_cm2(photons)))

        assert np.allclose(prob, expected)


def _one_photon_events(energy_keV, eps, n=1, seed=3):
    """One photon repeated n times, with events at the given Epsilon and a
    common scattered direction and angle."""

    rng = np.random.default_rng(seed)
    lon, lat, phi = rng.uniform(-np.pi, np.pi), rng.uniform(-1, 1), rng.uniform(0, np.pi)
    photons = _FakePhotonList(np.full(n, 0.3), np.full(n, 0.2), np.full(n, float(energy_keV)))
    events = _FakeEventData(np.full(n, lon), np.full(n, lat), np.full(n, phi),
                            energy_keV * (1 + np.asarray(eps, float)))
    return photons, events


class TestDifferentialEffectiveAreaNormalization:

    def test_zero_outside_epsilon_axis_range(self):
        """The Epsilon axis of _make_irf_hist spans [-0.5, 0.5]."""

        model = IRFRelativeHistUnpolarized(_make_irf_hist())
        eps = np.array([-3., -0.51, 0.51, 66.])
        photons, events = _one_photon_events(300., eps, n=len(eps))

        np.testing.assert_array_equal(model._differential_effective_area_cm2(photons, events), 0)

    def test_unchanged_inside_epsilon_axis_range(self):
        """Up to the edges of the axis, the density is the interpolated one."""

        model = IRFRelativeHistUnpolarized(_make_irf_hist())
        photons, events = _one_photon_events(300., [-0.5, -0.4999999, 0.4999999, 0.5], n=4)

        result = model._differential_effective_area_cm2(photons, events)

        assert np.all(result > 0)
        np.testing.assert_allclose(result[[0, 3]], result[[1, 2]], rtol=1e-5)

    def test_aeff_scales_diff_aeff_and_not_probability(self):
        eps = np.linspace(-0.4, 0.4, 7)
        photons, events = _one_photon_events(300., eps, n=len(eps))

        plain = IRFRelativeHistUnpolarized(_make_irf_hist())
        reference = plain._differential_effective_area_cm2(photons, events)

        aeff = _make_irf_hist().project('NuLambda', 'Ei')
        aeff *= 2.
        scaled = IRFRelativeHistUnpolarized(_make_irf_hist(), aeff=aeff)

        np.testing.assert_allclose(scaled._differential_effective_area_cm2(photons, events),
                                   2. * reference)
        np.testing.assert_allclose(scaled._event_probability(photons, events),
                                   plain._event_probability(photons, events))

    def test_aeff_equal_to_projection_changes_nothing(self):
        eps = np.linspace(-0.4, 0.4, 7)
        photons, events = _one_photon_events(300., eps, n=len(eps))

        plain = IRFRelativeHistUnpolarized(_make_irf_hist())
        same = IRFRelativeHistUnpolarized(_make_irf_hist(), aeff=_make_irf_hist().project('NuLambda', 'Ei'))

        np.testing.assert_allclose(same._differential_effective_area_cm2(photons, events),
                                   plain._differential_effective_area_cm2(photons, events))

    def test_aeff_varying_with_ei(self):
        """At an irf Ei center the ratio is exactly the factor of that Ei bin."""

        factors = np.array([1., 3., 0.5])
        aeff = _make_irf_hist().project('NuLambda', 'Ei')
        aeff *= factors[None, :]

        plain = IRFRelativeHistUnpolarized(_make_irf_hist())
        scaled = IRFRelativeHistUnpolarized(_make_irf_hist(), aeff=aeff)

        for ei, factor in zip(plain._event_prob.axes['Ei'].centers, factors):
            photons, events = _one_photon_events(ei, [0.1, -0.2], n=2)
            np.testing.assert_allclose(scaled._differential_effective_area_cm2(photons, events),
                                       factor * plain._differential_effective_area_cm2(photons, events))

    def test_zero_where_irf_total_is_zero(self):
        template = _make_irf_hist()
        irf = Histogram(template.axes, contents=np.zeros(template.axes.nbins), unit=u.cm * u.cm)
        photons, events = _one_photon_events(300., [0.1], n=1)
        model = IRFRelativeHistUnpolarized(irf, aeff=_make_aeff_hist())

        result = model._differential_effective_area_cm2(photons, events)

        assert np.all(np.isfinite(result)) and np.all(result == 0)

    def test_density_not_scaled_by_selections(self):
        """The cut scales the total area, not the density of events inside it."""

        eps = np.array([0.1, 0.2])
        photons, events = _one_photon_events(300., eps, n=2)  # Em = 330, 360 keV
        cut = EnergySelector(u.Quantity([320., 1000.], u.keV))

        for with_aeff in (False, True):
            def kwargs():
                return {'aeff': _make_irf_hist().project('NuLambda', 'Ei')} if with_aeff else {}

            no_cut = IRFRelativeHistUnpolarized(_make_irf_hist(), **kwargs())
            with_cut = IRFRelativeHistUnpolarized(_make_irf_hist(), selections=cut, **kwargs())

            np.testing.assert_allclose(with_cut._differential_effective_area_cm2(photons, events),
                                       no_cut._differential_effective_area_cm2(photons, events))
            assert with_cut._effective_area_cm2(photons)[0] < no_cut._effective_area_cm2(photons)[0]

    def test_integrates_to_total_effective_area(self):
        """Monte Carlo integral over Em (within the Epsilon range), Phi_kin
        and PsiChi of the density of one photon gives its effective area.
        Uses a Theta axis covering [-pi, pi] and no area in the unphysical
        region, so that the histogram content is fully reachable."""

        axes = Axes([
            HealpixAxis(nside=1, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
            Axis(np.geomspace(100, 1000, 4) * u.keV, label='Ei', scale='log'),
            Axis(np.linspace(-0.5, 0.5, 9), label='Epsilon'),
            Axis(np.linspace(0, 180, 13) * u.deg, label='Phi'),
            Axis(np.linspace(-180, 180, 25) * u.deg, label='Theta'),
            PolarizationAxis(np.linspace(0, 360, 13) * u.deg, convention=StereographicConvention(), label='Zeta'),
        ])
        phi = axes['Phi'].centers.to_value(u.rad)[:, None]
        theta = axes['Theta'].centers.to_value(u.rad)[None, :]
        physical = (phi + theta > 0) & (phi + theta < np.pi)
        contents = np.random.default_rng(0).random(axes.nbins) * physical[None, None, None, :, :, None]
        model = IRFRelativeHistUnpolarized(Histogram(axes, contents=contents, unit=u.cm * u.cm))

        n, energy = 400000, 300.
        rng = np.random.default_rng(1)
        eps = rng.uniform(-0.8, 0.8, n)  # wider than the axis, to also check the cutoff
        photons = _FakePhotonList(np.full(n, 0.3), np.full(n, 0.2), np.full(n, energy))
        events = _FakeEventData(rng.uniform(-np.pi, np.pi, n), np.arcsin(rng.uniform(-1, 1, n)),
                                rng.uniform(0, np.pi, n), energy * (1 + eps))

        volume = (1.6 * energy) * np.pi * 4 * np.pi
        integral = model._differential_effective_area_cm2(photons, events).mean() * volume
        total = model._effective_area_cm2(_FakePhotonList(np.array([0.3]), np.array([0.2]),
                                                          np.array([energy])))[0]

        assert integral == pytest.approx(total, rel=0.05)


def _make_physical_irf_hist(seed=0):
    """Random contents, only in the physical region of Phi + Theta, and
    zero in some (NuLambda, Ei) bins."""

    irf = _make_irf_hist(seed=seed)
    phi = irf.axes['Phi'].centers.to_value(u.rad)[:, None]
    theta = irf.axes['Theta'].centers.to_value(u.rad)[None, :]
    physical = (phi + theta > 0) & (phi + theta < np.pi)
    contents = irf.contents * physical[None, None, None, :, :, None]
    contents[0, 1] = 0
    return Histogram(irf.axes, contents=contents, unit=u.cm * u.cm)


def _phase_space(model):
    axes = model.axes
    phi, theta, zeta = np.meshgrid(axes['Phi'].edges, axes['Theta'].edges, axes['Zeta'].edges, indexing='ij')
    cds = RelativeCDSCoordinates.get_relative_cds_phase_space(
        phi[:-1, :-1, :-1], phi[1:, :-1, :-1], theta[:-1, :-1, :-1], theta[:-1, 1:, :-1],
        zeta[:-1, :-1, :-1], zeta[:-1, :-1, 1:])
    em = axes['Ei'].centers[:, None] * axes['Epsilon'].widths[None, :]
    return em[None, :, :, None, None, None] * cds[None, None, None, :, :, :]


class TestEventProbability:

    def test_interpolated_probability_integrates_to_one(self):
        """The probability as evaluated (multilinear interpolation between bin
        centers, flat up to the axis edges) integrates to 1 per (NuLambda, Ei)
        over Epsilon, Phi, Theta and Zeta. Splitting each bin at its center
        makes the interpolant multilinear in every sub-cell, where the
        midpoint rule is exact."""

        irf = _make_physical_irf_hist()
        model = IRFRelativeHistUnpolarized(irf)
        prob = model._event_prob
        labels = ['Epsilon', 'Phi', 'Theta', 'Zeta']

        sub_edges = [np.unique(np.concatenate([np.asarray(prob.axes[l].edges), np.asarray(prob.axes[l].centers)]))
                     for l in labels]
        mids = [0.5 * (e[:-1] + e[1:]) for e in sub_edges]
        widths = [np.diff(e) for e in sub_edges]
        bins = [np.searchsorted(np.asarray(prob.axes[l].edges), m, side='right') - 1 for l, m in zip(labels, mids)]
        mesh = np.meshgrid(*mids, indexing='ij')
        sub_volume = np.einsum('a,b,c,d->abcd', *widths)

        measure = _phase_space(model) / np.einsum('a,b,c,d->abcd', *[np.asarray(prob.axes[l].widths) for l in labels])

        cds_axes = Axes([prob.axes[l] for l in labels])
        nonzero = irf.contents.sum(axis=(2, 3, 4, 5)) > 0
        for pix in range(prob.contents.shape[0]):
            for e in range(prob.contents.shape[1]):
                values = Histogram(cds_axes, contents=prob.contents[pix, e]).interp(*(m.ravel() for m in mesh))
                weight = measure[0, e][np.ix_(*bins)]
                integral = np.sum(values.reshape(mesh[0].shape) * weight * sub_volume)
                assert integral == pytest.approx(1. if nonzero[pix, e] else 0., abs=1e-9)

    def test_differential_area_is_unselected_area_times_probability(self):
        photons, events = _make_photons_and_events(300)
        cut = EnergySelector(u.Quantity([150., 400.], u.keV))

        for kwargs in ({}, {'selections': cut}, {'aeff': _make_aeff_hist()},
                       {'aeff': _make_aeff_hist(), 'selections': cut}):
            model = IRFRelativeHistUnpolarized(_make_irf_hist(), **kwargs)
            diff = np.asarray(model._differential_effective_area_cm2(photons, events))
            prob = np.asarray(model._event_probability(photons, events))
            aeff_unselected = np.asarray(model._effective_area_cm2_unselected(photons))

            if 'selections' not in kwargs:
                np.testing.assert_allclose(aeff_unselected, model._effective_area_cm2(photons))
                np.testing.assert_allclose(prob, diff / aeff_unselected)
            else:
                selected = np.asarray(model._effective_area_cm2(photons))
                assert (selected > 0).any()
                np.testing.assert_allclose((selected * prob)[selected > 0], diff[selected > 0])
                assert np.all(prob[selected == 0] == 0)

    def test_selection_does_not_scale_diff_but_conditions_probability(self):
        eps = np.array([0.1, 0.2])
        photons, events = _one_photon_events(300., eps, n=2)  # Em = 330, 360 keV
        cut = EnergySelector(u.Quantity([320., 1000.], u.keV))

        no_cut = IRFRelativeHistUnpolarized(_make_irf_hist())
        with_cut = IRFRelativeHistUnpolarized(_make_irf_hist(), selections=cut)

        np.testing.assert_allclose(with_cut._differential_effective_area_cm2(photons, events),
                                   no_cut._differential_effective_area_cm2(photons, events))
        fraction = with_cut._effective_area_cm2(photons)[0] / no_cut._effective_area_cm2(photons)[0]
        assert 0 < fraction < 1
        np.testing.assert_allclose(with_cut._event_probability(photons, events),
                                   no_cut._event_probability(photons, events) / fraction)

    def test_zero_outside_epsilon_axis_range(self):
        model = IRFRelativeHistUnpolarized(_make_irf_hist())
        photons, events = _one_photon_events(300., [-3., -0.51, 0.51, 66.], n=4)

        np.testing.assert_array_equal(model._event_probability(photons, events), 0)

    def test_zero_outside_ei_axis_range(self):
        """The Ei axis of _make_irf_hist spans [100, 1000] keV."""

        for kwargs in ({}, {'aeff': _make_aeff_hist()},  # aeff spans [50, 2000] keV
                       {'selections': EnergySelector(u.Quantity([50., 5000.], u.keV))}):
            model = IRFRelativeHistUnpolarized(_make_irf_hist(), **kwargs)

            for energy in (99., 1001.):
                photons, events = _one_photon_events(energy, [0.1], n=1)

                assert model._event_probability(photons, events)[0] == 0
                assert model._differential_effective_area_cm2(photons, events)[0] == 0

            photons, events = _one_photon_events(500., [0.1], n=1)
            assert model._event_probability(photons, events)[0] > 0

        model = IRFRelativeHistUnpolarized(_make_irf_hist())
        photons, _ = _one_photon_events(99., [0.1])
        assert model._effective_area_cm2(photons)[0] == 0
        photons, _ = _one_photon_events(1001., [0.1])
        assert model._effective_area_cm2(photons)[0] == 0


class TestFromH5EiRange:

    @staticmethod
    def _write(path):
        _make_irf_hist().write(str(path), name="IRF", overwrite=True)
        _make_aeff_hist().write(str(path), name="AEFF", overwrite=True)

    def test_without_selections_loads_full_axes(self, tmp_path):
        path = tmp_path / "irf.h5"
        self._write(path)

        model = IRFRelativeHistUnpolarized.from_h5(path)

        np.testing.assert_allclose(model.axes['Ei'].edges, np.geomspace(100, 1000, 4))
        np.testing.assert_allclose(model._tot_aeff.axes['Ei'].edges, np.geomspace(50, 2000, 6))

    def test_selections_load_only_needed_ei_bins(self, tmp_path):
        """Ei edges 100, 215.4, 464.2, 1000; Epsilon in [-0.5, 0.5]. A cut at
        [400, 450] keV is reached from Ei in (400/1.5, 450/0.5) = (266.7, 900)."""

        path = tmp_path / "irf.h5"
        self._write(path)
        cut = EnergySelector(u.Quantity([400., 450.], u.keV))

        model = IRFRelativeHistUnpolarized.from_h5(path, selections=cut)

        np.testing.assert_allclose(model.axes['Ei'].edges, np.geomspace(100, 1000, 4)[1:])
        # aeff edges 50, 104.6, 218.7, 457.3, 956.4, 2000: those overlapping [215.4, 1000], plus the refinement near the cut
        np.testing.assert_allclose(model._tot_aeff.axes['Ei'].edges[[0, -1]], np.geomspace(50, 2000, 6)[[1, 5]])

    def test_high_cut_drops_low_ei_bins_and_matches_full_load(self, tmp_path):
        path = tmp_path / "irf.h5"
        self._write(path)

        cut = (EnergySelector(u.Quantity([500., 800.], u.keV)),)
        partial = IRFRelativeHistUnpolarized.from_h5(path, selections=cut)
        full = IRFRelativeHistUnpolarized(Histogram.open(path, "IRF"), aeff=Histogram.open(path, "AEFF"),
                                          selections=cut)

        assert partial.axes['Ei'].nbins < full.axes['Ei'].nbins

        # Inside the centers of the kept bins, where interpolation does not
        # use the dropped neighbors
        centers = partial.axes['Ei'].centers
        photons, events = _make_photons_and_events(400)
        energy = np.random.default_rng(5).uniform(centers[0], centers[-1], 400)
        photons = _FakePhotonList(photons.direction_lon_rad_sc, photons.direction_lat_rad_sc, energy)
        events.energy_keV = energy * (1 + np.random.default_rng(6).uniform(-0.4, 0.4, 400))

        for method in ('_differential_effective_area_cm2', '_event_probability'):
            np.testing.assert_allclose(getattr(partial, method)(photons, events),
                                       getattr(full, method)(photons, events), rtol=1e-10)
        np.testing.assert_allclose(partial._effective_area_cm2(photons), full._effective_area_cm2(photons),
                                   rtol=1e-10)

    def test_selection_beyond_all_bins_raises(self, tmp_path):
        path = tmp_path / "irf.h5"
        self._write(path)

        with pytest.raises(ValueError):
            IRFRelativeHistUnpolarized.from_h5(path, selections=EnergySelector(u.Quantity([1e5, 2e5], u.keV)))


class TestSelectionFractionBounded:
    """The selected fraction divides by the same piecewise-linear
    integral over the whole Epsilon axis, not by the plain sum of the
    contents. With non-uniform Epsilon bins (narrow central bin, wide
    neighbors, as in the real response files) the two differ a lot, and
    dividing by the plain sum gave fractions above 1."""

    eps_edges = np.array([-1., -0.5, -0.01, 0.01, 0.2])
    eps_centers = 0.5 * (eps_edges[:-1] + eps_edges[1:])
    eps_widths = np.diff(eps_edges)

    def _fraction(self, lo_keV, hi_keV, ei_keV=100.):
        content = np.array([[[0.1, 0.2, 1.0, 0.3]]])  # (npix=1, nEi=1, nEpsilon)
        return IRFRelativeHistUnpolarized._selection_fraction(
            content, np.array([ei_keV]), self.eps_centers, self.eps_widths, self.eps_edges,
            EnergySelector(u.Quantity([lo_keV, hi_keV], u.keV)))[0, 0]

    def test_partial_cut_fraction_at_most_one(self):
        density = np.array([0.1, 0.2, 1.0, 0.3]) / self.eps_widths
        piecewise_total = IRFRelativeHistUnpolarized._integrate_piecewise_linear(
            density, self.eps_centers, self.eps_edges[0], self.eps_edges[-1])
        assert piecewise_total > 2 * 1.6  # far from the plain sum of the contents (1.6)

        fraction = self._fraction(90., 110.)
        assert 0 < fraction <= 1

    def test_full_cut_fraction_is_one(self):
        assert self._fraction(0., 1e9) == pytest.approx(1.)

    def test_fraction_increases_with_the_cut(self):
        assert self._fraction(95., 105.) < self._fraction(90., 110.) < self._fraction(50., 120.) <= 1


def _direction(lon_rad, lat_rad):
    return UnitSphericalRepresentation(lon=Quantity(np.atleast_1d(lon_rad), 'rad'),
                                       lat=Quantity(np.atleast_1d(lat_rad), 'rad'))


def _nearest_bin_values(model, lon_rad, lat_rad, energy_keV):
    """Value of the (nearest-mode) event probability at the center of every
    (Epsilon, Phi, Theta, Zeta) bin, for one photon, and the bin volumes in
    the data space at that photon's own energy."""

    axes = model.axes
    eps, phi, theta, zeta = np.meshgrid(*[np.asarray(axes[l].centers) for l in ['Epsilon', 'Phi', 'Theta', 'Zeta']],
                                        indexing='ij')
    n = eps.size
    values = model._nearest_lookup(_direction(np.full(n, lon_rad), np.full(n, lat_rad)), np.full(n, energy_keV),
                                   eps.ravel(), phi.ravel(), theta.ravel(), zeta.ravel()).reshape(eps.shape)

    edges = np.meshgrid(axes['Phi'].edges, axes['Theta'].edges, axes['Zeta'].edges, indexing='ij')
    cds = RelativeCDSCoordinates.get_relative_cds_phase_space(
        edges[0][:-1, :-1, :-1], edges[0][1:, :-1, :-1], edges[1][:-1, :-1, :-1], edges[1][:-1, 1:, :-1],
        edges[2][:-1, :-1, :-1], edges[2][:-1, :-1, 1:])
    volume = energy_keV * np.asarray(axes['Epsilon'].widths)[:, None, None, None] * cds[None]

    return values, volume


class TestNearestInterpolation:

    photons_at = [(0.3, 0.2, 316.227766),  # Ei bin center, direction not at a pixel center
                  (-2.0, -0.7, 210.),  # Ei between centers
                  (2.5, 1.1, 640.)]

    def test_invalid_interpolation_raises(self):
        with pytest.raises(ValueError):
            IRFRelativeHistUnpolarized(_make_irf_hist(), interpolation='cubic')

    def test_default_is_linear(self):
        assert IRFRelativeHistUnpolarized(_make_irf_hist())._interpolation == 'linear'

    def test_linear_unchanged_by_nearest_option(self):
        photons, events = _make_photons_and_events(200)
        default = IRFRelativeHistUnpolarized(_make_irf_hist())
        linear = IRFRelativeHistUnpolarized(_make_irf_hist(), interpolation='linear')

        np.testing.assert_array_equal(default._event_probability(photons, events),
                                      linear._event_probability(photons, events))

    def test_probability_integrates_to_one(self):
        """The density is piecewise constant, so summing value x volume over
        the bins is the exact integral, for photons at any Ei and direction."""

        irf = _make_physical_irf_hist()
        model = IRFRelativeHistUnpolarized(irf, interpolation='nearest')

        for lon, lat, energy in self.photons_at:
            values, volume = _nearest_bin_values(model, lon, lat, energy)
            pix = model.axes['NuLambda'].find_bin(_direction(lon, lat))[0]
            iei = np.searchsorted(model.axes['Ei'].edges, energy, side='right') - 1
            assert irf.contents[pix, iei].sum() > 0
            assert np.sum(values * volume) == pytest.approx(1., rel=1e-12)

    def test_zero_probability_bins_have_zero_integral(self):
        """_make_physical_irf_hist has no content in NuLambda pixel 0, Ei bin 1."""

        model = IRFRelativeHistUnpolarized(_make_physical_irf_hist(), interpolation='nearest')
        lon, lat = model.axes['NuLambda'].pix2ang(0, lonlat=True)
        values, volume = _nearest_bin_values(model, np.deg2rad(lon), np.deg2rad(lat), 300.)

        assert np.all(values == 0)

    def test_value_is_that_of_the_bin_each_event_lands_in(self):
        model = IRFRelativeHistUnpolarized(_make_irf_hist(), interpolation='nearest')
        axes = model.axes
        lon, lat, energy = 0.3, 0.2, 210.
        pix = axes['NuLambda'].find_bin(_direction(lon, lat))[0]
        iei = np.searchsorted(axes['Ei'].edges, energy, side='right') - 1
        edges = [np.asarray(axes[l].edges) for l in ['Epsilon', 'Phi', 'Theta', 'Zeta']]
        centers = [np.asarray(axes[l].centers) for l in ['Epsilon', 'Phi', 'Theta', 'Zeta']]
        sizes = [len(c) for c in centers]

        for index in [(0, 0, 0, 0), (1, 2, 3, 4), tuple(s - 1 for s in sizes)]:
            args = [c[i] for c, i in zip(centers, index)]
            # Move within the bin, without crossing its edges
            shifted = [c[i] + 0.4 * (e[i + 1] - e[i]) for c, e, i in zip(centers, edges, index)]
            for point in (args, shifted):
                value = model._nearest_lookup(_direction(lon, lat), np.array([energy]), *[np.array([p]) for p in point])
                expected = (model._event_prob.contents[(pix, iei) + index]
                            * axes['Ei'].centers[iei] / energy)
                np.testing.assert_allclose(value, expected)

    def test_zero_outside_axes_ranges(self):
        model = IRFRelativeHistUnpolarized(_make_irf_hist(), interpolation='nearest')

        for energy, eps in [(99., 0.1), (1001., 0.1), (300., -0.51), (300., 0.51)]:
            photons, events = _one_photon_events(energy, [eps], n=1)
            assert model._event_probability(photons, events)[0] == 0

        photons, events = _one_photon_events(300., [0.1], n=1)
        assert model._event_probability(photons, events)[0] >= 0

    @pytest.mark.parametrize('use_aeff', [False, True])
    @pytest.mark.parametrize('ranges', [[[150., 400.]], [[160., 190.], [250., 330.]], [[0., 1e5]]])
    def test_selected_area_is_exact_integral_of_differential_area(self, ranges, use_aeff):
        selections = tuple(EnergySelector(u.Quantity(r, u.keV)) for r in ranges)
        kwargs = {'aeff': _make_aeff_hist()} if use_aeff else {}
        model = IRFRelativeHistUnpolarized(_make_physical_irf_hist(), selections=selections,
                                           interpolation='nearest', **kwargs)
        eps_edges = np.asarray(model.axes['Epsilon'].edges)

        for lon, lat, energy in self.photons_at:
            photons = _FakePhotonList(np.array([lon]), np.array([lat]), np.array([energy]))
            unselected = model._effective_area_cm2_unselected(photons)[0]
            values, volume = _nearest_bin_values(model, lon, lat, energy)
            per_eps = (values * volume).sum(axis=(1, 2, 3))

            # Integral of the differential effective area over Em in the ranges (lower
            # Em limit of Epsilon bin b to its upper, clipped to each range)
            em_edges = energy * (1 + eps_edges)
            selected = 0.
            for lo, hi in ranges:
                fraction = np.clip(np.minimum(hi, em_edges[1:]) - np.maximum(lo, em_edges[:-1]), 0, None) \
                           / np.diff(em_edges)
                selected += np.sum(per_eps * fraction)

            assert unselected > 0
            assert model._effective_area_cm2(photons)[0] == pytest.approx(unselected * selected, rel=1e-12)

    def test_conditional_probability_integrates_to_one_over_selected_range(self):
        """Dividing by the selected fraction makes the probability integrate to
        1 over the events inside the selection."""

        selections = (EnergySelector(u.Quantity([160., 190.], u.keV)), EnergySelector(u.Quantity([250., 330.], u.keV)))
        model = IRFRelativeHistUnpolarized(_make_physical_irf_hist(), selections=selections, interpolation='nearest')
        eps_edges = np.asarray(model.axes['Epsilon'].edges)

        lon, lat, energy = -2.0, -0.7, 210.
        photons = _FakePhotonList(np.array([lon]), np.array([lat]), np.array([energy]))
        values, volume = _nearest_bin_values(model, lon, lat, energy)
        per_eps = (values * volume).sum(axis=(1, 2, 3))
        em_edges = energy * (1 + eps_edges)
        inside = sum(np.clip(np.minimum(hi, em_edges[1:]) - np.maximum(lo, em_edges[:-1]), 0, None)
                     / np.diff(em_edges) for lo, hi in [(160., 190.), (250., 330.)])
        fraction = model._effective_area_cm2(photons)[0] / model._effective_area_cm2_unselected(photons)[0]

        assert 0 < fraction < 1
        assert np.sum(per_eps * inside) / fraction == pytest.approx(1., rel=1e-12)

    def test_from_h5_passes_interpolation(self, tmp_path):
        path = tmp_path / "irf.h5"
        _make_irf_hist().write(str(path), name="IRF", overwrite=True)

        assert IRFRelativeHistUnpolarized.from_h5(path, interpolation='nearest')._interpolation == 'nearest'
        assert IRFRelativeHistUnpolarized.from_h5(path)._interpolation == 'linear'

        with pytest.raises(ValueError):
            IRFRelativeHistUnpolarized.from_h5(path, interpolation='cubic')
