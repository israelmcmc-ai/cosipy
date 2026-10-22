from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import CartesianRepresentation, UnitSphericalRepresentation
from astropy.units import Quantity
from histpy import Axis, Axes, HealpixAxis, Histogram
from scoords import SpacecraftFrame

from cosipy.event_selection.distance_selection import DistanceSelector
from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.polarization import PolarizationAxis, StereographicConvention
from cosipy.response.relative_coordinates import RelativeCDSCoordinates
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized, _Hats


def _make_irf_hist(nside=2, seed=0, eps_edges=np.linspace(-0.5, 0.5, 5)):
    rng = np.random.default_rng(seed)

    axes = Axes([
        HealpixAxis(nside=nside, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis(np.geomspace(100, 1000, 4) * u.keV, label='Ei', scale='log'),
        Axis(eps_edges, label='Epsilon'),
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
    """A two-Ei-bin irf histogram with all contents equal to 1 cm^2
    and a uniform Epsilon axis, so the selection fraction for any cut
    has an exact, hand-computable closed form (constant density ->
    the piecewise-linear integration model is exact, not just an
    approximation)."""

    axes = Axes([
        HealpixAxis(nside=nside, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis([100., 300., 1000.] * u.keV, label='Ei'),
        Axis(np.linspace(*eps_range, n_eps + 1), label='Epsilon'),
        Axis(np.linspace(0, 180, 7) * u.deg, label='Phi'),
        Axis(np.linspace(-90, 90, 5) * u.deg, label='Theta'),
        PolarizationAxis(np.linspace(0, 360, 9) * u.deg, convention=StereographicConvention(), label='Zeta'),
    ])

    return Histogram(axes, contents=np.ones(axes.nbins), unit=u.cm * u.cm)


class TestEnergySelections:
    """selections=... on IRFRelativeHistUnpolarized scales the effective area
    of each photon by the fraction of its event probability inside the
    selection, and never touches the differential effective area."""

    @staticmethod
    def _photon(energy_keV, lon=0.3, lat=0.2):
        return _FakePhotonList(np.array([lon]), np.array([lat]), np.array([float(energy_keV)]))

    def test_non_energy_selector_raises_type_error(self):
        with pytest.raises(TypeError):
            IRFRelativeHistUnpolarized(_make_irf_hist(), selections=DistanceSelector())

    def test_selections_none_matches_omitting_the_parameter(self):
        photons, events = _make_photons_and_events(100)
        baseline = IRFRelativeHistUnpolarized(_make_irf_hist(seed=1))
        explicit_none = IRFRelativeHistUnpolarized(_make_irf_hist(seed=1), selections=None)

        np.testing.assert_array_equal(explicit_none._tot_aeff.contents, baseline._tot_aeff.contents)
        np.testing.assert_array_equal(explicit_none._effective_area_cm2(photons),
                                      baseline._effective_area_cm2(photons))

    @pytest.mark.parametrize('use_aeff', [False, True])
    def test_full_range_cut_gives_the_unselected_area(self, use_aeff):
        photons, _ = _make_photons_and_events(300)
        kwargs = {'aeff': _make_aeff_hist()} if use_aeff else {}
        model = IRFRelativeHistUnpolarized(_make_irf_hist(seed=2),
                                           selections=EnergySelector(u.Quantity([0., 1e9], u.keV)), **kwargs)
        unselected = model._effective_area_cm2_unselected(photons)

        assert (unselected > 0).any()
        np.testing.assert_allclose(model._effective_area_cm2(photons), unselected, rtol=1e-12)

    def test_uniform_binning_exact_fraction_bin_aligned_cut(self):
        """Epsilon edges are spaced by 0.2 over [-0.8, 0.8] and the contents are uniform
        in Epsilon. The density is flat between the outer centers and ramps to zero
        between them and the edges, so it integrates to 1.6 - 2 * 0.05 = 1.5 times its
        flat value. For a photon of 550 keV a [330, 550) keV cut is exactly eps in
        [-0.4, 0.0), in the flat part, so the fraction is 0.4 / 1.5."""

        model = IRFRelativeHistUnpolarized(
            _make_uniform_irf_hist(), copy=False,
            selections=EnergySelector(u.Quantity([330., 550.], u.keV)))
        photons = self._photon(550.)

        np.testing.assert_allclose(model._effective_area_cm2(photons),
                                   0.4 / 1.5 * model._effective_area_cm2_unselected(photons))

    def test_uniform_binning_exact_fraction_sub_bin_cut(self):
        """[385, 550) keV -> eps in [-0.3, 0.0): the boundary falls mid-bin."""

        model = IRFRelativeHistUnpolarized(
            _make_uniform_irf_hist(), copy=False,
            selections=EnergySelector(u.Quantity([385., 550.], u.keV)))
        photons = self._photon(550.)

        np.testing.assert_allclose(model._effective_area_cm2(photons),
                                   0.3 / 1.5 * model._effective_area_cm2_unselected(photons))

    def test_tuple_of_selectors_combines_via_union(self):
        """A tuple of EnergySelectors passed to selections= is OR'd
        together (each entry is an acceptable window), matching
        EnergySelector.union -- not intersected/AND'd."""

        a = EnergySelector(u.Quantity([100., 500.], u.keV))
        b = EnergySelector(u.Quantity([400., 900.], u.keV))  # overlaps a
        photons, _ = _make_photons_and_events(100)

        def area(selections):
            return IRFRelativeHistUnpolarized(_make_irf_hist(seed=3), selections=selections)._effective_area_cm2(photons)

        via_tuple, via_union, via_intersect = area((a, b)), area(a.union(b)), area(a.intersect(b))

        np.testing.assert_allclose(via_tuple, via_union)
        assert not np.allclose(via_tuple, via_intersect)

    def test_multi_range_selector_sums_disjoint_windows(self):
        """A single EnergySelector with two disjoint ranges gives the same area as
        summing each range's own single-range cut (no double-counting)."""

        range_a = EnergySelector(u.Quantity([330., 470.], u.keV))
        range_b = EnergySelector(u.Quantity([600., 750.], u.keV))
        photons, _ = _make_photons_and_events(100)

        def area(selections):
            return IRFRelativeHistUnpolarized(_make_uniform_irf_hist(), copy=False, selections=selections)._effective_area_cm2(photons)

        assert (area(range_a) > 0).any()
        np.testing.assert_allclose(area(range_a.union(range_b)), area(range_a) + area(range_b))

    def test_aeff_and_selections_together(self):
        photons, _ = _make_photons_and_events(200)
        no_cut = IRFRelativeHistUnpolarized(_make_irf_hist(seed=4), aeff=_make_aeff_hist(seed=5))
        cut = IRFRelativeHistUnpolarized(
            _make_irf_hist(seed=4), aeff=_make_aeff_hist(seed=5),
            selections=EnergySelector(u.Quantity([150., 400.], u.keV)))

        unselected = no_cut._effective_area_cm2(photons)

        np.testing.assert_array_equal(cut._effective_area_cm2_unselected(photons), unselected)
        assert np.all(cut._effective_area_cm2(photons) <= unselected * (1 + 1e-12))
        assert cut._effective_area_cm2(photons).sum() < unselected.sum()

    def test_diff_aeff_unaffected_by_selections(self):
        no_cut = IRFRelativeHistUnpolarized(_make_irf_hist(seed=6))
        cut = IRFRelativeHistUnpolarized(
            _make_irf_hist(seed=6),
            selections=EnergySelector(u.Quantity([150., 400.], u.keV)))

        np.testing.assert_array_equal(cut._event_prob.contents, no_cut._event_prob.contents)


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

    def test_ramps_to_zero_at_the_epsilon_edges(self):
        """Between the outer Epsilon centers (-0.375, 0.375) and the edges (-0.5, 0.5),
        the density ramps linearly to zero (histpy fill_value=0)."""

        model = IRFRelativeHistUnpolarized(_make_irf_hist())
        photons, events = _one_photon_events(300., [-0.5, -0.4375, -0.375, 0.375, 0.4375, 0.5], n=6)

        result = model._differential_effective_area_cm2(photons, events)

        assert np.all(result[[2, 3]] > 0)
        np.testing.assert_allclose(result[[0, 5]], 0, atol=1e-12 * result.max())
        np.testing.assert_allclose(result[[1, 4]], 0.5 * result[[2, 3]], rtol=1e-10)

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


def _make_physical_irf_hist(seed=0, **kwargs):
    """Random contents, only in the physical region of Phi + Theta, and
    zero in some (NuLambda, Ei) bins."""

    irf = _make_irf_hist(seed=seed, **kwargs)
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


def _direction(lon_rad, lat_rad):
    return UnitSphericalRepresentation(lon=Quantity(np.atleast_1d(lon_rad), 'rad'),
                                       lat=Quantity(np.atleast_1d(lat_rad), 'rad'))


# Photons off the Ei and NuLambda centers
PHOTONS_AT = [(0.3, 0.2, 316.227766),  # Ei bin center, direction not at a pixel center
              (-2.0, -0.7, 210.),  # Ei between centers
              (2.5, 1.1, 640.)]


class TestEventProbability:

    @staticmethod
    def _sub_cells(model, energy, extra_eps_edges=(), nsub=4):
        """Sub-cells of the Epsilon, Phi, Theta and Zeta axes: Epsilon and Zeta are
        split at the edges and centers (so the interpolant is multilinear in each
        sub-cell), and each Phi and Theta bin into ``nsub`` sub-bins and at the centers.

        Returns the midpoints of each axis and the weight of each sub-cell, which
        integrates over the true measure of Em (``dEm = energy * dEpsilon``), Phi,
        Theta and Zeta: ``sin(phi + theta) dphi dtheta dzeta``.
        """

        prob = model._event_prob

        def split(label, n=1, extra=()):
            axis = prob.axes[label]
            edges = np.asarray(axis.edges)
            sub = [np.linspace(lo, hi, n + 1) for lo, hi in zip(edges[:-1], edges[1:])]
            e = np.unique(np.concatenate(sub + [np.asarray(axis.centers), np.asarray(extra, dtype=float)]))
            return e[(e >= edges[0]) & (e <= edges[-1])]

        eps_edges, phi_edges, theta_edges, zeta_edges = (split('Epsilon', extra=extra_eps_edges), split('Phi', nsub),
                                                         split('Theta', nsub), split('Zeta'))
        mids = [0.5 * (e[:-1] + e[1:]) for e in (eps_edges, phi_edges, theta_edges, zeta_edges)]

        phi_lo, theta_lo = np.meshgrid(phi_edges[:-1], theta_edges[:-1], indexing='ij')
        phi_hi, theta_hi = np.meshgrid(phi_edges[1:], theta_edges[1:], indexing='ij')
        phase_space = RelativeCDSCoordinates.get_relative_cds_phase_space(
            phi_lo, phi_hi, theta_lo, theta_hi, np.zeros_like(phi_lo), np.ones_like(phi_lo))
        phase_space = np.nan_to_num(np.asarray(phase_space, dtype=float))

        weight = (energy * np.diff(eps_edges)[:, None, None, None] * phase_space[None, :, :, None]
                  * np.diff(zeta_edges)[None, None, None, :])

        return mids, weight

    @staticmethod
    def _density_on_sub_cells(model, lon, lat, energy, extra_eps_edges=(), nsub=4):
        """The probability density of one photon, per unit Em as evaluated (the
        interpolated table divided by the photon energy), at the midpoint of the
        sub-cells of :meth:`_sub_cells`. The midpoint rule converges as ``1 / nsub^2``,
        and the error is ~1e-3 for the default.

        Returns the density, the Epsilon midpoints, and the weight of each sub-cell.
        """

        mids, weight = TestEventProbability._sub_cells(model, energy, extra_eps_edges, nsub)
        mesh = np.meshgrid(*mids, indexing='ij')
        n = mesh[0].size

        photon_dir = _direction(np.full(n, lon), np.full(n, lat))
        density = model._event_prob.interp(photon_dir, np.full(n, energy),
                                           *(m.ravel() for m in mesh),
                                           fill_value=model._FILL_VALUE, extrapolate=True).reshape(mesh[0].shape)
        density = density / energy

        return density, mids[0], weight

    @pytest.mark.parametrize('eps_edges', [np.linspace(-0.5, 0.5, 5), np.array([-0.5, -0.3, -0.02, 0.02, 0.1, 0.5])])
    def test_interpolated_probability_integrates_to_one(self, eps_edges):
        """The probability as evaluated (multilinear interpolation, flat up to the
        axis edges) integrates to 1 over Em, Phi, Theta and Zeta for a photon at any
        Ei and direction: at Ei and NuLambda centers, and in between them. The
        measure is the true one, sin(phi + theta) dphi dtheta dzeta."""

        irf = _make_physical_irf_hist(eps_edges=eps_edges)
        model = IRFRelativeHistUnpolarized(irf)
        pixel_center = model.axes['NuLambda'].pix2ang(5, lonlat=True)
        photons_at = [(np.deg2rad(pixel_center[0]), np.deg2rad(pixel_center[1]), model.axes['Ei'].centers[1])] \
                     + PHOTONS_AT

        for lon, lat, energy in photons_at:
            density, _, weight = self._density_on_sub_cells(model, lon, lat, energy)
            pix, _ = model.axes['NuLambda'].interp_weights(_direction(lon, lat))
            assert (irf.contents[np.asarray(pix).ravel()].sum(axis=(2, 3, 4, 5)) > 0).all()
            assert np.sum(density * weight) == pytest.approx(1., rel=3e-3)

    def test_event_probability_integrates_to_one_over_the_data_space(self):
        """End to end: events are built from a grid of sub-cells in the relative
        coordinates (the scattered direction from ``to_cds``), evaluated with
        ``_interp_event_prob``, and summed with the exact volume of each sub-cell
        in the data space. It covers the physical region, Phi + Theta in [0, pi]."""

        model = IRFRelativeHistUnpolarized(_make_physical_irf_hist())
        pixel_center = model.axes['NuLambda'].pix2ang(5, lonlat=True)
        photons_at = [(np.deg2rad(pixel_center[0]), np.deg2rad(pixel_center[1]), model.axes['Ei'].centers[1]),
                      PHOTONS_AT[1]]

        for lon, lat, energy in photons_at:
            mids, weight = self._sub_cells(model, energy, nsub=8)
            eps, phi_kin, theta, zeta = np.meshgrid(*mids, indexing='ij')
            phi_geo = phi_kin + theta
            physical = (weight > 0) & (phi_geo > 0) & (phi_geo < np.pi)

            n = physical.sum()
            relcoords = RelativeCDSCoordinates(_direction(np.full(1, lon), np.full(1, lat)).to_cartesian().xyz,
                                               model._pol_convention)
            psichi = RelativeCDSCoordinates.to_cds(relcoords, phi_geo[physical], zeta[physical])
            psichi = UnitSphericalRepresentation.from_cartesian(
                CartesianRepresentation(*psichi, unit=u.one))

            photons = _FakePhotonList(np.full(n, lon), np.full(n, lat), np.full(n, energy))
            events = _FakeEventData(psichi.lon.to_value(u.rad), psichi.lat.to_value(u.rad), phi_kin[physical],
                                    energy * (1 + eps[physical]))

            prob = model._interp_event_prob(photons, events)

            assert np.sum(prob * weight[physical]) == pytest.approx(1., rel=2e-3)

    @pytest.mark.parametrize('ranges', [[[150., 400.]], [[160., 190.], [250., 330.]], [[300., 340.]],
                                        [[310., 330.]], [[300., 900.]]])
    @pytest.mark.parametrize('use_aeff', [False, True])
    def test_selected_area_is_exact_integral_of_differential_area(self, ranges, use_aeff):
        """For photons off the Ei and NuLambda centers, the selected area is the
        unselected area times the exact integral of the probability over the
        selected Em, splitting the Epsilon integration at the cut boundaries, with the
        true measure."""

        selections = tuple(EnergySelector(u.Quantity(r, u.keV)) for r in ranges)
        kwargs = {'aeff': _make_aeff_hist()} if use_aeff else {}
        model = IRFRelativeHistUnpolarized(_make_physical_irf_hist(), selections=selections, **kwargs)

        for lon, lat, energy in PHOTONS_AT:
            cuts = [x / energy - 1 for r in ranges for x in r]
            density, eps, weight = self._density_on_sub_cells(model, lon, lat, energy, cuts)
            per_eps = (density * weight).sum(axis=(1, 2, 3))
            inside = np.zeros(len(eps), bool)
            for lo, hi in ranges:
                inside |= (energy * (1 + eps) > lo) & (energy * (1 + eps) < hi)

            photons = _FakePhotonList(np.array([lon]), np.array([lat]), np.array([energy]))
            unselected = model._effective_area_cm2_unselected(photons)[0]

            assert unselected > 0
            assert 0 < per_eps[inside].sum() < 1
            assert model._effective_area_cm2(photons)[0] == pytest.approx(unselected * per_eps[inside].sum(), rel=3e-3)

    @pytest.mark.parametrize('eps_edges', [np.linspace(-0.5, 0.5, 5), np.array([-0.5, -0.3, -0.02, 0.02, 0.1, 0.5])])
    def test_full_range_selection_fraction_is_one(self, eps_edges):
        photons, _ = _make_photons_and_events(500)
        model = IRFRelativeHistUnpolarized(_make_irf_hist(eps_edges=eps_edges),
                                           selections=EnergySelector(u.Quantity([0., 1e9], u.keV)))

        np.testing.assert_allclose(model._selected_fraction(photons), 1., rtol=1e-12)

    def test_event_probability_is_per_unit_epsilon_over_energy(self):
        """Events built from known relative coordinates: the probability is the stored
        table, interpolated at the photon, divided by its energy."""

        model = IRFRelativeHistUnpolarized(_make_irf_hist())
        lon, lat, energy, eps, phi, theta, zeta = 0.3, 0.2, 210., 0.1, 1.5, 0.1, 2.0
        photon_dir = _direction(lon, lat)
        x, y, z = np.asarray(RelativeCDSCoordinates(photon_dir.to_cartesian().xyz, pol_convention=model._pol_convention
                                                    ).to_cds(phi + theta, zeta)).reshape(3)
        photons = _FakePhotonList(np.array([lon]), np.array([lat]), np.array([energy]))
        events = _FakeEventData(np.array([np.arctan2(y, x)]), np.array([np.arcsin(z)]), np.array([phi]), np.array([energy * (1 + eps)]))

        expected = model._event_prob.interp(photon_dir, np.array([energy]), np.array([eps]), np.array([phi]),
                                            np.array([theta]), np.array([zeta])) / energy

        assert expected[0] > 0
        np.testing.assert_allclose(model._event_probability(photons, events), expected)

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


class TestHats:
    """_Hats reproduces the weights of histpy's interp() for each edge treatment,
    and integrates them exactly."""

    axis = Axis([-0.5, -0.3, -0.02, 0.02, 0.1, 0.5], label='Epsilon')

    @staticmethod
    def _histpy_hats(axis, x, edges):
        kwargs = {'clamp': {},
                  'zero': {'extend_to_edges': True, 'extrapolate': True},
                  'periodic': {'periodic': True}}[edges]
        bins, weights = axis.interp_weights(x, **kwargs)
        hats = np.zeros((len(x), axis.nbins))
        for b, w in zip(bins, weights):
            valid = (b >= 0) & (b < axis.nbins)  # virtual edge bins hold the fill value 0
            np.add.at(hats, (np.flatnonzero(valid), b[valid]), w[valid])
        return hats

    @pytest.mark.parametrize('edges', ['clamp', 'zero', 'periodic'])
    def test_values_match_histpy_interp_weights(self, edges):
        x = np.concatenate((np.linspace(-0.5, 0.5, 1001), self.axis.centers))

        np.testing.assert_allclose(_Hats(self.axis, edges)(x), self._histpy_hats(self.axis, x, edges), atol=1e-12)

    @pytest.mark.parametrize('edges', ['clamp', 'zero', 'periodic'])
    def test_cumulative_and_integrals(self, edges):
        hats = _Hats(self.axis, edges)
        x = np.linspace(-0.5, 0.5, 200001)
        numerical = np.concatenate((np.zeros((1, self.axis.nbins)),
                                    np.cumsum(0.5 * (hats(x)[1:] + hats(x)[:-1]) * np.diff(x)[:, None], axis=0)))

        np.testing.assert_allclose(hats.cumulative(x), numerical, atol=1e-9)
        np.testing.assert_allclose(hats.integrals(), numerical[-1], atol=1e-9)
        np.testing.assert_allclose(hats.cumulative([-3., 7.]), [np.zeros(self.axis.nbins), hats.integrals()])

    def test_periodic_integrals_and_partition_of_unity(self):
        """Periodic hats add up to one everywhere, so they integrate to the axis
        range, and for uniform bins each one to the bin width."""

        hats = _Hats(self.axis, 'periodic')
        np.testing.assert_allclose(hats(np.linspace(-0.5, 0.5, 101)).sum(axis=1), 1)
        assert hats.integrals().sum() == pytest.approx(1.)

        uniform = Axis(np.linspace(0, 2 * np.pi, 9), label='Zeta')
        np.testing.assert_allclose(_Hats(uniform, 'periodic').integrals(), np.pi / 4)

    def test_zero_ramps_to_zero_at_the_edges(self):
        hats = _Hats(self.axis, 'zero')
        np.testing.assert_array_equal(hats([-0.5, 0.5]), 0)
        np.testing.assert_allclose(hats([-0.45])[0, 0], 0.5)
        # first hat: half bin ramp + half the distance to the next center
        assert hats.integrals()[0] == pytest.approx(0.25 * 0.2 + 0.5 * (-0.16 + 0.4))


class TestZetaPeriodic:
    """Zeta is interpolated periodically, so an event's Zeta can be anywhere
    (to_relative() returns (-pi, pi]) and the density is continuous across 0."""

    @staticmethod
    def _events(model, zeta, lon=0.3, lat=0.2, energy=300., eps=0.1, phi=1.2, theta=0.15):
        photon_dir = _direction(lon, lat)
        relcoords = RelativeCDSCoordinates(photon_dir.to_cartesian().xyz, pol_convention=model._pol_convention)
        zeta = np.asarray(zeta, dtype=float)
        n = len(zeta)
        x, y, z = (np.asarray(c).ravel() for c in relcoords.to_cds(np.full(n, phi + theta), zeta))
        photons = _FakePhotonList(np.full(n, lon), np.full(n, lat), np.full(n, energy))
        events = _FakeEventData(np.arctan2(y, x), np.arcsin(np.clip(z, -1, 1)), np.full(n, phi),
                                np.full(n, energy * (1 + eps)))
        return photons, events

    def test_matches_periodic_interpolation_of_the_table(self):
        model = IRFRelativeHistUnpolarized(_make_irf_hist())
        zeta = np.array([0.01, 0.2, 3.0, 6.1, 2 * np.pi - 0.01])
        photons, events = self._events(model, zeta)

        expected = model._event_prob.interp(_direction(np.full(5, 0.3), np.full(5, 0.2)), np.full(5, 300.),
                                            np.full(5, 0.1), np.full(5, 1.2), np.full(5, 0.15), zeta,
                                            fill_value=model._FILL_VALUE, extrapolate=True) / 300.

        np.testing.assert_allclose(model._interp_event_prob(photons, events), expected, rtol=1e-6)

    def test_continuous_across_zero(self):
        model = IRFRelativeHistUnpolarized(_make_irf_hist())
        photons, events = self._events(model, [-1e-6, 1e-6, 2 * np.pi - 1e-6])

        prob = model._interp_event_prob(photons, events)

        assert prob[0] > 0
        np.testing.assert_allclose(prob, prob[0], rtol=1e-4)

    def test_between_last_and_first_center_is_not_clamped(self):
        """At Zeta = 0 the density is the average of the first and last bins
        (uniform bins), not the value of the first one."""

        model = IRFRelativeHistUnpolarized(_make_irf_hist())
        centers = model.axes['Zeta'].centers
        photons, events = self._events(model, [0., centers[0], centers[-1]])

        prob = model._interp_event_prob(photons, events)

        assert prob[1] != pytest.approx(prob[2])
        assert prob[0] == pytest.approx(0.5 * (prob[1] + prob[2]), rel=1e-4)
