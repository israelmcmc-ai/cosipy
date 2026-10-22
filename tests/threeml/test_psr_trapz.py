from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astromodels import PointSource, Gaussian, DiracDelta, Powerlaw
from histpy import Axis, Axes, HealpixAxis, Histogram
from scoords import SpacecraftFrame

from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.polarization import PolarizationAxis, StereographicConvention
from cosipy.response.broadened_irf_cache import BroadenedIRFCache
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.threeml.psr_fixed_ei import UnbinnedThreeMLPointSourceResponseTrapz

LIVETIME = 10.  # s
LIVETIME_RATIO = 0.8
LON, LAT = 0.3, 0.6  # source direction in the spacecraft frame, rad


def make_irf_hist(seed=0):
    """Random contents in the physical region of Phi + Theta, a photopeak in Epsilon,
    over 50-2000 keV."""

    rng = np.random.default_rng(seed)
    axes = Axes([
        HealpixAxis(nside=1, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis(np.geomspace(50, 2000, 12) * u.keV, label='Ei', scale='log'),
        Axis([-1, -0.5, -0.2, -0.05, -0.01, 0.01, 0.05, 0.2], label='Epsilon'),
        Axis(np.linspace(0, 180, 7) * u.deg, label='Phi'),
        Axis(np.linspace(-90, 90, 5) * u.deg, label='Theta'),
        PolarizationAxis(np.linspace(0, 360, 9) * u.deg, convention=StereographicConvention(), label='Zeta'),
    ])
    phi = axes['Phi'].centers.to_value(u.rad)[:, None]
    theta = axes['Theta'].centers.to_value(u.rad)[None, :]
    physical = (phi + theta > 0) & (phi + theta < np.pi)
    contents = rng.random(axes.nbins) * physical[None, None, None, :, :, None]
    contents[:, :, 4] *= 20  # photopeak

    return Histogram(axes, contents=contents, unit=u.cm * u.cm)


@pytest.fixture(scope='module')
def irf_file(tmp_path_factory):
    path = tmp_path_factory.mktemp('irf') / 'irf.h5'
    make_irf_hist().write(path, name='IRF')
    return path


@pytest.fixture(scope='module')
def cache_file(tmp_path_factory):
    return tmp_path_factory.mktemp('cache') / 'cache.h5'


def make_cache(irf_file, cache_file):
    """From the IRF without selections, which are applied when the kernels are evaluated."""
    return BroadenedIRFCache(IRFRelativeHistUnpolarized.from_h5(irf_file), cache_file,
                             energy_range_keV=(100., 1000.), max_level=4)


@pytest.fixture
def energy_m():
    return np.array([120., 250., 400., 505., 511., 700., 950.])


@pytest.fixture
def data(energy_m):
    n = energy_m.size
    rng = np.random.default_rng(3)
    data = MagicMock(spec=TimeTagEmCDSEventDataInSCFrameInterface)
    data.nevents = n
    data.time.utc.unix = 1000. + np.arange(n) / n
    data.energy_keV = energy_m
    data.scattering_angle_rad = rng.uniform(0.5, 1.5, n)
    data.scattered_lon_rad_sc = rng.uniform(-np.pi, np.pi, n)
    data.scattered_lat_rad_sc = rng.uniform(-1, 1, n)
    return data


@pytest.fixture
def sc_history():
    sc = MagicMock(spec=SpacecraftHistory)
    sc.obstime = Time([1000., 1000. + LIVETIME / LIVETIME_RATIO], format='unix')
    sc.nintervals = 1
    sc.livetime = [LIVETIME] * u.s
    sc.intervals_duration = [LIVETIME / LIVETIME_RATIO] * u.s

    def interp(times):
        interp_sc = MagicMock()
        interp_sc.get_target_in_sc_frame.return_value = SkyCoord(lon=np.full(times.size, LON),
                                                                 lat=np.full(times.size, LAT),
                                                                 unit='rad', frame=SpacecraftFrame())
        interp_sc.get_earth_occ.return_value = np.zeros(times.size, dtype=bool)
        return interp_sc

    sc.interp.side_effect = interp
    return sc


def brute_force(irf, spectrum, data, emin=100., emax=1000., npoints=4001):
    """Expected counts and density per event, integrating over the photon energy with the
    original IRF on a fine grid."""

    energy = np.geomspace(emin, emax, npoints)
    de = np.gradient(energy)
    flux = spectrum(energy)
    n = data.nevents

    aeff = np.asarray(irf.effective_area_cm2(
        PhotonListWithDirectionAndEnergyInSCFrame(np.full(npoints, LON), np.full(npoints, LAT), energy)))
    counts = LIVETIME * np.sum(aeff * flux * de)

    events = EmCDSEventDataInSCFrameFromArrays(*(np.repeat(np.asarray(a), npoints)
                                                 for a in (data.energy_keV, data.scattered_lon_rad_sc,
                                                           data.scattered_lat_rad_sc, data.scattering_angle_rad)))
    photons = PhotonListWithDirectionAndEnergyInSCFrame(np.full(n * npoints, LON), np.full(n * npoints, LAT),
                                                        np.tile(energy, n))
    diff = np.asarray(irf.differential_effective_area_cm2(photons, events)).reshape(n, npoints)
    density = LIVETIME_RATIO * diff @ (flux * de)

    return counts, density


@pytest.mark.parametrize('index', [-2., -2.6])
@pytest.mark.parametrize('selection', [None, [200., 900.]])
def test_matches_brute_force(irf_file, cache_file, data, sc_history, index, selection):
    selections = None if selection is None else EnergySelector(selection * u.keV)
    irf = IRFRelativeHistUnpolarized.from_h5(irf_file, selections=selections)
    cache = make_cache(irf_file, cache_file)
    spectrum = Powerlaw(K=1e-2, index=index, piv=100)
    source = PointSource('src', l=0, b=0, spectral_shape=spectrum)

    cells = UnbinnedThreeMLPointSourceResponseTrapz.cells_for(source, cache, accuracy=1e-3)
    psr = UnbinnedThreeMLPointSourceResponseTrapz(data, irf, sc_history, cache, cells, offset=None)
    psr.set_source(source)

    counts, density = brute_force(irf, spectrum, data)

    # Events outside the selection would not be in the data
    inside = np.ones(data.nevents, dtype=bool) if selection is None else \
        (data.energy_keV >= selection[0]) & (data.energy_keV < selection[1])

    assert psr.expected_counts() == pytest.approx(counts, rel=5e-3)
    np.testing.assert_allclose(psr.expectation_density()[inside], density[inside], rtol=0.01)


def test_weight_like_the_spectrum_is_exact_for_the_counts(irf_file, cache_file, data, sc_history):
    """With F = w (g constant), the interpolation of g is exact, and with a single
    level-0 cell the expected counts are the integral of w Aeff."""

    irf = IRFRelativeHistUnpolarized.from_h5(irf_file)
    cache = make_cache(irf_file, cache_file)
    source = PointSource('src', l=0, b=0, spectral_shape=Powerlaw(K=1e-2, index=-2, piv=1))

    psr = UnbinnedThreeMLPointSourceResponseTrapz(data, irf, sc_history, cache, [(0, 0)], offset=None)
    psr.set_source(source)

    counts, _ = brute_force(irf, source.spectrum.main.shape, data, npoints=100_001)
    assert psr.expected_counts() == pytest.approx(counts, rel=1e-4)


def test_nodes_get_the_pieces_of_their_cells(irf_file, cache_file, data, sc_history):
    """The lower node of a cell gets its L piece, the upper node its R piece."""

    irf = IRFRelativeHistUnpolarized.from_h5(irf_file)
    cache = make_cache(irf_file, cache_file)
    cells = [(1, 0), (2, 2), (2, 3)]
    psr = UnbinnedThreeMLPointSourceResponseTrapz(data, irf, sc_history, cache, cells, offset=None)
    psr.set_source(PointSource('src', l=0, b=0, spectral_shape=Powerlaw()))
    psr.init_cache()

    nodes = [100., 10 ** 2.5, 10 ** 2.75, 1000.]
    np.testing.assert_allclose(psr._nodes_keV, nodes)

    photons = PhotonListWithDirectionAndEnergyInSCFrame(np.array([LON]), np.array([LAT]), np.array([1.]))
    expected = np.zeros(4)
    for n, cell in enumerate(cells):
        irf_r, irf_l = cache.piece_irfs(*cell)
        photons.energy_keV[:] = nodes[n]
        expected[n] += LIVETIME * irf_l.effective_area_cm2(photons)[0]
        expected[n + 1] += LIVETIME * irf_r.effective_area_cm2(photons)[0]

    np.testing.assert_allclose(psr._exposure, expected)


def test_lines_with_the_original_irf(irf_file, cache_file, data, sc_history):
    irf = IRFRelativeHistUnpolarized.from_h5(irf_file)
    cache = make_cache(irf_file, cache_file)
    spectrum = DiracDelta(value=1e-3, zero_point=511.)
    spectrum.zero_point.fix = True
    source = PointSource('src', l=0, b=0, spectral_shape=spectrum)

    mono_nodes = UnbinnedThreeMLPointSourceResponseTrapz.mono_nodes_of(source)
    np.testing.assert_allclose(mono_nodes.to_value(u.keV), [511.])

    psr = UnbinnedThreeMLPointSourceResponseTrapz(data, irf, sc_history, cache, [(0, 0)],
                                                  mono_nodes=mono_nodes, offset=None)
    psr.set_source(source)

    photons = PhotonListWithDirectionAndEnergyInSCFrame(np.full(data.nevents, LON), np.full(data.nevents, LAT),
                                                        np.full(data.nevents, 511.))
    events = EmCDSEventDataInSCFrameFromArrays(np.asarray(data.energy_keV), np.asarray(data.scattered_lon_rad_sc),
                                               np.asarray(data.scattered_lat_rad_sc),
                                               np.asarray(data.scattering_angle_rad))
    diff = np.asarray(irf.differential_effective_area_cm2(photons, events))

    assert psr.expected_counts() == pytest.approx(LIVETIME * 1e-3 * irf.effective_area_cm2(photons[0]))
    np.testing.assert_allclose(psr.expectation_density(), LIVETIME_RATIO * 1e-3 * diff)


def test_free_line_energy_raises():
    spectrum = DiracDelta(value=1e-3, zero_point=511.)
    spectrum.zero_point.free = True

    with pytest.raises(ValueError):
        UnbinnedThreeMLPointSourceResponseTrapz.mono_nodes_of(spectrum)


def test_cells_for(irf_file, cache_file):
    irf = IRFRelativeHistUnpolarized.from_h5(irf_file)
    cache = make_cache(irf_file, cache_file)

    # g is constant: no refinement
    assert UnbinnedThreeMLPointSourceResponseTrapz.cells_for(Powerlaw(index=-2), cache) == [(0, 0)]

    # Steeper: refined uniformly, more for a smaller target
    steep = Powerlaw(index=-2.5)
    coarse = UnbinnedThreeMLPointSourceResponseTrapz.cells_for(steep, cache, accuracy=1e-2)
    fine = UnbinnedThreeMLPointSourceResponseTrapz.cells_for(steep, cache, accuracy=1e-3)
    assert 1 < len(coarse) < len(fine)

    # A narrow feature is refined around it, down to the finest level
    line = Gaussian(F=1e-3, mu=400., sigma=10.) + Powerlaw(K=1e-4, index=-2, piv=100)
    cells = UnbinnedThreeMLPointSourceResponseTrapz.cells_for(line, cache, accuracy=1e-2)
    finest = [cache.cell_edges_keV(*c) for c in cells if c[0] == cache.max_level]
    assert finest and all(300 < lo < 500 for lo, _ in finest)
    assert any(level < cache.max_level for level, _ in cells)


def test_cells_must_tile_the_range(irf_file, cache_file, data, sc_history):
    irf = IRFRelativeHistUnpolarized.from_h5(irf_file)
    cache = make_cache(irf_file, cache_file)

    with pytest.raises(ValueError):
        UnbinnedThreeMLPointSourceResponseTrapz(data, irf, sc_history, cache, [(1, 0)])
    with pytest.raises(ValueError):
        UnbinnedThreeMLPointSourceResponseTrapz(data, irf, sc_history, cache, [(1, 0), (1, 1), (2, 3)])
