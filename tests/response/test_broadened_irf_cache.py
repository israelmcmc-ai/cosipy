import numpy as np
import pytest

import astropy.units as u
from histpy import Axis, Axes, HealpixAxis, Histogram
from scoords import SpacecraftFrame

from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.polarization import PolarizationAxis, StereographicConvention
from cosipy.response.broadened_irf_cache import BroadenedIRFCache
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized


def make_irf_hist(seed=0):
    """Random contents in the physical region of Phi + Theta, over 100-1000 keV."""

    rng = np.random.default_rng(seed)
    axes = Axes([
        HealpixAxis(nside=1, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis(np.geomspace(100, 1000, 9) * u.keV, label='Ei', scale='log'),
        Axis([-1, -0.5, -0.2, -0.05, 0.05, 0.2], label='Epsilon'),
        Axis(np.linspace(0, 180, 7) * u.deg, label='Phi'),
        Axis(np.linspace(-90, 90, 5) * u.deg, label='Theta'),
        PolarizationAxis(np.linspace(0, 360, 9) * u.deg, convention=StereographicConvention(), label='Zeta'),
    ])
    phi = axes['Phi'].centers.to_value(u.rad)[:, None]
    theta = axes['Theta'].centers.to_value(u.rad)[None, :]
    physical = (phi + theta > 0) & (phi + theta < np.pi)
    contents = rng.random(axes.nbins) * physical[None, None, None, :, :, None]

    return Histogram(axes, contents=contents, unit=u.cm * u.cm)


@pytest.fixture
def irf(tmp_path):
    path = tmp_path / 'irf.h5'
    make_irf_hist().write(path, name='IRF')
    return IRFRelativeHistUnpolarized.from_h5(path)


@pytest.fixture
def cache(irf, tmp_path):
    return BroadenedIRFCache(irf, tmp_path / 'cache.h5', energy_range_keV=(100., 1000.), max_level=3)


PIXEL = 5


def pixel_direction(irf, pixel=PIXEL):
    lon, lat = irf.axes['NuLambda'].pix2ang(pixel, lonlat=True)
    return np.deg2rad(float(lon)), np.deg2rad(float(lat))


def test_geometry(cache):
    assert cache.ncells(0) == 1 and cache.ncells(2) == 4
    np.testing.assert_allclose(cache.cell_edges_keV(1, 1), [10 ** 2.5, 1000])

    edges = cache.epsilon_edges(2, 3)
    assert np.all(np.diff(edges) > 0)
    assert np.all(np.isin([-1, -0.5, -0.2, -0.05, 0.05, 0.2], edges))
    # Highest Em reachable in a cell: its upper edge times 1 + eps_max
    assert edges[-1] == pytest.approx(10 ** 0.25 * 1.2 - 1)


def test_area_pieces_are_the_integral_of_the_hats(cache, irf):
    cache.ensure([(1, 1)], [PIXEL])
    rows = cache._read_rows(1, 1, [PIXEL])

    lo, hi = cache.cell_edges_keV(1, 1)
    energy = np.geomspace(lo, hi, 100_001)
    t = np.log(energy / lo) / np.log(hi / lo)
    aeff = irf._effective_area_at_pixel(PIXEL, energy)

    assert rows['AEFF_R'].item() == pytest.approx(np.trapezoid(t * energy ** -2. * aeff, energy), rel=1e-3)
    assert rows['AEFF_L'].item() == pytest.approx(np.trapezoid((1 - t) * energy ** -2. * aeff, energy), rel=1e-3)


@pytest.mark.parametrize('piece', ['R', 'L'])
def test_kernel_matches_the_integral_over_energy(cache, irf, piece):
    """The differential effective area of a piece, evaluated at the lower edge of the
    cell, is the integral of the hat times w times the differential effective area of
    the IRF, up to the binning of the piece in Epsilon'."""


    level, index = 1, 1
    cache.ensure([(level, index)], [PIXEL])
    piece_irf = dict(zip('RL', cache.piece_irfs(level, index)))[piece]
    lo, hi = cache.cell_edges_keV(level, index)
    lon, lat = pixel_direction(irf)

    rng = np.random.default_rng(1)
    n = 200
    events = EmCDSEventDataInSCFrameFromArrays(rng.uniform(0.7 * lo, 1.1 * hi, n), rng.uniform(-np.pi, np.pi, n),
                                               np.arcsin(rng.uniform(-1, 1, n)), rng.uniform(0.3, 2.5, n))
    kernel = np.asarray(piece_irf.differential_effective_area_cm2(
        PhotonListWithDirectionAndEnergyInSCFrame(np.full(n, lon), np.full(n, lat), np.full(n, lo)), events))

    energy = np.geomspace(lo, hi, 2001)
    t = np.log(energy / lo) / np.log(hi / lo)
    hat = t if piece == 'R' else 1 - t
    brute = np.zeros(n)
    for e, h, de in zip(energy, hat, np.gradient(energy)):
        photons = PhotonListWithDirectionAndEnergyInSCFrame(np.full(n, lon), np.full(n, lat), np.full(n, e))
        brute += h * e ** -2. * np.asarray(irf.differential_effective_area_cm2(photons, events)) * de

    # Weighted by the kernel: the relative error is large only in the far tails of a
    # piece, where it is tiny and the neighboring pieces dominate. It decreases as the
    # square of steps_per_cell (1.2-2.7% with 8, 0.4-0.8% with the default 16).
    assert (brute > 0).sum() > 100
    assert np.sum(np.abs(kernel - brute)) / np.sum(brute) < 0.01


def test_merge_from_children_equals_direct(cache):
    cache.ensure([(1, 0), (1, 1)], [PIXEL])
    cache.ensure([(0, 0)], [PIXEL])  # from the children
    merged = cache._read_rows(0, 0, [PIXEL])
    direct = cache._direct(0, 0, np.array([PIXEL]))

    for name in ('AEFF_R', 'AEFF_L'):
        np.testing.assert_allclose(merged[name], direct[name], rtol=1e-4)
    for name in ('R', 'L'):
        # The tables are rebinned from the children's Epsilon' bins
        np.testing.assert_allclose(merged[name].sum(axis=2), direct[name].sum(axis=2), rtol=1e-3)


def test_precompute_and_persistence(cache, irf, tmp_path):
    cache.precompute(1, pixels=[PIXEL, PIXEL + 1])

    for level, n in ((0, 1), (1, 2)):
        for index in range(n):
            computed = cache.computed(level, index)
            assert computed[PIXEL] and computed[PIXEL + 1] and computed.sum() == 2
    assert not cache.computed(2, 0).any()

    reopened = BroadenedIRFCache(irf, tmp_path / 'cache.h5', energy_range_keV=(100., 1000.), max_level=3)
    assert reopened.computed(1, 1)[PIXEL]

    with pytest.raises(ValueError):
        BroadenedIRFCache(irf, tmp_path / 'cache.h5', energy_range_keV=(100., 1000.), weight_index=-1.)


def test_single_writer(cache, tmp_path):
    lock = tmp_path / 'cache.h5.lock'
    lock.touch()

    with pytest.raises(RuntimeError):
        cache.ensure([(0, 0)], [PIXEL])
    assert not cache.computed(0, 0).any()

    lock.unlink()
    cache.ensure([(0, 0)], [PIXEL])
    assert cache.computed(0, 0)[PIXEL]
    assert not lock.exists()


def test_selected_area_is_consistent_with_the_density(cache, irf):
    """The piece IRFs apply selections like any IRF: a cut covering everything leaves
    the area unchanged, and a partial cut reduces it."""

    cache.ensure([(1, 0)], [PIXEL])
    lo = cache.cell_edges_keV(1, 0)[0]
    lon, lat = pixel_direction(irf)
    photons = PhotonListWithDirectionAndEnergyInSCFrame(np.array([lon]), np.array([lat]), np.array([lo]))

    full = cache.piece_irfs(1, 0)[0].effective_area_cm2(photons)
    everything = cache.piece_irfs(1, 0, selections=EnergySelectorAll())[0].effective_area_cm2(photons)
    cut = cache.piece_irfs(1, 0, selections=EnergySelectorCut())[0].effective_area_cm2(photons)

    assert everything == pytest.approx(full, rel=1e-12)
    assert 0 < cut < full


def EnergySelectorAll():
    from cosipy.event_selection.energy_selection import EnergySelector
    return EnergySelector([0., 1e9] * u.keV)


def EnergySelectorCut():
    from cosipy.event_selection.energy_selection import EnergySelector
    return EnergySelector([150., 250.] * u.keV)
