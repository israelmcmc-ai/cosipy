from unittest.mock import MagicMock

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astromodels import PointSource

from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.response.broadened_relative_irf_hist import hat_integrals_keV
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.threeml.psr_broadened import UnbinnedThreeMLPointSourceResponseBroadened

NODES = np.geomspace(100, 10_000, 13)
LIVETIME = np.array([10., 8., 9.])
DURATION = np.array([10., 10., 10.])
OCCULTED = np.array([False, False, True])
LON = np.array([0.1, 0.2, 0.3])


def aeff(energy):
    return 100 * (energy / 1000) ** 0.3


def diff_aeff(energy, em):
    return np.exp(-(em - energy) ** 2 / (0.1 * energy) ** 2) / energy


def spectrum(energy):
    return 1e-3 * (energy / 1000) ** -2.3


def _irf(nodes):
    irf = MagicMock(spec=IRFRelativeHistUnpolarized)
    irf.axes = {'Ei': MagicMock(centers=nodes)}
    irf.effective_area_cm2.side_effect = lambda photons: aeff(np.asarray(photons.energy_keV))
    irf.differential_effective_area_cm2.side_effect = \
        lambda photons, events: diff_aeff(np.asarray(photons.energy_keV), np.asarray(events.energy_keV))
    return irf


@pytest.fixture
def sc_history():
    sc = MagicMock(spec=SpacecraftHistory)
    sc.obstime = Time(1e9 + np.array([0., 10., 20., 30.]), format='unix')
    sc.nintervals = 3
    sc.livetime = LIVETIME * u.s
    sc.intervals_duration = DURATION * u.s
    mid = MagicMock()
    mid.get_target_in_sc_frame.return_value = MagicMock(lon=MagicMock(rad=LON), lat=MagicMock(rad=np.zeros(3)))
    mid.get_earth_occ.return_value = OCCULTED
    sc.interp.return_value = mid
    return sc


@pytest.fixture
def data():
    data = MagicMock(spec=TimeTagEmCDSEventDataInSCFrameInterface)
    data.nevents = 4
    data.time = Time(1e9 + np.array([1., 12., 15., 25.]), format='unix')
    data.energy_keV = np.array([300., 1000., 1500., 800.])
    data.scattering_angle_rad = np.full(4, 0.5)
    data.scattered_lon_rad_sc = np.full(4, 0.1)
    data.scattered_lat_rad_sc = np.full(4, 0.2)
    return data


@pytest.fixture
def source():
    source = MagicMock(spec=PointSource)
    source.position = MagicMock(sky_coord=SkyCoord(10, 10, unit='deg'))
    source.to_dict.return_value = {'a': 1}
    source.side_effect = lambda energy: spectrum(np.asarray(energy))
    return source


def _psr(data, sc_history, source, **kwargs):
    psr = UnbinnedThreeMLPointSourceResponseBroadened(data, _irf(NODES), sc_history, offset=None, **kwargs)
    psr.set_source(source)
    return psr


def test_expected_counts_and_density_use_the_same_nodes(data, sc_history, source):
    psr = _psr(data, sc_history, source)
    delta = hat_integrals_keV(NODES)

    exposure = np.sum(LIVETIME * ~OCCULTED)
    assert psr.expected_counts() == pytest.approx(exposure * np.sum(delta * spectrum(NODES) * aeff(NODES)))

    interval = np.array([0, 1, 1, 2])
    weight = (LIVETIME / DURATION * ~OCCULTED)[interval]
    expected = weight * np.array([np.sum(delta * spectrum(NODES) * diff_aeff(NODES, em)) for em in data.energy_keV])
    assert np.allclose(psr.expectation_density(), expected)
    assert psr.expectation_density()[3] == 0


def test_fine_region_keeps_the_integration_weights_a_partition(data, sc_history, source):
    """With a flat integrand, the coarse and fine weights add up to the length of the range."""

    psr = _psr(data, sc_history, source, fine_irf=_irf(NODES), fine_regions=[[400, 700] * u.keV])
    assert np.sum(psr._coarse_weights) + np.sum(psr._fine_weights) == pytest.approx(NODES[-1] - NODES[0])
    assert np.count_nonzero(psr._coarse_weights == 0) == 2

    coarse = _psr(data, sc_history, source)
    assert psr.expected_counts() == pytest.approx(coarse.expected_counts(), rel=2e-3)


def test_lines_add_their_flux_times_the_response(data, sc_history, source):
    psr = _psr(data, sc_history, source, fine_irf=_irf(NODES), mono_nodes=[1000] * u.keV)
    coarse = _psr(data, sc_history, source)
    exposure = np.sum(LIVETIME * ~OCCULTED)
    assert psr.expected_counts() - coarse.expected_counts() == pytest.approx(exposure * spectrum(1000.) * aeff(1000.))


def test_requires_fine_irf_and_nodes_in_region(data, sc_history):
    with pytest.raises(ValueError):
        UnbinnedThreeMLPointSourceResponseBroadened(data, _irf(NODES), sc_history, fine_regions=[[400, 700] * u.keV])
    with pytest.raises(ValueError):
        UnbinnedThreeMLPointSourceResponseBroadened(data, _irf(NODES), sc_history, fine_irf=_irf(NODES),
                                                    fine_regions=[[410, 420] * u.keV])
    with pytest.raises(ValueError):
        UnbinnedThreeMLPointSourceResponseBroadened(data, _irf(NODES), sc_history, mono_nodes=[511] * u.keV)
