from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astromodels import PointSource, Gaussian, DiracDelta, Powerlaw
from histpy import Axis
from scoords import SpacecraftFrame
from scipy.integrate import quad
from scipy.stats import norm

from cosipy.interfaces.data_interface import TimeTagEmCDSEventDataInSCFrameInterface
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.threeml.psr_fixed_ei import UnbinnedThreeMLPointSourceResponseTrapz

AEFF = 100.  # cm2
RES = 0.01  # Fractional energy resolution (sigma)
LIVETIME = 10.  # s


class ToyIRF:
    """
    Constant effective area, Gaussian energy dispersion, no angular dependence.
    """

    def __init__(self, res=RES):
        self.res = res

    def effective_area_cm2(self, photons):
        return np.full(photons.nphotons, AEFF)

    def differential_effective_area_cm2(self, photons, events):
        ei = np.asarray(photons.energy_keV)
        em = np.asarray(events.energy_keV)
        return AEFF * norm.pdf(em, loc=ei, scale=self.res * ei)


@pytest.fixture
def energy_m():
    return np.array([400., 505., 1000., 1805., 1808., 1812.])


@pytest.fixture
def data(energy_m):
    n = energy_m.size
    data = MagicMock(spec=TimeTagEmCDSEventDataInSCFrameInterface)
    data.nevents = n
    data.time.utc.unix = 1000. + np.arange(n) / n
    data.energy_keV = energy_m
    data.scattering_angle_rad = np.full(n, 0.5)
    data.scattered_lon_rad_sc = np.full(n, 0.1)
    data.scattered_lat_rad_sc = np.full(n, 0.2)
    return data


@pytest.fixture
def sc_history():
    sc = MagicMock(spec=SpacecraftHistory)
    sc.obstime = Time([1000., 1001.], format='unix')
    sc.nintervals = 1
    sc.livetime = [LIVETIME] * u.s
    sc.intervals_duration = [LIVETIME] * u.s

    def interp(times):
        interp_sc = MagicMock()
        interp_sc.get_target_in_sc_frame.return_value = SkyCoord(lon=np.zeros(times.size), lat=np.zeros(times.size),
                                                                 unit='rad', frame=SpacecraftFrame())
        interp_sc.get_earth_occ.return_value = np.zeros(times.size, dtype=bool)
        return interp_sc

    sc.interp.side_effect = interp
    return sc


def make_psr(data, sc_history, res=RES, **kwargs):
    kwargs.setdefault('energies', np.geomspace(100, 5000, 200) * u.keV)
    kwargs.setdefault('epsilon_axis', Axis(np.linspace(-0.1, 0.1, 41)))
    kwargs.setdefault('offset', None)
    return UnbinnedThreeMLPointSourceResponseTrapz(data, ToyIRF(res), sc_history, **kwargs)


def expected_density(em, spectrum, emin, emax, eps_min=-0.1, eps_max=0.1, res=RES):
    # Dispersion is truncated at the Epsilon axis range
    lo = max(emin, em / (1 + eps_max))
    hi = min(emax, em / (1 + eps_min))
    if lo >= hi:
        return 0
    return AEFF * quad(lambda e: norm.pdf(em, loc=e, scale=res * e) * spectrum(e), lo, hi,
                       points=[em] if lo < em < hi else None, limit=200)[0]


def test_continuum(data, sc_history, energy_m):
    spectrum = Powerlaw(K=1e-2, index=-2, piv=100)
    source = PointSource('src', l=0, b=0, spectral_shape=spectrum)

    psr = make_psr(data, sc_history)
    psr.set_source(source)

    nexp = LIVETIME * AEFF * quad(spectrum, 100, 5000)[0]
    assert psr.expected_counts() == pytest.approx(nexp, rel=1e-3)

    density = psr.expectation_density()
    for em, d in zip(energy_m, density):
        assert d == pytest.approx(expected_density(em, spectrum, 100, 5000), rel=1e-3)


def test_broad_dispersion(data, sc_history, energy_m):
    # Wide Epsilon bins with a steep spectrum. The Epsilon nodes alone are not enough.
    spectrum = Powerlaw(K=1e-2, index=-3, piv=100)
    source = PointSource('src', l=0, b=0, spectral_shape=spectrum)

    psr = make_psr(data, sc_history, res=0.3, epsilon_axis=Axis(np.linspace(-0.9, 0.9, 7)),
                   energies=np.geomspace(100, 5000, 100) * u.keV)
    psr.set_source(source)

    density = psr.expectation_density()
    for em, d in zip(energy_m, density):
        assert d == pytest.approx(expected_density(em, spectrum, 100, 5000, -0.9, 0.9, res=0.3), rel=1e-2)


def test_narrow_line(data, sc_history, energy_m):
    spectrum = Gaussian(F=1e-3, mu=1805., sigma=1.)
    source = PointSource('src', l=0, b=0, spectral_shape=spectrum)

    # With a coarse grid the 1 keV wide line falls between nodes
    psr = make_psr(data, sc_history, energies=np.linspace(1790, 1830, 3) * u.keV)
    psr.set_source(source)
    coarse = psr.expected_counts()

    psr = make_psr(data, sc_history, energies=np.linspace(1790, 1830, 401) * u.keV)
    psr.set_source(source)

    nexp = LIVETIME * AEFF * 1e-3
    assert psr.expected_counts() == pytest.approx(nexp, rel=1e-3)
    assert abs(coarse - nexp) / nexp > 0.1

    density = psr.expectation_density()
    for em, d in zip(energy_m, density):
        assert d == pytest.approx(expected_density(em, spectrum, 1790, 1830), rel=1e-3, abs=1e-12)


def test_dirac_delta_line(data, sc_history, energy_m):
    spectrum = DiracDelta(value=1e-3, zero_point=1808.)
    source = PointSource('src', l=0, b=0, spectral_shape=spectrum)

    psr = make_psr(data, sc_history, line_energies=[1808.] * u.keV)
    psr.set_source(source)

    assert psr.expected_counts() == pytest.approx(LIVETIME * AEFF * 1e-3)

    density = psr.expectation_density()
    expected = AEFF * norm.pdf(energy_m, loc=1808., scale=RES * 1808.) * 1e-3
    expected[np.abs(energy_m / 1808. - 1) > 0.1] = 0
    np.testing.assert_allclose(density, expected, rtol=1e-10)


def test_event_nodes_follow_epsilon(data, sc_history):
    psr = make_psr(data, sc_history, energies=np.geomspace(100, 5000, 5) * u.keV)

    nodes, weights = psr._event_nodes(np.array([1000., 5000.]))

    # Nodes cover Em/(1+eps) for eps in [-0.1, 0.1], limited to the energies range
    nonzero = weights[0] > 0
    assert nodes[0][nonzero].min() == pytest.approx(1000 / 1.1)
    assert nodes[0][nonzero].max() == pytest.approx(1000 / 0.9)
    assert weights[0].sum() == pytest.approx(1000 / 0.9 - 1000 / 1.1)

    assert nodes[1][weights[1] > 0].max() == pytest.approx(5000)
    assert weights[1].sum() == pytest.approx(5000 - 5000 / 1.1)
