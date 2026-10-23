from unittest.mock import MagicMock

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astromodels import PointSource, Powerlaw
from histpy import Axis, Axes, HealpixAxis, Histogram
from scipy.stats import kstest
from scoords import SpacecraftFrame

from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.polarization import PolarizationAxis, StereographicConvention
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized
from cosipy.source_injector.point_source_events import simulate_point_source_events
from cosipy.spacecraftfile import SpacecraftHistory

EDGES = 1000. + np.array([0., 100., 200., 300.])
LIVETIME = np.array([90., 95., 80.])
OCCULTED = np.array([False, True, False])
LON, LAT = 0.3, 0.6


def make_irf(**kwargs):
    rng = np.random.default_rng(0)
    axes = Axes([
        HealpixAxis(nside=1, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis(np.geomspace(50, 2000, 8) * u.keV, label='Ei', scale='log'),
        Axis([-1, -0.5, -0.2, -0.05, 0.05, 0.2], label='Epsilon'),
        Axis(np.linspace(0, 180, 7) * u.deg, label='Phi'),
        Axis(np.linspace(-90, 90, 5) * u.deg, label='Theta'),
        PolarizationAxis(np.linspace(0, 360, 9) * u.deg, convention=StereographicConvention(), label='Zeta'),
    ])
    phi = axes['Phi'].centers.to_value(u.rad)[:, None]
    theta = axes['Theta'].centers.to_value(u.rad)[None, :]
    physical = (phi + theta > 0) & (phi + theta < np.pi)
    contents = rng.random(axes.nbins) * physical[None, None, None, :, :, None] * 10
    return IRFRelativeHistUnpolarized(Histogram(axes, contents=contents, unit=u.cm ** 2), **kwargs)


@pytest.fixture
def sc_history():
    sc = MagicMock(spec=SpacecraftHistory)
    sc.obstime = Time(EDGES, format='unix')
    sc.nintervals = EDGES.size - 1
    sc.livetime = LIVETIME * u.s

    def interp(times):
        n = np.atleast_1d(times.unix).size
        mock = MagicMock()
        mock.get_target_in_sc_frame.return_value = SkyCoord(lon=np.full(n, LON), lat=np.full(n, LAT), unit='rad',
                                                            frame=SpacecraftFrame())
        mock.get_earth_occ.return_value = OCCULTED if n == OCCULTED.size else np.zeros(n, bool)
        return mock

    sc.interp.side_effect = interp
    return sc


@pytest.mark.parametrize('selection', [None, [200., 600.]])
def test_counts_energies_and_times(sc_history, selection):
    irf = make_irf(selections=None if selection is None else EnergySelector(selection * u.keV))
    source = PointSource('src', l=0, b=0, spectral_shape=Powerlaw(K=4e-5, index=-2, piv=100))
    erange = (100., 1000.)

    data, ei = simulate_point_source_events(irf, sc_history, source, erange, rng=np.random.default_rng(3))

    energy = np.geomspace(*erange, 20001)
    aeff = np.asarray(irf.effective_area_cm2(PhotonListWithDirectionAndEnergyInSCFrame(
        np.full(energy.size, LON), np.full(energy.size, LAT), energy)))
    rate = aeff * source(energy)
    expected = np.sum(LIVETIME * ~OCCULTED) * np.trapezoid(rate, energy)

    assert expected > 200
    assert abs(data.nevents - expected) < 4 * np.sqrt(expected)

    # Photon energies follow F Aeff
    cdf = np.concatenate([[0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(energy))])
    assert kstest(ei, lambda x: np.interp(x, energy, cdf / cdf[-1])).pvalue > 1e-3

    # No events in the occulted interval; times sorted
    unix = data.time.utc.unix
    assert np.all(np.diff(unix) >= 0)
    assert not np.any((unix >= EDGES[1]) & (unix < EDGES[2]))

    if selection is not None:
        em = np.asarray(data.energy_keV)
        assert np.all((em >= selection[0]) & (em < selection[1]))
