from unittest.mock import MagicMock

import numpy as np
import pytest
import astropy.units as u
from astropy.time import Time
from scipy.stats import chi2, kstest

from cosipy.background_estimation.analytic_background import IsotropicPowerLawBackground
from cosipy.spacecraftfile import SpacecraftHistory

EDGES = 1000. + np.array([0., 10., 25., 30., 50.])  # unix s
LIVETIME = np.array([8., 15., 0., 18.])  # s


@pytest.fixture
def sc_history():
    sc = MagicMock(spec=SpacecraftHistory)
    sc.obstime = Time(EDGES, format='unix')
    sc.tstart, sc.tstop = sc.obstime[0], sc.obstime[-1]
    sc.nintervals = EDGES.size - 1
    sc.livetime = LIVETIME * u.s
    sc.intervals_duration = np.diff(EDGES) * u.s
    sc.cumulative_livetime.return_value = LIVETIME.sum() * u.s
    return sc


def test_random_events_follow_the_model(sc_history):
    rng = np.random.default_rng(0)
    rate, index, erange = 400 * u.Hz, -2.3, (200., 5000.)
    events = IsotropicPowerLawBackground.random_events(sc_history, rate, index, erange, rng)

    n = events.nevents
    expected = 400 * LIVETIME.sum()
    assert abs(n - expected) < 4 * np.sqrt(expected)

    # Times: sorted, within the intervals, in proportion to their live time
    unix = events.time.utc.unix
    assert np.all(np.diff(unix) >= 0)
    counts = np.histogram(unix, EDGES)[0]
    assert counts[2] == 0
    live = LIVETIME > 0
    assert chi2.sf(np.sum((counts[live] - n * LIVETIME[live] / LIVETIME.sum()) ** 2
                          / (n * LIVETIME[live] / LIVETIME.sum())), live.sum() - 1) > 1e-3

    # Energy: the power law
    e = np.asarray(events.energy_keV)
    a, b = erange[0] ** (index + 1), erange[1] ** (index + 1)
    assert kstest(e, lambda x: (x ** (index + 1) - a) / (b - a)).pvalue > 1e-3

    # Phi uniform in [0, pi], PsiChi isotropic
    assert kstest(np.asarray(events.scattering_angle_rad) / np.pi, 'uniform').pvalue > 1e-3
    assert kstest((np.sin(np.asarray(events.scattered_lat_rad_sc)) + 1) / 2, 'uniform').pvalue > 1e-3
    assert kstest(np.asarray(events.scattered_lon_rad_sc) / (2 * np.pi), 'uniform').pvalue > 1e-3


def test_density_and_counts(sc_history):
    rng = np.random.default_rng(1)
    events = IsotropicPowerLawBackground.random_events(sc_history, 50 * u.Hz, -2., (200., 5000.), rng)
    bkg = IsotropicPowerLawBackground(events, sc_history, 10 * u.Hz, index=-2., energy_range_keV=(200., 5000.))

    assert bkg.expected_counts() == pytest.approx(10 * LIVETIME.sum())

    e = np.asarray(events.energy_keV)
    interval = np.searchsorted(EDGES, events.time.utc.unix, side='right') - 1
    live_fraction = (LIVETIME / np.diff(EDGES))[interval]
    expected = 10 * live_fraction * e ** -2. / (1 / 200 - 1 / 5000) / np.pi / (4 * np.pi)
    np.testing.assert_allclose(bkg.expectation_density(), expected)

    bkg.set_parameters(bkg_rate=20 * u.Hz)
    assert bkg.parameters['bkg_rate'] == 20 * u.Hz
    np.testing.assert_allclose(bkg.expectation_density(), 2 * expected)
    assert bkg.expected_counts() == pytest.approx(20 * LIVETIME.sum())


def test_events_outside_the_history_raise(sc_history):
    events = IsotropicPowerLawBackground.random_events(sc_history, 5 * u.Hz, rng=np.random.default_rng(2))
    sc_history.tstop = Time(1010., format='unix')

    with pytest.raises(ValueError):
        IsotropicPowerLawBackground(events, sc_history, 1 * u.Hz)
