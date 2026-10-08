from types import SimpleNamespace

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import GCRS, Galactic, SkyCoord
from astropy.time import Time
from histpy import Axes, Axis, HealpixAxis, Histogram
from scoords import Attitude, SpacecraftFrame

from cosipy import SpacecraftHistory
from cosipy.background_estimation import (FreeNormHistBackgroundDensity,
                                          HistBackgroundTemplate,
                                          rocking_angle)
from cosipy.event_selection import EnergySelector
from cosipy.interfaces import BackgroundDensityInterface

DT = 1000.  # s
NSIDE = 4
ENERGY_EDGES = np.geomspace(100., 1000., 5)
T0 = Time('2028-03-24T10:00:00', scale='utc')


def make_history(tilts_deg, livetime=None, dt=DT):
    """
    One interval per tilt. The z axis is tilted from the Earth zenith by
    `tilt` degrees, positive toward the celestial north.
    """

    npoints = len(tilts_deg) + 1
    tilts = np.deg2rad(np.append(tilts_deg, tilts_deg[-1]))
    obstime = T0 + np.arange(npoints) * dt * u.s
    location = GCRS(ra=np.full(npoints, 40.) * u.deg, dec=np.full(npoints, 20.) * u.deg,
                    distance=np.full(npoints, 7000.) * u.km)

    if livetime is None:
        livetime = np.full(npoints - 1, dt) * u.s

    zenith = SpacecraftHistory(obstime, Attitude.identity(frame='galactic'), location,
                               livetime).earth_zenith.cartesian.xyz.value
    pole = SkyCoord(ra=0 * u.deg, dec=90 * u.deg).galactic.cartesian.xyz.value[:, None]
    north = pole - np.sum(pole * zenith, axis=0) * zenith
    north /= np.linalg.norm(north, axis=0)

    z = np.cos(tilts) * zenith + np.sin(tilts) * north
    x = np.cross(north.T, z.T).T
    x /= np.linalg.norm(x, axis=0)

    def to_coord(v):
        return SkyCoord(*v, representation_type='cartesian', frame=Galactic())

    attitude = Attitude.from_axes(x=to_coord(x), z=to_coord(z), frame=Galactic())

    return SpacecraftHistory(obstime, attitude, location, livetime)


def make_data(sc_history, dt_since_start, energy, phi, lon=0.3, lat=0.2):
    n = len(dt_since_start)
    return SimpleNamespace(time=sc_history.tstart + np.asarray(dt_since_start, dtype=float) * u.s,
                           energy_keV=np.broadcast_to(np.asarray(energy, dtype=float), n),
                           scattering_angle_rad=np.broadcast_to(np.asarray(phi, dtype=float), n),
                           scattered_lon_rad_sc=np.broadcast_to(np.asarray(lon, dtype=float), n),
                           scattered_lat_rad_sc=np.broadcast_to(np.asarray(lat, dtype=float), n))


def make_axes(time_edges, nphi=6, em_unit=None, phi_unit=None):
    psichi = HealpixAxis(nside=NSIDE, scheme='nested', coordsys=SpacecraftFrame(), label='PsiChi')
    em = Axis(ENERGY_EDGES * (em_unit.to(u.keV) ** -1 if em_unit else 1), scale='log', label='Em',
              unit=em_unit)
    rocking = Axis([-90., 0., 90.], label='Rocking', unit=u.deg)
    phi_edges = np.linspace(0, np.pi, nphi + 1)
    phi = Axis(np.rad2deg(phi_edges) if phi_unit == u.deg else phi_edges, label='Phi', unit=phi_unit)
    return (Axes([Axis(time_edges, label='Time'), em]),
            Axes([rocking, em, phi]),
            Axes([rocking, em, phi, psichi]))


def analytic_template(time_edges, livetime, rate=0.01, **kwargs):
    """
    Flat rate (counts/s/keV) and uniform PsiChi. The phi density is
    2 phi/pi^2 for the rocking bin [-90, 0) and 2 (pi - phi)/pi^2 for [0, 90]. All are
    linear (flat) in phi (Em, PsiChi), so interpolation is exact.
    """

    rate_axes, phi_axes, psichi_axes = make_axes(time_edges, **kwargs)
    template = HistBackgroundTemplate.empty(rate_axes, phi_axes, psichi_axes)

    template.livetime[...] = livetime
    template.rate_counts[...] = rate * livetime[:, None] * np.diff(ENERGY_EDGES)[None, :]

    phi_edges = np.linspace(0, np.pi, 7)
    phi_center, phi_width = (phi_edges[1:] + phi_edges[:-1]) / 2, np.diff(phi_edges)
    pdf = np.array([2 * phi_center / np.pi ** 2, 2 * (np.pi - phi_center) / np.pi ** 2])
    phi_counts = 1000 * pdf * phi_width
    template.phi_counts[...] = np.broadcast_to(phi_counts[:, None, :], template.phi_counts.shape)
    template.psichi_counts[...] = np.broadcast_to(phi_counts[:, None, :, None] / template.psichi_counts.axes['PsiChi'].npix,
                                                  template.psichi_counts.shape)

    return template


def test_rocking_angle():
    tilts = np.array([22., 22., -22., -22., 0., 150., -150.])
    angle = rocking_angle(make_history(tilts))

    assert angle.unit.is_equivalent(u.deg)
    assert angle.shape == tilts.shape
    assert np.allclose(angle.to_value(u.deg), tilts, atol=1e-6)


def test_fill():
    # Intervals 0,1 are tilted north, 2,3 south. Livetime fractions 0.8, 1, 0.5, 1
    live = np.array([800., 1000., 500., 1000.])
    sc_history = make_history([22., 22., -22., -22.], live * u.s)
    t0 = sc_history.tstart.utc.unix
    time_edges = t0 + np.array([-500., 500., 1500., 2500., 3500., 4500.])

    template = HistBackgroundTemplate.empty(*make_axes(time_edges))

    # (t since start, Em, phi), psichi fixed at (lon, lat) = (0.3, 0.2)
    events = np.array([[100., 150., 0.2],    # time bin 0, north
                       [1200., 150., 0.2],   # time bin 1, north
                       [2200., 500., 3.0],   # time bin 2, south
                       [3500., 999., 1.0],   # time bin 3, south
                       [3900., 50., 1.0],    # energy out of range: only in the Time/livetime part
                       ])
    template.fill(make_data(sc_history, *events.T), sc_history)

    rate = template.rate_counts.contents
    assert rate.sum() == 4
    assert rate[0, 0] == 1 and rate[1, 0] == 1 and rate[2, 2] == 1 and rate[4, 3] == 1

    # Rocking bin 0 is [-90, 0) i.e. south, bin 1 is [0, 90] i.e. north
    phi = template.phi_counts.contents
    assert phi.sum() == 4
    assert phi[1, 0, 0] == 2 and phi[0, 2, 5] == 1 and phi[0, 3, 1] == 1

    psichi = template.psichi_counts.contents
    pixel = template.psichi_counts.axes['PsiChi'].ang2pix(np.pi / 2 - 0.2, 0.3)
    assert psichi.sum() == 4
    assert psichi[1, 0, 0, pixel] == 2 and psichi[0, 2, 5, pixel] == 1 and psichi[0, 3, 1, pixel] == 1

    # Livetime density is piecewise constant, so compute the overlap with each interval
    interval_edges = t0 + np.arange(5) * DT
    expected = np.zeros(5)
    for i in range(5):
        for j in range(4):
            overlap = (min(time_edges[i + 1], interval_edges[j + 1]) - max(time_edges[i], interval_edges[j]))
            expected[i] += max(overlap, 0.) * live[j] / DT
    assert expected.sum() == pytest.approx(live.sum())
    assert np.allclose(template.livetime.contents, expected)

    # Second chunk accumulates
    template.fill(make_data(sc_history, [100.], 150., 0.2), sc_history)
    assert template.rate_counts.contents.sum() == 5
    assert template.livetime.contents.sum() == pytest.approx(2 * live.sum())


def test_fill_errors():
    sc_history = make_history([22., 22.])
    template = HistBackgroundTemplate.empty(*make_axes(sc_history.tstart.utc.unix + np.array([0., 2000.])))

    with pytest.raises(ValueError):
        template.fill(make_data(sc_history, [2500.], 150., 0.2), sc_history)

    with pytest.raises(ValueError):
        template.fill(make_data(sc_history, [-1.], 150., 0.2), sc_history)

    rate_axes, phi_axes, psichi_axes = make_axes(np.array([0., 1.]))
    galactic = HealpixAxis(nside=NSIDE, coordsys='galactic', label='PsiChi')
    bad = HistBackgroundTemplate.empty(rate_axes, phi_axes,
                                       Axes(list(psichi_axes[:3]) + [galactic]))
    with pytest.raises(NotImplementedError):
        bad.fill(make_data(sc_history, [10.], 150., 0.2), sc_history)


def filled_template(seed, sc_history):
    rng = np.random.default_rng(seed)
    n = 200
    template = HistBackgroundTemplate.empty(*make_axes(sc_history.tstart.utc.unix + np.arange(0, 4001, 500.)))
    data = make_data(sc_history, rng.uniform(0, 3999, n), rng.uniform(100, 1000, n),
                     rng.uniform(0, np.pi, n), rng.uniform(-np.pi, np.pi, n), rng.uniform(-1.5, 1.5, n))
    template.fill(data, sc_history)
    return template


def test_write_open_and_add(tmp_path):
    first = make_history([22., 22., -22., -22.])
    second = make_history([22., -22., -22., 22.])

    both = filled_template(0, first)
    both.fill(make_data(second, [10., 3000.], 200., 1., 0.1, 0.1), second)

    summed = filled_template(0, first)
    other = HistBackgroundTemplate.empty(*make_axes(first.tstart.utc.unix + np.arange(0, 4001, 500.)))
    other.fill(make_data(second, [10., 3000.], 200., 1., 0.1, 0.1), second)
    summed += other

    for name in HistBackgroundTemplate._names:
        assert np.array_equal(getattr(summed, name).contents, getattr(both, name).contents)
        assert getattr(summed, name).axes == getattr(both, name).axes

    assert (filled_template(0, first) + other).livetime.contents.sum() == pytest.approx(2 * 4 * DT)

    mismatch = HistBackgroundTemplate.empty(*make_axes(first.tstart.utc.unix + np.arange(0, 4001, 1000.)))
    with pytest.raises(ValueError):
        both + mismatch

    filename = tmp_path / 'template.h5'
    both.write(filename)

    with pytest.raises(FileExistsError):
        both.write(filename)
    both.write(filename, overwrite=True)

    loaded = HistBackgroundTemplate.open(filename)
    for name in HistBackgroundTemplate._names:
        assert np.array_equal(getattr(loaded, name).contents, getattr(both, name).contents)
        assert getattr(loaded, name).axes == getattr(both, name).axes
    assert isinstance(loaded.psichi_counts.axes['PsiChi'].coordsys, SpacecraftFrame)


def test_smooth():
    sc_history = make_history([22., 22., -22., -22.])
    template = filled_template(1, sc_history)
    original = template.psichi_counts.contents.copy()

    smoothed = template.smooth(psichi_fwhm=40 * u.deg)

    # The input is untouched and the totals of every map are preserved
    assert np.array_equal(template.psichi_counts.contents, original)
    assert not np.allclose(smoothed.psichi_counts.contents, original)
    assert smoothed.psichi_counts.contents.min() >= 0
    assert np.allclose(smoothed.psichi_counts.contents.sum(axis=-1), original.sum(axis=-1))
    assert np.array_equal(smoothed.phi_counts.contents, template.phi_counts.contents)

    # A single count of a map spreads out to the neighbouring pixels
    point = HistBackgroundTemplate.empty(*make_axes(sc_history.tstart.utc.unix + np.array([0., 4000.])))
    point.psichi_counts[0, 0, 0, 10] = 5.
    spread = point.smooth(psichi_fwhm=40 * u.deg).psichi_counts.contents[0, 0, 0]
    assert spread.sum() == pytest.approx(5.)
    assert spread.argmax() == 10 and (spread > 0).sum() > 1


def test_smooth_time_gap():
    time_edges = np.arange(0., 2001., 100.)
    template = HistBackgroundTemplate.empty(*make_axes(time_edges))

    livetime = np.full(20, 80.)
    livetime[8:11] = 0.
    template.livetime[...] = livetime
    template.rate_counts[...] = 0.5 * livetime[:, None] * np.diff(ENERGY_EDGES)[None, :]

    smoothed = template.smooth(time_fwhm=300 * u.s)

    rate = smoothed.rate_counts.contents / smoothed.livetime.contents[:, None]
    expected = 0.5 * np.diff(ENERGY_EDGES)[None, :]

    assert np.allclose(rate, np.broadcast_to(expected, rate.shape))
    assert smoothed.livetime.contents[9] > 0     # The gap is now filled

    with pytest.raises(ValueError):
        HistBackgroundTemplate.empty(*make_axes(np.array([0., 1., 3.]))).smooth(time_fwhm=1 * u.s)


@pytest.fixture
def window():
    # 4 intervals of 1000 s: north, north, south, south. Livetime fractions 0.8, 1, 0.5, 1
    live = np.array([800., 1000., 500., 1000.])
    sc_history = make_history([22., 22., -22., -22.], live * u.s)
    t0 = sc_history.tstart.utc.unix

    # Different time binning than the history, with the livetime of a flat 0.8 fraction
    time_edges = t0 + np.array([-1000., 1000., 3000., 5000.])
    template = analytic_template(time_edges, np.array([1600., 1600., 1600.]))

    return sc_history, template


def test_interface_and_parameters(window):
    sc_history, template = window
    data = make_data(sc_history, [100., 2500.], 300., 1.)

    bkg = FreeNormHistBackgroundDensity(data, sc_history, template)
    assert isinstance(bkg, BackgroundDensityInterface)
    assert bkg.labels == ('bkg_norm',)
    assert list(bkg.parameters) == ['bkg_norm']

    components = {'a': template, 'b': template}
    bkg = FreeNormHistBackgroundDensity(data, sc_history, components)
    assert list(bkg.parameters) == ['a', 'b']

    density = np.asarray(bkg.expectation_density())
    counts = bkg.expected_counts()

    norm_b = bkg.parameters['b']
    bkg.set_parameters(a=2 * bkg.parameters['a'])
    assert bkg.parameters['b'] == norm_b
    assert bkg.expected_counts() == pytest.approx(1.5 * counts)
    assert np.allclose(bkg.expectation_density(), 1.5 * density)

    bkg.set_parameters(a=0 * u.Hz, b=3 * u.mHz)
    assert bkg.expected_counts() == pytest.approx(3e-3 * 3300.)

    with pytest.raises(ValueError):
        bkg.set_parameters(c=1 * u.Hz)


def test_density_values(window):
    sc_history, template = window

    # Event 0: north (rocking bin 1), event 1: south (rocking bin 0)
    phi = np.array([1.0, 1.0])
    energy = np.array([300., 700.])
    data = make_data(sc_history, [100., 2500.], energy, phi)

    bkg = FreeNormHistBackgroundDensity(data, sc_history, template)

    total_live = 3300.
    # 0.01 counts/s/keV over the template livetime of 1600 s in each of 3 bins, restricted to the
    # analysis window by the livetime of the history, 900 keV wide
    expected_counts = 0.01 * 900. * total_live
    assert bkg.expected_counts() == pytest.approx(expected_counts)
    assert bkg.parameters['bkg_norm'].to_value(u.Hz) == pytest.approx(0.01 * 900.)

    f_live = np.array([0.8, 0.5])
    p_phi = np.array([2 * (np.pi - 1.0), 2 * 1.0]) / np.pi ** 2
    p_psichi = 1 / (4 * np.pi)

    # rate * f_live * p(phi) * p(psichi) / N, times the counts
    expected = 0.01 * f_live * p_phi * p_psichi
    assert np.allclose(bkg.expectation_density(), expected)

    # Norm scales the result, the shape is unchanged
    bkg.set_parameters(bkg_norm=2 * bkg.parameters['bkg_norm'])
    assert bkg.expected_counts() == pytest.approx(2 * expected_counts)
    assert np.allclose(bkg.expectation_density(), 2 * expected)


def test_density_integrates_to_one(window):
    sc_history, template = window

    # Non-uniform PsiChi distribution
    psichi_axis = template.psichi_counts.axes['PsiChi']
    z = psichi_axis.pix2vec(np.arange(psichi_axis.npix))[2]
    template.psichi_counts[...] = (1 + 0.5 * z)[None, None, None, :] * np.ones(template.psichi_counts.shape)

    # Grid over (Em, phi, PsiChi) at a fixed time in interval 1 (north, f_live = 1)
    nem, nphi = 10, 30
    energy_edges = np.geomspace(100., 1000., nem + 1)
    energy = np.sqrt(energy_edges[1:] * energy_edges[:-1])
    phi = (np.arange(nphi) + 0.5) * np.pi / nphi

    npix = psichi_axis.npix
    ie, ip, ipix = np.meshgrid(np.arange(nem), np.arange(nphi), np.arange(npix), indexing='ij')
    ie, ip, ipix = ie.ravel(), ip.ravel(), ipix.ravel()

    colat, lon = psichi_axis.pix2ang(ipix, lonlat=False)
    data = make_data(sc_history, np.full(ie.size, 1500.), energy[ie], phi[ip], lon, np.pi / 2 - colat)

    bkg = FreeNormHistBackgroundDensity(data, sc_history, template)
    prob = np.asarray(bkg.expectation_density()) / bkg.expected_counts()

    weights = np.diff(energy_edges)[ie] * (np.pi / nphi) * (4 * np.pi / npix)

    # P integrates to 1 over time, so at a fixed time it gives the instantaneous fraction.
    # The rate is constant, so it is f_live(t)/T_live
    assert np.sum(prob * weights) == pytest.approx(1.0 / 3300., rel=0.03)


def test_energy_selection(window):
    sc_history, template = window
    data = make_data(sc_history, [100., 2500.], 300., 1.)

    full = FreeNormHistBackgroundDensity(data, sc_history, template)

    # The flat rate makes the fraction the width ratio. The 100 - 1000 keV bins are log-spaced
    selection = EnergySelector(u.Quantity([[200., 300.], [500., 600.]], u.keV))
    selected = FreeNormHistBackgroundDensity(data, sc_history, template, selection)

    assert selected.expected_counts() == pytest.approx(full.expected_counts() * 200. / 900.)
    assert selected.parameters['bkg_norm'].value == pytest.approx(full.parameters['bkg_norm'].value * 200. / 900.)

    # The default norm scales with N, so the density at the events is untouched
    assert np.allclose(selected.expectation_density(), full.expectation_density())

    # Outside the template energy range
    empty = EnergySelector(u.Quantity([[2000., 3000.]], u.keV))
    with pytest.raises(ValueError):
        FreeNormHistBackgroundDensity(data, sc_history, template, empty)


def test_unit_axes_equivalent(window):
    sc_history, template = window
    data = make_data(sc_history, [100., 2500.], [300., 700.], 1.)

    in_units = analytic_template(sc_history.tstart.utc.unix + np.array([-1000., 1000., 3000., 5000.]),
                                 np.array([1600., 1600., 1600.]), em_unit=u.MeV, phi_unit=u.deg)

    reference = FreeNormHistBackgroundDensity(data, sc_history, template)
    converted = FreeNormHistBackgroundDensity(data, sc_history, in_units)

    assert converted.expected_counts() == pytest.approx(reference.expected_counts())
    assert np.allclose(converted.expectation_density(), reference.expectation_density())


def test_window_errors(window):
    sc_history, template = window
    data = make_data(sc_history, [100.], 300., 1.)

    # Time range not covered by the template
    short = analytic_template(sc_history.tstart.utc.unix + np.array([0., 1000., 2000.]),
                              np.array([1600., 1600.]))
    with pytest.raises(ValueError):
        FreeNormHistBackgroundDensity(data, sc_history, short)

    # Time bin without livetime
    gap = analytic_template(sc_history.tstart.utc.unix + np.array([-1000., 1000., 3000., 5000.]),
                            np.array([1600., 0., 1600.]))
    with pytest.raises(ValueError):
        FreeNormHistBackgroundDensity(data, sc_history, gap)

    # Events out of the history
    with pytest.raises(ValueError):
        FreeNormHistBackgroundDensity(make_data(sc_history, [5000.], 300., 1.), sc_history, template)

    # Not in the SC frame
    rate_axes, phi_axes, psichi_axes = make_axes(np.array([0., 1.]))
    galactic = HealpixAxis(nside=NSIDE, coordsys='galactic', label='PsiChi')
    bad = HistBackgroundTemplate.empty(rate_axes, phi_axes, Axes(list(psichi_axes[:3]) + [galactic]))
    with pytest.raises(NotImplementedError):
        FreeNormHistBackgroundDensity(data, sc_history, bad)
