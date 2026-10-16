import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import UnitSphericalRepresentation
from histpy import Axis, Axes, HealpixAxis, Histogram
from scoords import SpacecraftFrame

from cosipy.data_io.EmCDSUnbinnedData import EmCDSEventDataInSCFrameFromArrays
from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.polarization import PolarizationAxis, StereographicConvention
from cosipy.response.broadened_relative_irf_hist import (hat_functions, hat_integrals_keV, broadened_epsilon_edges,
                                                        write_broadened_irf)
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized


def _Photons(lon_rad, lat_rad, energy_keV):
    return PhotonListWithDirectionAndEnergyInSCFrame(lon_rad, lat_rad, energy_keV)


def _Events(lon_rad, lat_rad, phi_rad, energy_keV):
    return EmCDSEventDataInSCFrameFromArrays(energy_keV, lon_rad, lat_rad, phi_rad)


def _make_irf_file(path, seed=0):
    """Physical toy IRF whose Epsilon profile changes with Ei, plus a separate AEFF."""

    rng = np.random.default_rng(seed)
    axes = Axes([
        HealpixAxis(nside=1, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
        Axis(np.geomspace(50, 2000, 7) * u.keV, label='Ei', scale='log'),
        Axis([-1, -0.5, -0.2, -0.05, -0.01, 0.01, 0.05, 0.2], label='Epsilon'),
        Axis(np.linspace(0, 180, 7) * u.deg, label='Phi'),
        Axis([-180, -40, -10, 0, 10, 40, 180] * u.deg, label='Theta'),
        PolarizationAxis(np.linspace(0, 360, 5) * u.deg, convention=StereographicConvention(), label='Zeta'),
    ])
    contents = rng.random(axes.nbins) * np.linspace(1, 3, axes['Ei'].nbins)[None, :, None, None, None, None] ** \
        np.arange(axes['Epsilon'].nbins)[None, None, :, None, None, None]
    phi = axes['Phi'].centers.to_value(u.rad)[:, None]
    theta = axes['Theta'].centers.to_value(u.rad)[None, :]
    contents *= ((phi + theta > 0) & (phi + theta < np.pi))[None, None, None, :, :, None]
    Histogram(axes, contents=contents, unit=u.cm ** 2).write(str(path), name='IRF', overwrite=True)

    aeff_axes = Axes([HealpixAxis(nside=2, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
                      Axis(np.geomspace(50, 2000, 15) * u.keV, label='Ei', scale='log')])
    aeff = 50 + 100 * rng.random(aeff_axes.nbins)
    Histogram(aeff_axes, contents=aeff, unit=u.cm ** 2).write(str(path), name='AEFF', overwrite=True)


def _direction(lon, lat):
    return UnitSphericalRepresentation(lon=np.atleast_1d(lon) * u.rad, lat=np.atleast_1d(lat) * u.rad)


class TestHats:

    def test_partition_of_unity_inside_the_range(self):
        nodes = np.geomspace(100, 1000, 5)
        energy = np.geomspace(100, 1000, 101)
        assert np.allclose(hat_functions(nodes, energy).sum(axis=1), 1)
        assert np.all(hat_functions(nodes, [90, 1100]) == 0)

    def test_hat_integrals(self):
        nodes = np.geomspace(100, 1000, 5)
        energy = np.geomspace(100, 1000, 200001)
        numeric = np.trapezoid(hat_functions(nodes, energy), energy, axis=0)
        assert np.allclose(hat_integrals_keV(nodes), numeric, rtol=1e-8)
        assert hat_integrals_keV(nodes).sum() == pytest.approx(900)

    def test_epsilon_edges_cover_the_broadened_range(self):
        log_step = np.log(10) / 6
        edges = broadened_epsilon_edges(log_step, 0.2)
        assert edges[0] == -1 and np.all(np.diff(edges) > 0)
        assert edges[-1] >= 1.2 * np.exp(log_step) - 1


@pytest.fixture(scope='module')
def path(tmp_path_factory):
    path = tmp_path_factory.mktemp('broadened') / 'irf.h5'
    _make_irf_file(path)
    write_broadened_irf(path, nodes_per_decade=4, energy_range_keV=(100., 1000.))
    return path


class TestWriteBroadened:

    def test_refuses_to_overwrite(self, path):
        with pytest.raises(ValueError):
            write_broadened_irf(path, nodes_per_decade=4, energy_range_keV=(100., 1000.))

    def test_loads_as_irf_with_the_nodes_as_ei_centers(self, path):
        irf = IRFRelativeHistUnpolarized.from_h5(path, broadened=True)
        assert np.allclose(irf.axes['Ei'].centers, np.geomspace(100, 1000, 5))
        assert irf._tot_aeff.axes['NuLambda'].nside == 2

    def test_broadened_area_is_hat_weighted_average(self, path):
        """At the centers of the AEFF pixels, A_n = int h_n (E/E_n)^-2 Aeff dE / Delta_n."""

        orig = IRFRelativeHistUnpolarized.from_h5(path)
        irf = IRFRelativeHistUnpolarized.from_h5(path, broadened=True)
        nodes = np.asarray(irf.axes['Ei'].centers)
        energy = np.geomspace(100, 1000, 20001)
        hats = hat_functions(nodes, energy)
        lon, lat = (np.deg2rad(np.asarray(a)) for a in irf._tot_aeff.axes['NuLambda'].pix2ang(np.array([7]), lonlat=True))

        aeff = np.asarray(orig._effective_area_cm2_unselected(_Photons(np.full(energy.size, lon[0]),
                                                                       np.full(energy.size, lat[0]), energy)))
        expected = (np.trapezoid(hats * (energy[:, None] / nodes[None, :]) ** -2 * aeff[:, None], energy, axis=0)
                    / hat_integrals_keV(nodes))
        broadened = irf._effective_area_cm2_unselected(_Photons(np.full(nodes.size, lon[0]),
                                                                np.full(nodes.size, lat[0]), nodes))
        assert np.allclose(broadened, expected, rtol=1e-4)

    def test_reproduces_the_folding_of_the_weight_spectrum(self, path):
        """For F = E^-2 (the weight), the hat interpolation of F/w is exact, so at the
        broadened table's own grid points sum_n Delta_n F(E_n) dAeff_n(x) must equal the
        integral of F Aeff p(x|E) over the nodes' range with the original IRF. They
        differ only because, when it is loaded, the broadened table is normalized so
        that its interpolation integrates exactly to the broadened area, which changes
        the values at the centers by the integration error of the interpolation."""

        orig = IRFRelativeHistUnpolarized.from_h5(path)
        irf = IRFRelativeHistUnpolarized.from_h5(path, broadened=True)
        nodes = np.asarray(irf.axes['Ei'].centers)
        delta = hat_integrals_keV(nodes)

        nulambda = irf.axes['NuLambda']
        lon_c, lat_c = (np.deg2rad(np.asarray(a)) for a in nulambda.pix2ang(np.array([3]), lonlat=True))
        aeff_lon, aeff_lat = lon_c[0], lat_c[0]
        phi = np.asarray(irf.axes['Phi'].centers)[2]
        theta = np.asarray(irf.axes['Theta'].centers)[3]
        zeta = np.asarray(irf.axes['Zeta'].centers)[1]

        from cosipy.response.relative_coordinates import RelativeCDSCoordinates
        src = _direction(aeff_lon, aeff_lat).to_cartesian().xyz
        vec = RelativeCDSCoordinates.to_cds(RelativeCDSCoordinates(src, irf._pol_convention),
                                            np.array([phi + theta]), np.array([zeta]))
        slon, slat = np.arctan2(vec[1], vec[0]), np.arcsin(np.clip(vec[2], -1, 1))

        for n in [1, 2, 3]:
            for eps_c in np.asarray(irf.axes['Epsilon'].centers)[[5, 9, 12]]:
                em = nodes[n] * (1 + eps_c)
                energy = np.geomspace(100, 1000, 200001)
                m = energy.size
                ev = _Events(np.full(m, slon[0]), np.full(m, slat[0]), np.full(m, phi), np.full(m, em))
                ph = _Photons(np.full(m, aeff_lon), np.full(m, aeff_lat), energy)
                integrand = (energy ** -2.) * np.asarray(orig.differential_effective_area_cm2(ph, ev))
                expected = np.trapezoid(integrand, energy)

                ph_n = _Photons(np.full(nodes.size, aeff_lon), np.full(nodes.size, aeff_lat), nodes)
                ev_n = _Events(np.full(nodes.size, slon[0]), np.full(nodes.size, slat[0]), np.full(nodes.size, phi),
                               np.full(nodes.size, em))
                broadened = np.asarray(irf.differential_effective_area_cm2(ph_n, ev_n))
                total = np.sum(delta * nodes ** -2. * broadened)

                assert total == pytest.approx(expected, rel=1.5e-2)

    def test_selected_area_is_integral_of_the_density_over_the_cut(self, path):
        """The broadened IRF is a regular IRF: at each node its density integrates over
        the selected Em to the selected area."""

        sel = EnergySelector([180, 700] * u.keV)
        irf = IRFRelativeHistUnpolarized.from_h5(path, broadened=True, selections=sel)
        unsel = IRFRelativeHistUnpolarized.from_h5(path, broadened=True)
        nodes = np.asarray(irf.axes['Ei'].centers)
        ph = _Photons(np.full(nodes.size, 0.3), np.full(nodes.size, 0.4), nodes)
        fraction = np.asarray(irf._effective_area_cm2(ph)) / np.asarray(unsel._effective_area_cm2(ph))
        assert np.all((fraction > 0) & (fraction < 1))
