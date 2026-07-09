from typing import Iterable, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import spherical_to_cartesian, UnitSphericalRepresentation
from astropy.io.fits import update
from astropy.units import Quantity
from histpy import Histogram, HealpixAxis, Axis
from mhealpy.plot.axes import HealpyAxes

from cosipy.interfaces import EventDataInterface
from cosipy.interfaces.data_interface import EmCDSEventDataInSCFrameInterface
from cosipy.interfaces.event import EmCDSEventInSCFrameInterface
from cosipy.interfaces.instrument_response_interface import FarFieldSpectralInstrumentResponseFunctionInterface
from cosipy.interfaces.photon_parameters import PhotonWithDirectionAndEnergyInSCFrameInterface, PhotonListInterface, \
    PhotonListWithDirectionInSCFrameInterface, PhotonListWithDirectionAndEnergyInSCFrameInterface

import h5py as h5

from cosipy.polarization import PolarizationAxis
from cosipy.response.relative_coordinates import RelativeCDSCoordinates
from cosipy.util.iterables import asarray


class IRFRelativeHistUnpolarized(FarFieldSpectralInstrumentResponseFunctionInterface):

    event_data_type = EmCDSEventDataInSCFrameInterface
    photon_list_type = PhotonListWithDirectionAndEnergyInSCFrameInterface

    def __init__(self,
                 irf: Histogram,
                 copy = True,
                 batch_size=100000):

        if copy:
            irf = irf.copy()

        # Checks
        if not irf.unit.is_equivalent('cm^2'):
            raise ValueError("IRF contents are expected to have units of area.")

        axes = irf.axes

        if not np.array_equal(axes.labels, ['NuLambda', 'Ei', 'Epsilon', 'Phi', 'Theta', 'Zeta']):
            raise ValueError("IRF axes label must be ['NuLambda', 'Ei', 'Epsilon', 'Phi', 'Theta', 'Zeta']")

        if not isinstance(axes['NuLambda'], HealpixAxis):
            raise ValueError("IRF NuLambda axis is expected to be of HealpixAxis type")

        if not axes['Ei'].unit.is_equivalent('keV'):
            raise ValueError("Ei axis is expected to have units of energy.")

        if not axes['Epsilon'].unit.is_equivalent(''):
            raise ValueError("Ei axis is expected to be unitless")

        if not axes['Phi'].unit.is_equivalent('deg'):
            raise ValueError("Phi axis is expected to have units of angle.")

        if not axes['Theta'].unit.is_equivalent('deg'):
            raise ValueError("Theta axis is expected to have units of angle.")

        if not isinstance(axes['Zeta'], PolarizationAxis):
            raise ValueError("IRF Zeta axis is expected to be of PolarizationAxis type")

        # Standardize units
        axes['Ei'] = axes['Ei'].to(u.keV, copy = False).to(None, update = False, copy = False)
        axes['Epsilon'] = axes['Epsilon'].to(None, update = False, copy = False)
        axes['Phi'] = axes['Phi'].to(u.rad, copy = False).to(None, update = False, copy = False)
        axes['Theta'] = axes['Theta'].to(u.rad, copy = False).to(None, update = False, copy = False)
        self._pol_convention = axes['Zeta'].convention
        axes['Zeta'] = Axis(axes['Zeta'].edges.angle.to(u.rad).value, label = 'Zeta')

        irf = irf.to(u.cm * u.cm, copy=False).to(None, copy=False, update=False) # To cm2 and remove units

        # Get the total effective area
        self._tot_aeff = irf.project('NuLambda','Ei') # cm^2

        # Phase space
        # Final content units will be cm^2/sr/rad/keV
        phi_edges_mesh, arm_edges_mesh, az_edges_mesh = np.meshgrid(axes['Phi'].edges,
                                                                    axes['Theta'].edges,
                                                                    axes['Zeta'].edges, indexing='ij')

        phase_space_cds = RelativeCDSCoordinates.get_relative_cds_phase_space(phi_edges_mesh[:-1, :-1, :-1],
                                                                              phi_edges_mesh[1:, :-1, :-1],
                                                                              arm_edges_mesh[:-1, :-1, :-1],
                                                                              arm_edges_mesh[:-1, 1:, :-1:],
                                                                              az_edges_mesh[:-1, :-1, :-1],
                                                                              az_edges_mesh[:-1, :-1, 1:])

        ei_centers_mesh, em_widths_mesh = np.meshgrid(axes['Ei'].centers,
                                                      axes['Epsilon'].widths,
                                                      indexing='ij')

        phase_space_em = ei_centers_mesh * em_widths_mesh

        irf /= axes.expand_dims(phase_space_cds, axes.label_to_index(['Phi', 'Theta', 'Zeta']))
        irf /= axes.expand_dims(phase_space_em, axes.label_to_index(['Ei', 'Epsilon']))

        self._diff_aeff = irf

        # Extra params
        self._batch_size = batch_size

    @classmethod
    def from_h5(cls, filename, *args, **kwargs):
        """

        Parameters
        ----------
        filename

        Returns
        -------

        """

        return cls(Histogram.open(filename, "IRF"), *args, **kwargs)

    def _effective_area_cm2(self, photons: PhotonListWithDirectionAndEnergyInSCFrameInterface) -> Iterable[float]:
        """

        Parameters
        ----------
        photons

        Returns
        -------

        """

        photon_dir, photon_energy_keV = self._photon_list_to_raw_values(photons)

        return self._tot_aeff.interp(photon_dir, photon_energy_keV)

    @staticmethod
    def _photon_list_to_raw_values(photons:PhotonListWithDirectionAndEnergyInSCFrameInterface):

        photon_lon_rad = asarray(photons.direction_lon_rad_sc, float)
        photon_lat_rad = asarray(photons.direction_lat_rad_sc, float)

        photon_dir = UnitSphericalRepresentation(lon=Quantity(photon_lon_rad, 'rad', copy=False),
                                                 lat=Quantity(photon_lat_rad, 'rad', copy=False))

        photon_energy_keV = asarray(photons.energy_keV, float)

        return photon_dir, photon_energy_keV

    def _differential_effective_area_cm2(self, photons:PhotonListWithDirectionAndEnergyInSCFrameInterface, events: EmCDSEventDataInSCFrameInterface) -> Iterable[float]:
        """

        Parameters
        ----------
        query

        Returns
        -------

        """

        photon_dir, photon_energy_keV = self._photon_list_to_raw_values(photons)

        psichi_lon_rad = asarray(events.scattered_lon_rad_sc, float)
        psichi_lat_rad = asarray(events.scattered_lat_rad_sc, float)

        psichi_dir = UnitSphericalRepresentation(lon = Quantity(psichi_lon_rad, 'rad', copy = False),
                                                 lat = Quantity(psichi_lat_rad, 'rad', copy = False))

        phi_kin_rad = asarray(events.scattering_angle_rad, float)
        measured_energy_keV = asarray(events.energy_keV, float)

        # Convert to relative coordinates
        epsilon = (measured_energy_keV - photon_energy_keV)/photon_energy_keV

        relcoords = RelativeCDSCoordinates(photon_dir.to_cartesian().xyz, pol_convention=self._pol_convention)
        phi_geo, zeta = relcoords.to_relative(psichi_dir.to_cartesian().xyz)

        phi_geo_rad = phi_geo.to_value(u.rad)
        zeta_rad = zeta.to_value(u.rad)

        theta_rad = phi_geo_rad - phi_kin_rad

        return self._diff_aeff.interp({'NuLambda': photon_dir,
                                       'Ei': photon_energy_keV,
                                       'Epsilon': epsilon,
                                       'Phi': phi_kin_rad,
                                       'Theta': theta_rad,
                                       'Zeta': zeta_rad})


    def _random_events(self, photons: PhotonListWithDirectionInSCFrameInterface) -> EventDataInterface:
        """
        """
        raise NotImplementedError("random_events not implemented yet.")

