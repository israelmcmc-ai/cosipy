import sys, warnings
import numpy as np, healpy as hp
import astropy.units as u
from astropy.time import Time
warnings.filterwarnings('ignore', category=RuntimeWarning)
from astropy.coordinates import UnitSphericalRepresentation
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.background_estimation.free_norm_hist_background import HistBackgroundTemplate, FreeNormHistBackgroundDensity as F, _unitless
from cosipy.event_selection.energy_selection import EnergySelector

d = '/home/user/dc4data/'
t = HistBackgroundTemplate.open(d + sys.argv[1]).smooth(psichi_fwhm=4 * u.deg, phi_fwhm=20 * u.deg, time_counts=500)
ax = t.psichi_counts.axes
print(flush=True); print('nside', ax['PsiChi'].nside, 'Em', _unitless(ax['Em'], u.keV).edges[[0, -1]], len(ax['Em']), 'Phi bins', len(ax['Phi']),
      'phi rate Em', len(t.phi_counts.axes['Em']), 'rate Em', len(t.rate_counts.axes['Em']))
fine = int(sys.argv[2]) if len(sys.argv) > 2 else 4
nside = ax['PsiChi'].nside * fine
theta, lon = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
from astropy.coordinates import SkyCoord
from scoords import SpacecraftFrame
psichi_fine = SkyCoord(lon=lon * u.rad, lat=(np.pi / 2 - theta) * u.rad, frame=SpacecraftFrame())
dOmega = hp.nside2pixarea(nside)

em = np.geomspace(200, 5000, 9)
phi = np.linspace(0, np.pi, 121)[:-1] + np.pi / 240   # midpoints
res = {}
for k, rock in enumerate([-22., 22.]):
    Ipsi = np.empty((em.size, phi.size))
    for i, e in enumerate(em):
        for j, p in enumerate(phi):
            n = lon.size
            Ipsi[i, j] = F._psichi_density(t, np.full(n, rock), np.full(n, e), np.full(n, p), psichi_fine).sum() * dOmega
    Iphi = np.array([np.sum(F._phi_density(t, np.full(phi.size, rock), np.full(phi.size, e), phi) * Ipsi[i]) * np.pi / phi.size
                     for i, e in enumerate(em)])
    phinorm = np.array([np.sum(F._phi_density(t, np.full(phi.size, rock), np.full(phi.size, e), phi)) * np.pi / phi.size for e in em])
    print('rock', rock)
    print(' psichi integral range', Ipsi.min(), Ipsi.max(), 'mean', Ipsi.mean())
    print(' phi pdf integral range', phinorm.min(), phinorm.max())
    print(' combined', Iphi.min(), Iphi.max())
