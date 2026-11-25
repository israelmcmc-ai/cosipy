# Matrix M[k, e]: exposure-weighted selected area (cm2 s) of Em bin k for photon energy Ec[e], real IRF code.
import sys, numpy as np, healpy as hp, astropy.units as u, warnings
warnings.filterwarnings('ignore')
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.event_selection.energy_selection import EnergySelector
S, name, expkey, out = sys.argv[1:5]
em = np.geomspace(100, 10000, 21)
irf = IRFRelativeHistUnpolarized.from_h5(f'{S}/out/reduced_{name}.h5', selections=EnergySelector([100, 200] * u.keV))
T = np.load(f'{S}/out/exposure.npz')[expkey]; nside = hp.npix2nside(len(T))
pix = np.nonzero(T)[0]; th, ph = hp.pix2ang(nside, pix)
E = np.geomspace(100, 10000, 801); Ec = np.sqrt(E[1:] * E[:-1])
photons = PhotonListWithDirectionAndEnergyInSCFrame(np.repeat(ph, len(Ec)), np.repeat(np.pi / 2 - th, len(Ec)), np.tile(Ec, len(pix)))
M = np.zeros((len(em) - 1, len(Ec)))
for k, (lo, hi) in enumerate(zip(em[:-1], em[1:])):
    irf._selection_ranges = np.array([[lo, hi]])
    M[k] = (T[pix][:, None] * np.asarray(irf._effective_area_cm2(photons)).reshape(len(pix), len(Ec))).sum(0)
np.savez(out, M=M, E=E, em=em)
print('ok', out)
