# Expected counts per Em bin at the injected spectrum, with the real IRFRelativeHistUnpolarized code
# (reduced IRF: same Ei/Epsilon behavior), integrated over the exposure map.
import sys, numpy as np, healpy as hp, astropy.units as u, warnings
warnings.filterwarnings('ignore')
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.event_selection.energy_selection import EnergySelector
S, name, expkey, specname = sys.argv[1:5]
edges = np.array([float(x) for x in sys.argv[5].split(',')])
irf = IRFRelativeHistUnpolarized.from_h5(f'{S}/out/reduced_{name}.h5', selections=EnergySelector([100, 200] * u.keV))
T = np.load(f'{S}/out/exposure.npz')[expkey]; nside = hp.npix2nside(len(T))
pix = np.nonzero(T)[0]; th, ph = hp.pix2ang(nside, pix)
E = np.geomspace(100, 10000, 801); Ec = np.sqrt(E[1:] * E[:-1]); dE = np.diff(E)
if specname == 'crab':
    F = 4.94717171 * Ec ** -2.
else:  # GRB Band, astromodels convention: K at piv=100 keV, xp=Ep
    from astromodels import Band
    b = Band(K=5.2385, alpha=-1.022, beta=-2.763, xp=160.0, piv=100.)
    F = b(Ec)
lon = np.repeat(ph, len(Ec)); lat = np.repeat(np.pi / 2 - th, len(Ec)); en = np.tile(Ec, len(pix))
photons = PhotonListWithDirectionAndEnergyInSCFrame(lon, lat, en)
w = (T[pix][:, None] * (F * dE)[None, :]).ravel()
irf._selection_ranges = None
Atot = np.asarray(irf._effective_area_cm2(photons))
print('total', np.sum(w * Atot))
out = []
for lo, hi in zip(edges[:-1], edges[1:]):
    irf._selection_ranges = np.array([[lo, hi]])
    out.append(np.sum(w * np.asarray(irf._effective_area_cm2(photons))))
print('EM', ' '.join(f'{x:.2f}' for x in out))
