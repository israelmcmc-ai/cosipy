import sys, numpy as np, hdf5plugin, h5py, warnings
warnings.filterwarnings('ignore')
from astropy.io import fits
from astropy.time import Time
from scipy.optimize import minimize
from astromodels import Band
S = sys.argv[1]
d = fits.open(f'{S}/data/GRB_bn090424592_3months_unbinned_data_filtered_with_SAAcut.fits.gz')[1].data
t0, t1 = Time('2028-03-24 10:36:42').unix, Time('2028-03-24 10:36:51.921').unix
win = (d['TimeTags'] >= t0) & (d['TimeTags'] <= t1)
print('window events', win.sum(), 'D>=1', (win & (d['Distance'] >= 1)).sum())
band = Band(K=5.2385, alpha=-1.022, beta=-2.763, xp=160.0, piv=100.)
band.alpha.bounds = (-1.5, 3); band.beta.bounds = (-5, -2.0)
inj = np.array([np.log(5.2385), -1.022, -2.763, np.log(160.)])
def F(x, E):
    band.K.value, band.alpha.value, band.beta.value, band.xp.value = np.exp(x[0]), x[1], x[2], np.exp(x[3])
    return band(E)
def fit(pred, obs):
    def nll(x):
        if not (-1.5 <= x[1] <= 3 and -5 <= x[2] <= -2.0): return 1e30
        mu = pred(x); return np.sum(mu - obs * np.log(mu))
    best = None
    for start in (inj, inj + [0, 0.5, 0, 0.2], inj + [0, -0.3, 0, -0.2]):
        r = minimize(nll, start, method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-6, maxiter=20000))
        if best is None or r.fun < best.fun: best = r
    return best.x
def report(tag, pred, obs):
    x = fit(pred, obs); mu0 = pred(inj)
    print(f'{tag}: alpha {x[1]:+.3f} beta {x[2]:.3f} Ep {np.exp(x[3]):.1f} K {np.exp(x[0]):.3f}  | obs/pred@inj per Em bin: ' +
          ' '.join(f'{o/m:.2f}' for o, m in zip(obs, mu0)))
for name in ('simple', 'nn'):
    Z = np.load(f'{S}/out/M_{name}_grb.npz'); M, E, em = Z['M'], Z['E'], Z['em']
    Ec = np.sqrt(E[1:] * E[:-1]); dE = np.diff(E)
    sel = win & ((d['Distance'] >= 1) if name == 'simple' else True)
    obs, _ = np.histogram(d['Energies'][sel], em)
    print('Em edges', em[:9].round(0))
    report(f'hist_{name}', lambda x: M @ (F(x, Ec) * dE), obs)
with h5py.File(f'{S}/data/binned_continuum.h5') as f:
    eff = f['DRM/EFF_AREA'][:]; ei = f['DRM/AXES/1'][:]; em = f['DRM/AXES/2'][:]
    R = np.zeros((768, 10, 10))
    for i in range(0, 768, 32): R[i:i+32] = f['DRM/COUNTS'][i:i+32].astype(float).sum(axis=(3, 4))
R *= eff[None, :, None]
Rw = np.einsum('p,pie->ie', np.load(f'{S}/out/exposure.npz')['grb_n8_occ0'], R)
obs, _ = np.histogram(d['Energies'][win], em)
Ef = np.geomspace(100, 10000, 2001); Efc = np.sqrt(Ef[1:] * Ef[:-1]); dEf = np.diff(Ef)
idx = np.searchsorted(ei, Efc) - 1
report('binned m2284', lambda x: np.bincount(idx, F(x, Efc) * dEf, minlength=10) @ Rw, obs)
