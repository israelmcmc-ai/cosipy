import sys, numpy as np, hdf5plugin, h5py
from astropy.io import fits
from astropy.time import Time
from scipy.optimize import minimize
S = sys.argv[1]
d = fits.open(f'{S}/data/crab_powerlaw_3months_unbinned_data_filtered_with_SAAcut.fits.gz')[1].data
t0, t1 = Time('2028-03-23 10:36:00').unix, Time('2028-03-23 13:36:00').unix
win3h = (d['TimeTags'] >= t0) & (d['TimeTags'] < t1)
print('3h events', win3h.sum(), 'with D>=1', (win3h & (d['Distance'] >= 1)).sum())

def fit(pred, obs):
    def nll(x):
        mu = pred(np.exp(x[0]), x[1]); return np.sum(mu - obs * np.log(mu))
    r = minimize(nll, [np.log(4.947), -2.0], method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-6))
    # numerical hessian for index error
    h = 1e-3; x = r.x
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            e_i = np.eye(2)[i] * h; e_j = np.eye(2)[j] * h
            H[i, j] = (nll(x + e_i + e_j) - nll(x + e_i - e_j) - nll(x - e_i + e_j) + nll(x - e_i - e_j)) / (4 * h * h)
    return x, np.sqrt(np.diag(np.linalg.inv(H)))

for name in ('nn', 'simple'):
    for w in ('crab3h', 'crab3m'):
        Z = np.load(f'{S}/out/M_{name}_{w}.npz'); M, E, em = Z['M'], Z['E'], Z['em']
        Ec = np.sqrt(E[1:] * E[:-1]); dE = np.diff(E)
        sel = (win3h if w == 'crab3h' else np.ones(len(d), bool)) & ((d['Distance'] >= 1) if name == 'simple' else True)
        obs, _ = np.histogram(d['Energies'][sel], em)
        pred = lambda K, g: M @ (K * Ec ** g * dE)
        x, e = fit(pred, obs)
        mu0 = pred(4.94717, -2.0)
        print(f'hist_{name:6s} {w}: index {x[1]:.3f} +- {e[1]:.3f}, K/K_inj {np.exp(x[0])/4.94717:.3f}; obs {obs.sum()} pred@inj {mu0.sum():.0f}')

with h5py.File(f'{S}/data/binned_continuum.h5') as f:
    eff = f['DRM/EFF_AREA'][:]; ei = f['DRM/AXES/1'][:]; em = f['DRM/AXES/2'][:]
    R = np.zeros((768, 10, 10))
    for i in range(0, 768, 32): R[i:i+32] = f['DRM/COUNTS'][i:i+32].astype(float).sum(axis=(3, 4))
R *= eff[None, :, None]
X = np.load(f'{S}/out/exposure.npz')
for w in ('crab3h', 'crab3m'):
    Rw = np.einsum('p,pie->ie', X[f'{w}_n8_occ0'], R)
    sel = win3h if w == 'crab3h' else np.ones(len(d), bool)
    obs, _ = np.histogram(d['Energies'][sel], em)
    def pred(K, g):
        Fb = K * (ei[1:] ** (g + 1) - ei[:-1] ** (g + 1)) / (g + 1)
        return Fb @ Rw
    x, e = fit(pred, obs)
    print(f'binned DC4 m2284 {w}: index {x[1]:.3f} +- {e[1]:.3f}, K/K_inj {np.exp(x[0])/4.94717:.3f}; obs {obs.sum()} pred@inj {pred(4.94717,-2).sum():.0f}')
