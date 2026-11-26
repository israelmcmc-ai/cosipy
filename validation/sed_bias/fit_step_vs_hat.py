import sys, glob, numpy as np
from astropy.io import fits
from astropy.time import Time
from scipy.optimize import minimize
S = sys.argv[1]
d = fits.open(f'{S}/data/crab_powerlaw_3months_unbinned_data_filtered_with_SAAcut.fits.gz')[1].data
Emd, D, t = d['Energies'], d['Distance'], d['TimeTags']
w3h = (t >= Time('2028-03-23 10:36:00').unix) & (t < Time('2028-03-23 13:36:00').unix)
def fit(M, Ec, dE, obs, sl):
    def nll(x):
        mu = (M @ (np.exp(x[0]) * Ec ** x[1] * dE))[sl]; return np.sum(mu - obs[sl] * np.log(mu))
    return minimize(nll, [np.log(4.947), -2.], method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-7)).x[1]
ranges = {'100 keV-10 MeV': slice(0, 20), '200 keV-5 MeV': slice(3, 17)}
for name in ('nn', 'simple'):
    sel = (D >= 1) if name == 'simple' else np.ones(len(D), bool)
    for kind in ('M', 'Mstep'):
        Z = np.load(f'{S}/out/{kind}_{name}_crab3m.npz'); E = Z['E']; Ec = np.sqrt(E[1:] * E[:-1])
        obs, _ = np.histogram(Emd[sel], Z['em'])
        print(f'hist_{name:6s} {"hats (#31)" if kind == "M" else "bin-uniform":12s}', '  '.join(f'{r}: {fit(Z["M"], Ec, np.diff(E), obs, s):.3f}' for r, s in ranges.items()))
# NN MC: 3 months and the 3 h window (reweighting the 3-month samples by the exposure ratio per pixel)
import healpy as hp
X = np.load(f'{S}/out/exposure.npz'); T3m, T3h = X['crab3m_n32_occ0'], X['crab3h_n32_occ0']
fs = sorted(glob.glob(f'{S}/out/nnlog_mc_*.npz')); Ztot = sum(len(np.load(f)['E']) for f in fs)
E, A, Em, nrm, pix = [], [], [], [], []
for f in fs:
    z = np.load(f); seed = int(f.split('_')[-1].split('.')[0]); n = len(z['E'])
    rng = np.random.default_rng(seed); pix.append(rng.choice(len(T3m), n, p=T3m / T3m.sum()))
    E.append(z['E']); A.append(z['A']); Em.append(z['Em']); nrm.append(z['norm'] * n / Ztot)
E, A, Em, nrm, pix = map(np.concatenate, (E, A, Em, nrm, pix))
em = np.geomspace(100, 10000, 21)
def mcfit(w0, obs, sl):
    def nll(x):
        mu = np.histogram(Em, em, weights=w0 * np.exp(x[0]) * E ** x[1])[0][sl]; return np.sum(mu - obs[sl] * np.log(mu))
    return -2 + minimize(nll, [0, 0], method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-7)).x[1]
for tag, wexp, sel in (('3 months', np.ones_like(E), np.ones(len(Emd), bool)), ('3 h window', T3h[pix] / T3m[pix], w3h)):
    obs, _ = np.histogram(Emd[sel], em)
    print(f'NN MC {tag:10s}', '  '.join(f'{r}: {mcfit(A * nrm * wexp, obs, s):.3f}' for r, s in ranges.items()),
          f' (pred/obs {np.sum(A*nrm*wexp*((Em>=100)&(Em<1e4)))/obs.sum():.3f})')
