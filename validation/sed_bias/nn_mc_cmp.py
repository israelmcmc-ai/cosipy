import sys, glob, numpy as np, hdf5plugin, h5py
from astropy.io import fits
from scipy.optimize import minimize
S = sys.argv[1]
fs = sorted(glob.glob(f'{S}/out/nnlog_mc_*.npz'))
Z = [np.load(f) for f in fs]
E = np.concatenate([z['E'] for z in Z]); A = np.concatenate([z['A'] for z in Z]); Em = np.concatenate([z['Em'] for z in Z])
N = len(E); norm = np.concatenate([z['norm'] * len(z['E']) for z in Z]) / N
print('files', len(fs), 'samples', N)
em = np.geomspace(100, 10000, 11)
d = fits.open(f'{S}/data/crab_powerlaw_3months_unbinned_data_filtered_with_SAAcut.fits.gz')[1].data
obs, _ = np.histogram(d['Energies'], em)
pred = np.histogram(Em, em, weights=A * norm)[0]; err = np.sqrt(np.histogram(Em, em, weights=(A * norm) ** 2)[0])
for k in range(10): print(f'{em[k]:6.0f}-{em[k+1]:6.0f}  obs {obs[k]:8d}  NN pred {pred[k]:10.0f} +- {err[k]:7.0f}  obs/pred {obs[k]/pred[k]:.3f} +- {obs[k]/pred[k]*err[k]/pred[k]:.3f}')
# fit index by reweighting the MC (sampled from E^-2): w = (E/1)^(g+2) * K/K0
def nll(x):
    w = A * norm * np.exp(x[0]) * E ** x[1]
    mu = np.histogram(Em, em, weights=w)[0]; return np.sum(mu - obs * np.log(mu))
r = minimize(nll, [0, 0], method='Nelder-Mead'); print('NN Em-only fit: index', -2 + r.x[1], 'K/K_inj', np.exp(r.x[0]))
# MC uncertainty: the spread of the fits of each file
if len(Z) > 1:
    gs = []
    for z in Z:
        Ez, Az, Emz, nz = z['E'], z['A'], z['Em'], z['norm']
        f = lambda x: np.sum(np.histogram(Emz, em, weights=Az * nz * np.exp(x[0]) * Ez ** x[1])[0]
                             - obs * np.log(np.histogram(Emz, em, weights=Az * nz * np.exp(x[0]) * Ez ** x[1])[0]))
        gs.append(-2 + minimize(f, [0, 0], method='Nelder-Mead').x[1])
    print('per-file indices', np.round(gs, 3), 'MC error of the combined fit ~', np.std(gs, ddof=1) / np.sqrt(len(gs)))
