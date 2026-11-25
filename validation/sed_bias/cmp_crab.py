import sys, numpy as np
sys.path.insert(0, sys.argv[1] + '/scripts')
from predict_em import predict
from astropy.io import fits
S = sys.argv[1]
X = np.load(f'{S}/out/exposure.npz')
d = fits.open(f'{S}/data/crab_powerlaw_3months_unbinned_data_filtered_with_SAAcut.fits.gz')[1].data
E, D = d['Energies'], d['Distance']
K = 4.94717171
flux = lambda e: K * e ** -2.
edges = np.array([100, 120, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000.])
for name, sel in (('nn', np.ones_like(E, bool)), ('simple', D >= 1.0)):
    obs, _ = np.histogram(E[sel], edges)
    for occ in (0, 1):
        N, tot = predict(name, X[f'crab3m_n32_occ{occ}'], flux, edges)
        print(f'== {name} occ={occ}: observed {obs.sum()}  predicted(in Em 100-10000) {N.sum():.0f}  predicted all Em {tot:.0f}  obs/pred={obs.sum()/N.sum():.4f}')
        r = obs / N
        for a, b, o, n, rr in zip(edges[:-1], edges[1:], obs, N, r):
            print(f'  {a:6.0f}-{b:6.0f}  obs {o:8d}  pred {n:10.1f}  obs/pred {rr:.3f} +- {rr/np.sqrt(max(o,1)):.3f}   shape {rr/(obs.sum()/N.sum()):.3f}')
