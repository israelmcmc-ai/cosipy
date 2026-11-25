import sys, numpy as np, hdf5plugin, h5py
sys.path.insert(0, sys.argv[1] + '/scripts')
from predict_em import predict
from astropy.io import fits
S = sys.argv[1]
X = np.load(f'{S}/out/exposure.npz')
T8 = X['crab3m_n8_occ0']
with h5py.File(f'{S}/data/binned_continuum.h5') as f:
    eff = f['DRM/EFF_AREA'][:]
    ei = f['DRM/AXES/1'][:]; em = f['DRM/AXES/2'][:]
    R = np.zeros((768, 10, 10))
    for i in range(0, 768, 32):
        R[i:i+32] = f['DRM/COUNTS'][i:i+32].astype(float).sum(axis=(3, 4))
    print(f['DRM/AXES/0'].attrs.items() if f['DRM/AXES/0'].attrs else '', list(f['DRM/AXES/0'].attrs.keys()))
R *= eff[None, :, None]                       # cm2 per (pix, Ei, Em)
K = 4.94717171; flux = lambda e: K * e ** -2.
Fbin = K * (1 / ei[:-1] - 1 / ei[1:])           # photons/cm2/s per Ei bin
Nb = np.einsum('p,pie,i->e', T8, R, Fbin)
d = fits.open(f'{S}/data/crab_powerlaw_3months_unbinned_data_filtered_with_SAAcut.fits.gz')[1].data
obs, _ = np.histogram(d['Energies'], em)
Nn, _ = predict('nn', X['crab3m_n32_occ0'], flux, em)
D = d['Distance']; obs_s, _ = np.histogram(d['Energies'][D >= 1], em)
Ns, _ = predict('simple', X['crab3m_n32_occ0'], flux, em)
print('total obs', obs.sum(), 'binned pred', Nb.sum(), 'nn pred', Nn.sum())
print(f'{"Em bin":>14} {"obs":>9} {"obs/binned":>10} {"obs/hist_nn":>11} {"obs_d1/hist_simple":>18}')
for k in range(10):
    print(f'{em[k]:6.0f}-{em[k+1]:6.0f} {obs[k]:9d} {obs[k]/Nb[k]:10.3f} {obs[k]/Nn[k]:11.3f} {obs_s[k]/Ns[k]:18.3f}')
# within-Ei-bin weighting check: what sim spectrum? EFF_AREA ratios
print('EFF_AREA ratios', eff[:-1] / eff[1:])
