import sys, numpy as np, hdf5plugin, h5py
S = sys.argv[1]
np.set_printoptions(linewidth=220, precision=3, suppress=True)
with h5py.File(f'{S}/data/binned_continuum.h5') as f:
    eff = f['DRM/EFF_AREA'][:]; ei = f['DRM/AXES/1'][:]
    R = np.zeros((768, 10, 10))
    for i in range(0, 768, 32): R[i:i+32] = f['DRM/COUNTS'][i:i+32].astype(float).sum(axis=(3, 4))
R *= eff[None, :, None]
T8 = np.load(f'{S}/out/exposure.npz')['crab3m_n8_occ0']
B = np.einsum('p,pie->ie', T8, R) / T8.sum()          # mean area [Ei bin, Em bin]
out = {'binned': B}
for name in ('nn', 'simple'):
    Z = np.load(f'{S}/out/M_{name}_crab3m.npz'); M = Z['M'] / T8.sum(); E = Z['E']; Ec = np.sqrt(E[1:] * E[:-1]); dE = np.diff(E)
    M10 = M[0::2] + M[1::2]                            # 10 Em bins
    H = np.zeros((10, 10))
    for i in range(10):
        m = (Ec >= ei[i]) & (Ec < ei[i + 1])
        H[i] = (M10[:, m] * dE[m]).sum(1) / dE[m].sum()   # flat-in-E average, like the binned sim
    out[name] = H
print('Ei bins', ei.round(0))
print('total area with Em in 100-10000 keV (cm2), per Ei bin')
for k, v in out.items(): print(f'{k:7s}', v.sum(1))
print('ratio to binned'); [print(f'{k:7s}', out[k].sum(1) / B.sum(1)) for k in ('nn', 'simple')]
print('photopeak fraction (Em bin == Ei bin)')
for k, v in out.items(): print(f'{k:7s}', np.diag(v) / v.sum(1))
