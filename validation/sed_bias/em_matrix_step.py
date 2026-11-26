# Em matrix like em_matrix.py, but each Epsilon bin's stored probability is spread uniformly within the bin
# (Ei: log interpolation between centers of the normalized probabilities; total area: AEFF interpolated in log Ei).
import sys, numpy as np, healpy as hp
S, name, expkey, out = sys.argv[1:5]
P = np.load(f'{S}/out/proj_{name}.npz')
T32 = np.load(f'{S}/out/exposure.npz')[expkey]
ei, eps = P['ei'], P['eps']; p = P['p_eps']
tot = p.sum(-1, keepdims=True); prob = np.divide(p, tot, out=np.zeros_like(p), where=tot > 0)
E = np.geomspace(100, 10000, 801); Ec = np.sqrt(E[1:] * E[:-1])
a_c = np.sqrt(P['aeff_ei'][1:] * P['aeff_ei'][:-1])
th, ph = hp.pix2ang(32, np.arange(12 * 32 ** 2)); parent = hp.ang2pix(8, th, ph)
W8 = np.zeros((768, P['aeff'].shape[1])); np.add.at(W8, parent, T32[:, None] * P['aeff'])
W8f = np.array([np.interp(np.log(Ec), np.log(a_c), w) for w in W8])
c = np.sqrt(ei[1:] * ei[:-1]); lx = np.log(Ec)
j = np.clip(np.searchsorted(np.log(c), lx) - 1, 0, len(c) - 2)
w = np.clip((lx - np.log(c[j])) / (np.log(c[j + 1]) - np.log(c[j])), 0, 1)
PE = (W8f[:, :, None] * ((1 - w)[None, :, None] * prob[:, j] + w[None, :, None] * prob[:, j + 1])).sum(0)
em = np.geomspace(100, 10000, 21)
M = np.zeros((20, len(Ec)))
for k in range(len(eps) - 1):
    lo, hi = Ec * (1 + eps[k]), Ec * (1 + eps[k + 1])
    for m in range(20):
        M[m] += PE[:, k] * np.clip(np.minimum(hi, em[m + 1]) - np.maximum(lo, em[m]), 0, None) / (hi - lo)
np.savez(out, M=M, E=E, em=em); print('ok', out)
