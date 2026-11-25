# Project the 6D IRF contents onto [NuLambda, Ei, Epsilon] and [NuLambda, Ei, Phi] (sums of per-bin area).
import sys, h5py, numpy as np
fn, out = sys.argv[1], sys.argv[2]
with h5py.File(fn) as f:
    c = f['IRF/contents']
    n = c.shape[0]
    p_eps = np.zeros(c.shape[:3]); p_phi = np.zeros(c.shape[:2] + (c.shape[3],))
    for i in range(0, n, 16):
        blk = np.nan_to_num(c[i:i+16].astype(float))
        p_eps[i:i+16] = blk.sum(axis=(3, 4, 5))
        p_phi[i:i+16] = blk.sum(axis=(2, 4, 5))
    np.savez(out, p_eps=p_eps, p_phi=p_phi,
             ei=f['IRF/axes/1'][:], eps=f['IRF/axes/2'][:], phi=f['IRF/axes/3'][:],
             aeff=f['AEFF/contents'][:], aeff_ei=f['AEFF/axes/1'][:])
print('ok', out)
