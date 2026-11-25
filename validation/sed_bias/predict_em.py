# Predicted Em spectrum of a point source at the injected spectrum, straight from the response tables.
import sys, numpy as np, healpy as hp
S = sys.argv[1]

def load(name):
    z = np.load(f'{S}/out/proj_{name}.npz')
    return {k: z[k] for k in z.files}

def predict(name, T32, flux, em_edges, ei_mode='interp', nE=3000):
    P = load(name)
    ei_e, eps_e = P['ei'], P['eps']
    p = P['p_eps']                                      # (768, 15, 18) area per eps bin
    tot = p.sum(-1, keepdims=True)
    prob = np.divide(p, tot, out=np.zeros_like(p), where=tot > 0)   # P(eps bin | pix, Ei bin)
    E = np.geomspace(100., 10000., nE + 1); Ec = np.sqrt(E[1:] * E[:-1]); dE = np.diff(E)
    # total area at fine Ei: AEFF (nside 32) interpolated linearly in log Ei between centers
    a_ei = P['aeff_ei']; a_c = np.sqrt(a_ei[1:] * a_ei[:-1])
    aeffT = T32 @ P['aeff']                                # (nEi_aeff,) cm2 s, weighted by exposure
    # exposure-weighted dispersion: need P per nside-8 parent pixel weighted by T32*A32(Ei)
    pix32 = np.arange(12 * 32 ** 2)
    th, ph = hp.pix2ang(32, pix32)
    parent = hp.ang2pix(8, th, ph)
    W8 = np.zeros((768, P['aeff'].shape[1]))
    np.add.at(W8, parent, T32[:, None] * P['aeff'])         # (768, nEi_aeff) exposure*area per parent pixel
    lx = np.log(Ec)
    def interp_aeff(W):  # W (..., nEi_aeff) -> at Ec
        return np.array([np.interp(lx, np.log(a_c), w) for w in W.reshape(-1, W.shape[-1])]).reshape(W.shape[:-1] + (len(Ec),))
    W8f = interp_aeff(W8)                                    # (768, nE)
    # dispersion at fine Ei: interpolate prob in log Ei between bin centers at fixed eps bin (clamped)
    ic = np.sqrt(ei_e[1:] * ei_e[:-1])
    if ei_mode == 'interp':
        j = np.clip(np.searchsorted(np.log(ic), lx) - 1, 0, len(ic) - 2)
        w = np.clip((lx - np.log(ic[j])) / (np.log(ic[j + 1]) - np.log(ic[j])), 0, 1)
    else:  # nearest bin
        j = np.clip(np.searchsorted(ei_e, Ec) - 1, 0, len(ic) - 1); w = np.zeros_like(lx)
        j = np.minimum(j, len(ic) - 2); w = np.where(np.searchsorted(ei_e, Ec) - 1 > j, 1., 0.)
    # exposure-weighted probability per (Ei fine, eps bin)
    PE = (W8f[:, :, None] * ((1 - w)[None, :, None] * prob[:, j, :] + w[None, :, None] * prob[:, j + 1, :])).sum(0)  # (nE, neps)
    # Em bin counts: eps bin [a,b] -> Em [Ei(1+a), Ei(1+b)], uniform in Em within bin
    N = np.zeros(len(em_edges) - 1)
    F = flux(Ec) * dE
    for k in range(len(eps_e) - 1):
        lo = Ec * (1 + eps_e[k]); hi = Ec * (1 + eps_e[k + 1])
        for m in range(len(em_edges) - 1):
            ov = np.clip(np.minimum(hi, em_edges[m + 1]) - np.maximum(lo, em_edges[m]), 0, None) / (hi - lo)
            N[m] += np.sum(F * PE[:, k] * ov)
    return N, (F * PE.sum(1)).sum()
