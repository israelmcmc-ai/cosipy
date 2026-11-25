import sys, numpy as np, healpy as hp
from astropy.io import fits
S = sys.argv[1]
X = np.load(f'{S}/out/exposure.npz'); T32 = X['crab3m_n32_occ0']
d = fits.open(f'{S}/data/crab_powerlaw_3months_unbinned_data_filtered_with_SAAcut.fits.gz')[1].data
for name, sel in (('nn', np.ones(len(d), bool)), ('simple', d['Distance'] >= 1)):
    P = np.load(f'{S}/out/proj_{name}.npz')
    p = P['p_phi']; tot = p.sum(-1, keepdims=True); prob = np.divide(p, tot, out=np.zeros_like(p), where=tot > 0)
    ei = P['ei']; a_ei = P['aeff_ei']; a_c = np.sqrt(a_ei[1:] * a_ei[:-1])
    E = np.geomspace(100, 10000, 2001); Ec = np.sqrt(E[1:] * E[:-1]); F = 4.94717 * Ec ** -2 * np.diff(E)
    th, ph = hp.pix2ang(32, np.arange(12 * 32 ** 2)); parent = hp.ang2pix(8, th, ph)
    W8 = np.zeros((768, P['aeff'].shape[1])); np.add.at(W8, parent, T32[:, None] * P['aeff'])
    W8f = np.array([np.interp(np.log(Ec), np.log(a_c), w) for w in W8])
    j = np.clip(np.searchsorted(ei, Ec) - 1, 0, len(ei) - 2)
    pred = np.einsum('pe,e,pek->k', W8f, F, prob[:, j, :])
    phi_e = P['phi']
    obs, _ = np.histogram(np.degrees(d['Phi'][sel]), phi_e)
    r = obs / pred; r /= obs.sum() / pred.sum()
    print(name, 'Phi bins, shape ratio obs/pred:')
    print('  ' + ' '.join(f'{a:.0f}-{b:.0f}:{x:.2f}' for a, b, x in zip(phi_e[:-1], phi_e[1:], r)))
    D = d['Distance'][sel]
    print('  Distance pctl', np.percentile(D, [0, 1, 5, 50, 95, 99, 100]))
print('Phi max deg', np.degrees(d['Phi'].max()))
