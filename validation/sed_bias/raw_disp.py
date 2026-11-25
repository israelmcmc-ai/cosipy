import sys, numpy as np, healpy as hp
S = sys.argv[1]
np.set_printoptions(linewidth=220, precision=3, suppress=True)
T8 = np.load(f'{S}/out/exposure.npz')['crab3m_n8_occ0']
bins = [(1585, 2512), (2512, 3981), (3981, 6310), (6310, 10000)]
for name in ('nn', 'simple'):
    P = np.load(f'{S}/out/proj_{name}.npz'); ei, eps = P['ei'], P['eps']
    p = np.einsum('p,pie->ie', T8, P['p_eps'])
    for mode in ('nearest', 'interp'):
        res = []
        for lo, hi in bins:
            E = np.linspace(lo, hi, 400); fr = []
            for e in E:
                if mode == 'nearest':
                    j = np.searchsorted(ei, e) - 1; q = p[j] / p[j].sum()
                else:
                    c = np.sqrt(ei[1:] * ei[:-1]); j = np.clip(np.searchsorted(c, e) - 1, 0, len(c) - 2)
                    w = np.clip(np.log(e / c[j]) / np.log(c[j + 1] / c[j]), 0, 1)
                    q = (1 - w) * p[j] / p[j].sum() + w * p[j + 1] / p[j + 1].sum()
                a = np.clip((lo / e - 1 - eps[:-1]) / np.diff(eps), 0, 1); b = np.clip((min(hi, 1e9) / e - 1 - eps[:-1]) / np.diff(eps), 0, 1)
                fr.append(np.sum(q * (b - a)))
            res.append(np.mean(fr))
        print(f'{name:6s} {mode:8s}', np.array(res))
    print(f'{name:6s} raw eps distribution per Ei bin (fraction):')
    for j in range(8, 15):
        print(f'   Ei {ei[j]:7.0f}-{ei[j+1]:7.0f}: ', p[j] / p[j].sum())
print('eps edges', P['eps'])
