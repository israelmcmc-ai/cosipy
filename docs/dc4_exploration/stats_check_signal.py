"""Signal statistics per measured-energy bin through the cumulative cuts: events, effective events N_eff = (sum w)^2 / sum w^2 (E_true^-GAMMA weights) and
the same inside the optimal-ish |ARM| < 5 deg window, plus the file-to-file scatter of the fraction of events surviving each cut.
Usage: GAMMA=2 python stats_check_signal.py signal_dir outdir"""
import sys, os, glob
import numpy as np
from tra_quality import E_EDGES

sigdir, outdir = sys.argv[1:3]; G = float(os.environ.get('GAMMA', '2')); files = sorted(glob.glob(sigdir + '/*.npz'))
keys = ('E', 'phi', 'z1', 'z2', 'd01', 'sq', 'cat1', 'theta_src', 'Etrue', 'arm')
per = [{k: np.load(f)[k] for k in keys} for f in files]
LAB = [f'{E_EDGES[i]:.0f}–{E_EDGES[i + 1]:.0f} keV' for i in range(6)]
def prep(d):
    phi = np.rad2deg(d['phi'].astype(float)); psi = np.rad2deg(np.arccos(np.clip(-(d['z2'].astype(float) - d['z1']) / d['d01'], -1, 1)))
    cuts = [('no cuts', np.ones(len(phi), bool))]; m = np.ones(len(phi), bool)
    for nm, c in (('+ Distance ≥ 1 cm', d['d01'] >= 1), ('+ SQ ≤ 5', d['sq'] <= 5), ('+ upper two layers', d['cat1'] // 3 >= 2), ('+ φ < 90°', phi < 90), ('+ filter |ψ−φ| < 45°', np.abs(psi - phi) < 45)):
        m = m & c; cuts.append((nm, m.copy()))
    return cuts, (np.digitize(d['E'], E_EDGES) - 1), (d['Etrue'].astype(float) / 1000.0) ** -G
pre = [prep(d) for d in per]
md = []
def neff(w): return w.sum() ** 2 / max((w ** 2).sum(), 1e-300)
md += ['| bin | cuts | events (all θ) | N_eff | events θ_src < 45° | N_eff (θ<45°) | events with |ARM| < 5° | file-to-file scatter of the surviving fraction [%] |', '|---|---|---|---|---|---|---|---|']
for i in range(6):
    for c, (nm, _) in enumerate(pre[0][0]):
        n = ne = n45 = ne45 = nc = 0; frac = []; wl = []; wl45 = []
        for (cuts, iE, w), d in zip(pre, per):
            sel = cuts[c][1] & (iE == i); sel0 = (iE == i); n += sel.sum(); wl.append(w[sel]); wl45.append(w[sel & (d['theta_src'] < 45)]); n45 += (sel & (d['theta_src'] < 45)).sum(); nc += (sel & (np.abs(d['arm']) < 5)).sum()
            frac.append(w[sel].sum() / max(w[sel0].sum(), 1e-300))
        wa = np.concatenate(wl); wb = np.concatenate(wl45)
        md.append(f'| {LAB[i]} | {nm} | {n} | {neff(wa):.0f} | {n45} | {neff(wb):.0f} | {nc} | {100 * np.std(frac) / max(np.mean(frac), 1e-300):.1f} |')
open(os.path.join(outdir, 'stats_check_signal.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
