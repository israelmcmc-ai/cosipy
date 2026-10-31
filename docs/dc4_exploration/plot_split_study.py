"""Sensitivity when the events passing a rectangular cut are split into Distance bins, each subset gets its own optimal |ARM| window,
and the subsets' S/sqrt(B) are added in quadrature (independent subsamples). Gain relative to the no-cut single-window S/sqrt(B), versus energy.
Usage: python plot_split_study.py signal_dir bkgorder_dir outdir"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from order_common import *

DL = ['<0.5', '0.5–1', '1–2', '2–3', '3–4', '4–6', '6–8', '8–12', '12–20', '20+']
def split_fom(m, S_=Sc_all, groups=None):
    """m: bool mask over (NE,)+CS. Returns (NE,) quadrature S/sqrt(B) over Distance subsets, (NE, ND) per-subset values and windows."""
    groups = groups or [[j] for j in range(ND)]
    f = np.zeros((NE, len(groups))); wv = np.zeros((NE, len(groups))); nb = np.zeros((NE, len(groups)))
    for g, js in enumerate(groups):
        mg = m & np.isin(np.arange(ND), js).reshape(1, 1, ND, 1, 1, 1, 1)
        for i in range(NE):
            if not mg[i].any(): continue
            mf = mg[i].reshape(-1).astype(np.float32); S = mf @ S_[i].reshape(-1, NA + 1)[:, :NA]; B = (Nb[i] * mg[i]).sum((1, 2, 3, 4, 5)) @ AREA
            fv = np.where(B > 0, S / np.sqrt(np.maximum(B, 1e-30)), 0); k = fv.argmax(); f[i, g] = fv[k]; wv[i, g] = W[k]; nb[i, g] = (Nb[i] * mg[i]).sum()
    return np.sqrt((f ** 2).sum(1)), f, wv, nb
base = B0[:, 0]
sets = [('no cut (split by Distance only)', dict()),
        ('second hit below first', dict(down=1)),
        ('Distance ≥ 1 cm, SQ ≤ 5, upper two layers, φ < 90°', dict(dmin=1, sq=(2, 5), lay=(2, 3), phimax=90)),
        ('Distance ≥ 1 cm, SQ ≤ 5, upper two layers, φ < 90°, second hit below first', dict(dmin=1, sq=(2, 5), lay=(2, 3), phimax=90, down=1)),
        ('SQ ≤ 5, upper two layers, φ < 90°, second hit below first (no Distance cut)', dict(sq=(2, 5), lay=(2, 3), phimax=90, down=1)),
        ('Distance ≥ 1 cm, SQ ≤ 5, first hit above the bottom layer, φ < 90°, second hit below first', dict(dmin=1, sq=(2, 5), lay=(1, 2, 3), phimax=90, down=1))]
md = []
def table(t, rows, h):
    md.append(f'\n### {t}\n'); md.append('| ' + ' | '.join(h) + ' |'); md.append('|' + '---|' * len(h)); md.extend('| ' + ' | '.join(r) + ' |' for r in rows)
res = []
for nm, kw in sets:
    m = rect(**kw); single = fom_mask(m); q, f, wv, nb = split_fom(m)
    h = []
    for a in (0, 1):
        qa = split_fom(m, Sc[a])[0]; b0 = fom_mask(FULL, Sc[a])[:, 0]; h.append(100 * (qa / b0 - 1))
    err = np.abs(h[0] - h[1]) / 2
    res.append(dict(nm=nm, rem=100 * (1 - (Nb * m).sum() / Nb.sum()), rem_i=100 * single[:, 3], sig=100 * single[:, 2], one=100 * (single[:, 0] / base - 1), split=100 * (q / base - 1), err=err, f=f, wv=wv, nb=nb,
                    allone=100 * (np.sqrt((single[:, 0] ** 2).sum()) / np.sqrt((base ** 2).sum()) - 1), allsplit=100 * (np.sqrt((q ** 2).sum()) / np.sqrt((base ** 2).sum()) - 1)))
table('S/√B change vs the no-cut single-window value [%], per energy bin: one window for all events passing the cut (first row) vs Distance split with quadrature sum (second row)',
      [[r['nm'], nm2] + [f'{v:+.1f}' for v in r[key]] + [f'{r[ak]:+.1f}'] for r in res for nm2, key, ak in (('single window', 'one', 'allone'), ('split by Distance', 'split', 'allsplit'))], ['cut', 'analysis'] + LAB + ['all bins'])
table('Background events removed [%] (events NOT used) and signal kept [%], by energy bin', [[r['nm'], f'{r["rem"]:.0f}'] + [f'{v:.0f}' for v in r['rem_i']] + [f'{v:.0f}' for v in r['sig']] for r in res], ['cut', 'all'] + [f'bkg {l}' for l in LAB] + [f'sig {l}' for l in LAB])
table('Half-sample difference of the split result (±, percentage points)', [[r['nm']] + [f'{v:.1f}' for v in r['err']] for r in res], ['cut'] + LAB)
r = res[3]
rows = []
for j in range(ND):
    rows.append([DL[j]] + [f'{(r["f"][i, j] / np.sqrt((r["f"][i] ** 2).sum()) ** 1) ** 2 * 100:.0f} / {100 * r["nb"][i, j] / max(r["nb"][i].sum(), 1):.0f} / {r["wv"][i, j]:g}' for i in range(NE)])
table(f'Per Distance subset for "{r["nm"]}": share of the S/√B² / share of the background events kept / optimal ARM half-width [deg]', rows, ['Distance [cm]'] + LAB)
# plot
fig, axs = plt.subplots(1, 2, figsize=(15, 5.2)); x = np.arange(NE)
for j, r in enumerate(res):
    axs[0].errorbar(x + 0.07 * (j - 2.5), r['split'], r['err'], marker='o', ms=4, capsize=2, label=r['nm'] + f' ({r["rem"]:.0f}% bkg removed)')
    axs[1].plot(x + 0.07 * (j - 2.5), r['one'], marker='s', ms=4, ls='--')
for a, t in zip(axs, ('Distance split + quadrature sum', 'single window for all events passing the cut')): a.set_xticks(x); a.set_xticklabels(LAB, rotation=25); a.axhline(0, color='k', lw=.5); a.set_ylabel('S/√B change vs no cut [%]'); a.set_title(t)
axs[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'split_distance_vs_energy.png'), dpi=100)
open(os.path.join(outdir, 'split_summary.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
