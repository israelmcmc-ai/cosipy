"""Signal |ARM| core width and tail versus Distance bin, for the no-cut sample and with the other cuts added one by one.
Usage: python plot_arm_vs_distance.py signal_dir bkgorder_dir outdir"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from order_common import *

DL = ['<0.5', '0.5–1', '1–2', '2–3', '3–4', '4–6', '6–8', '8–12', '12–20', '>20']
cuts = [('no other cut', dict()), ('SQ ≤ 5', dict(sq=(2, 5))), ('first hit in the upper two layers', dict(lay=(2, 3))), ('φ < 90°', dict(phimax=90)), ('second hit below the first', dict(down=1)),
        ('all four', dict(sq=(2, 5), lay=(2, 3), phimax=90, down=1))]
def profile(kw):
    """per energy bin and Distance bin: signal events (weighted), w50, w68 (of events with |ARM| < 30 deg), tail fraction (|ARM| > 30), optimal window for S/sqrt(B)."""
    m = rect(**kw); out = np.full((4, NE, ND), np.nan); n = np.zeros((NE, ND))
    for i in range(NE):
        for j in range(ND):
            mj = (m[i] & (np.arange(ND) == j).reshape(1, ND, 1, 1, 1, 1))
            if not mj.any(): continue
            cum = mj.reshape(-1).astype(np.float32) @ Sc_all[i].reshape(-1, NA + 1)          # cumulative signal vs window, last = all
            core = cum[NA - 1]
            if core <= 0: continue
            n[i, j] = cum[-1]; out[0, i, j] = W[np.searchsorted(cum[:NA], 0.5 * core)]; out[1, i, j] = W[min(np.searchsorted(cum[:NA], 0.68 * core), NA - 1)]
            out[2, i, j] = 1 - core / cum[-1]
            B = (Nb[i] * mj).sum((1, 2, 3, 4, 5)) @ AREA
            if (B > 0).all(): out[3, i, j] = W[np.argmax(cum[:NA] / np.sqrt(B))]
    return out, n
prof = [profile(kw) for _, kw in cuts]
LABS = [f'{E_EDGES[i]:.0f}–{E_EDGES[i + 1]:.0f} keV' for i in range(NE)]
for q, (qi, ylab, fname) in enumerate(((0, 'core half-width w50 (50% of the events within 30°) [deg]', 'arm_vs_distance_w50.png'), (3, 'optimal ARM window for S/√B [deg]', 'arm_vs_distance_wopt.png'), (2, 'fraction of the signal with |ARM| > 30°', 'arm_vs_distance_tail.png'))):
    fig, axs = plt.subplots(2, 3, figsize=(17, 8.5), sharex=True)
    for i, ax in enumerate(axs.flat):
        for (nm, _), (pr, n) in zip(cuts, prof):
            ax.plot(range(ND), pr[qi, i], marker='o', ms=3, lw=1.3, label=nm, ls='-' if nm in ('no other cut', 'all four') else '--')
        ax.set_xticks(range(ND)); ax.set_xticklabels(DL, rotation=45, fontsize=8); ax.set_title(LABS[i]); ax.set_xlabel('Distance [cm]'); ax.set_ylabel(ylab, fontsize=8)
        if qi != 2: ax.set_yscale('log')
    axs[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(outdir, fname), dpi=100); plt.close(fig)
md = ['| cut | quantity | ' + ' | '.join(LABS) + ' |', '|' + '---|' * (NE + 2)]
for (nm, _), (pr, n) in zip(cuts[:1] + cuts[-1:], [prof[0], prof[-1]]):
    for qn, qi in (('w50 [deg]', 0), ('w_opt [deg]', 3)):
        for j in (2, 4, 5, 7):
            md.append(f'| {nm} | {qn}, Distance {DL[j]} | ' + ' | '.join(f'{pr[qi, i, j]:.1f}' for i in range(NE)) + ' |')
open(os.path.join(outdir, 'arm_vs_distance.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
