"""SNR loss and background rejection of the quality cuts (Distance >= 1 cm, SQ <= 5, upper two layers, phi < 90 deg, upward-going filter |psi - phi| < 45 deg), with and without a 45 deg pointing cut.
E^-2 source, background at the source position, reference = no cut over the transit theta < 90 deg (see plot_sort_out.py).
Usage: GAMMA=2 python plot_cuts_pointing.py signal_dir bkgpsichi_dir outdir"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from psichi_pos_common import *

ref, _ = snr2(*cut(sq5=False)); THREF = 90.0; TP = 45.0
steps = [('Distance ≥ 1 cm', dict(dmin=1, sq5=False)), ('SQ ≤ 5', dict(sq5=True)), ('upper two layers', dict(lay=(2, 3), sq5=False)), ('φ < 90°', dict(phimax=90, sq5=False)), ('filter |ψ−φ| < 45°', dict(zcap=45, sq5=False))]
cum = []; kw = dict(sq5=False)
for nm, k in steps:
    kw = dict(kw); kw.update({a: b for a, b in k.items() if not (a == 'sq5' and not b)}); cum.append((nm, dict(kw)))
LABS = LAB; md = []
def table(t, rows, h):
    md.append(f'\n### {t}\n'); md.append('| ' + ' | '.join(h) + ' |'); md.append('|' + '---|' * len(h)); md.extend('| ' + ' | '.join(r) + ' |' for r in rows)
def evaluate(kwargs, tmax):
    s2, rem = snr2(*cut(**kwargs)); sel = TC < tmax; r = np.sqrt(s2[:, sel].sum(1) / ref[:, TC < THREF].sum(1)); ra = np.sqrt(s2[:, sel].sum() / ref[:, TC < THREF].sum())
    rp = np.sqrt(s2[:, sel].sum(1) / ref[:, sel].sum(1)); rpa = np.sqrt(s2[:, sel].sum() / ref[:, sel].sum()); tot = 1 - (tmax / THREF) * (1 - rem); return r, ra, rp, rpa, rem, tot
H = ['cuts'] + LABS + ['all bins']
for tmax, nm in ((THREF, 'no pointing cut (transit θ < 90°)'), (TP, 'with the pointing cut θ < 45°')):
    rows_s, rows_r, rows_p = [], [], []
    for cn, kw in cum:
        r, ra, rp, rpa, rem, tot = evaluate(kw, tmax); label = ' + '.join(c[0] for c in cum[:[c[0] for c in cum].index(cn) + 1]) if False else cn
        rows_s.append([f'+ {cn}'] + [f'{100 * (v - 1):+.1f}' for v in r] + [f'{100 * (ra - 1):+.1f}'])
        rows_p.append([f'+ {cn}'] + [f'{100 * (v - 1):+.1f}' for v in rp] + [f'{100 * (rpa - 1):+.1f}'])
        rows_r.append([f'+ {cn}'] + [f'{100 * v:.0f}' for v in rem] + [f'{100 * rem.mean():.0f}'] + ([f'{100 * v:.0f}' for v in tot] + [f'{100 * tot.mean():.0f}'] if tmax < THREF else []))
    table(f'{nm}: change of SNR relative to the reference (no cuts, θ < 90°) [%], cuts added cumulatively', rows_s, H)
    if tmax < THREF: table(f'{nm}: change of SNR relative to the pointing cut alone [%]', rows_p, H)
    table(f'{nm}: background rejected [%] (cuts added cumulatively)' + (' — event rejection of the cuts, then total including the time not used' if tmax < THREF else ''), rows_r, ['cuts'] + [f'ev {l}' for l in LABS] + ['ev mean'] + ([f'tot {l}' for l in LABS] + ['tot mean'] if tmax < THREF else []))
# each cut alone, and the full set without one of them
alone = [(cn, dict(k)) for cn, k in steps]; full = cum[-1][1]
for tmax, nm in ((THREF, 'no pointing cut'), (TP, 'with pointing cut θ < 45°')):
    rows = []
    for cn, k in alone:
        r, ra, rp, rpa, rem, tot = evaluate(k, tmax); rows.append([f'only {cn}'] + [f'{100 * (v - 1):+.1f}' for v in r] + [f'{100 * (ra - 1):+.1f}', f'{100 * rem.mean():.0f}', f'{100 * tot.mean():.0f}'])
    for cn, k in zip([c[0] for c in steps], range(len(steps))):
        kw = {}
        for j, (cn2, k2) in enumerate(steps):
            if j != k: kw.update({a: b for a, b in k2.items() if not (a == 'sq5' and not b)})
        kw.setdefault('sq5', False); r, ra, rp, rpa, rem, tot = evaluate(kw, tmax); rows.append([f'all but {cn}'] + [f'{100 * (v - 1):+.1f}' for v in r] + [f'{100 * (ra - 1):+.1f}', f'{100 * rem.mean():.0f}', f'{100 * tot.mean():.0f}'])
    r, ra, rp, rpa, rem, tot = evaluate(full, tmax); rows.append(['all five'] + [f'{100 * (v - 1):+.1f}' for v in r] + [f'{100 * (ra - 1):+.1f}', f'{100 * rem.mean():.0f}', f'{100 * tot.mean():.0f}'])
    table(f'{nm}: each cut alone and the full set with one cut removed: SNR change vs the reference [%], mean event rejection, total rejection (including time)', rows, ['cuts'] + LABS + ['all bins', 'events rejected [%]', 'total rejected [%]'])
# figure: SNR change and rejection per energy bin for the cumulative sets
fig, axs = plt.subplots(2, 2, figsize=(14, 9)); x = np.arange(len(cum))
for col, (tmax, nm) in enumerate(((THREF, 'no pointing cut'), (TP, 'pointing cut θ < 45°'))):
    for i in range(NE):
        axs[0, col].plot(x, [100 * (evaluate(kw, tmax)[0][i] - 1) for _, kw in cum], color=COL[i], marker='o', ms=3, label=LABS[i]); axs[1, col].plot(x, [100 * (evaluate(kw, tmax)[5][i] if tmax < THREF else 100 * evaluate(kw, tmax)[4][i]) for _, kw in cum], color=COL[i], marker='o', ms=3)
    axs[0, col].plot(x, [100 * (evaluate(kw, tmax)[1] - 1) for _, kw in cum], 'k--', marker='s', label='all bins'); axs[0, col].set_title(nm); axs[0, col].set_ylabel('SNR change vs reference [%]'); axs[1, col].set_ylabel('background rejected [%]' + (' (time + events)' if tmax < THREF else ' (events)'))
    for r in (0, 1): axs[r, col].set_xticks(x); axs[r, col].set_xticklabels(['+ ' + c[0] for c in cum], rotation=25, fontsize=8, ha='right')
axs[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'cuts_pointing.png'), dpi=100)
open(os.path.join(outdir, 'cuts_pointing_summary.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
