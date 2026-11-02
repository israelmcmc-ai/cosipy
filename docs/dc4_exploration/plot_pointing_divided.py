"""Effect of the pointing cut (only zenith bins theta < theta_max) on the SNR with and without dividing the transit in zenith bins.
divided: SNR^2 = sum_{theta<theta_max} max_w A^2/B_theta;  not divided: SNR = max_w sum_{theta<theta_max} A / sqrt(sum B_theta) (one common window).
E^-2 source, background at the source position, reference transit theta < THREF (default 180 deg). Background rejection = time not used = 1 - theta_max/THREF.
Usage: GAMMA=2 THREF=180 python plot_pointing_divided.py signal_dir bkgpsichi_dir outdir"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from psichi_pos_common import *

THREF = float(os.environ.get('THREF', '180')); SUF = '' if THREF == 90 else f'_ref{THREF:g}'
sm, bm = cut(sq5=False); ok = ok0 & sm
H = np.bincount(((iE * NT + iT) * (NA + 1) + iA)[ok], weights=wt[ok], minlength=NE * NT * (NA + 1)).reshape(NE, NT, NA + 1); A = np.cumsum(H, -1)[..., :NA] / SOL[None, :, None]
Mc = (Np * bm).sum((3, 4, 5, 6)); B = np.stack([np.einsum('pq,pqtk->tk', Mc[i], F) for i in range(NE)])
def div(tmax): sel = TC < tmax; return np.sqrt((A[:, sel] ** 2 / np.maximum(B[:, sel], 1e-30)).max(2).sum(1))
def und(tmax): sel = TC < tmax; return (A[:, sel].sum(1) / np.sqrt(np.maximum(B[:, sel].sum(1), 1e-30))).max(1)
def und_w(tmax): sel = TC < tmax; return W[(A[:, sel].sum(1) / np.sqrt(np.maximum(B[:, sel].sum(1), 1e-30))).argmax(1)]
allq = lambda v: np.sqrt((v ** 2).sum())
tms = [5 * k for k in range(1, int(THREF // 5) + 1)]; Dref = div(THREF); Uref = und(THREF)
md = []
def table(t, rows, h):
    md.append(f'\n### {t}\n'); md.append('| ' + ' | '.join(h) + ' |'); md.append('|' + '---|' * len(h)); md.extend('| ' + ' | '.join(r) + ' |' for r in rows)
sel_t = [10, 15, 20, 30, 45, 60, 75, 90, 120, 150, 180]
table('Pointing cut: SNR change [%] relative to the divided no-cut reference (divided / not divided), and time rejected', [[f'{t}°'] + [f'{100 * (div(t)[i] / Dref[i] - 1):+.1f} / {100 * (und(t)[i] / Dref[i] - 1):+.1f}' for i in range(NE)] + [f'{100 * (allq(div(t)) / allq(Dref) - 1):+.1f} / {100 * (allq(und(t)) / allq(Dref) - 1):+.1f}', f'{100 * (1 - t / THREF):.0f}'] for t in sel_t], ['θmax'] + LAB + ['all bins', 'time rejected [%]'])
table('Pointing cut: SNR change [%] relative to the same analysis without a pointing cut (divided / not divided)', [[f'{t}°'] + [f'{100 * (div(t)[i] / Dref[i] - 1):+.1f} / {100 * (und(t)[i] / Uref[i] - 1):+.1f}' for i in range(NE)] + [f'{100 * (allq(div(t)) / allq(Dref) - 1):+.1f} / {100 * (allq(und(t)) / allq(Uref) - 1):+.1f}'] for t in sel_t], ['θmax'] + LAB + ['all bins'])
best = [(int(tms[int(np.argmax(und(0 + 0) if False else [und(t)[i] for t in tms]))]), max(und(t)[i] for t in tms) / Dref[i]) for i in range(NE)]
table('Best pointing cut for the undivided analysis (maximum SNR over θmax), relative to the divided reference', [[LAB[i], f'{best[i][0]}°', f'{100 * (best[i][1] - 1):+.1f}', f'{und_w(best[i][0])[i]:g}'] for i in range(NE)], ['bin', 'best θmax', 'SNR vs divided reference [%]', 'window [deg]'])
bt = max(tms, key=lambda t: allq(und(t))); md.append(f'\nAll bins: the undivided SNR is maximal for θmax = {bt}° at {100 * (allq(und(bt)) / allq(Dref) - 1):+.1f}% relative to the divided reference (no pointing cut, undivided: {100 * (allq(Uref) / allq(Dref) - 1):+.1f}%).')
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i in range(NE):
    axs[0].plot(tms, [100 * (div(t)[i] / Dref[i] - 1) for t in tms], color=COL[i], label=LAB[i]); axs[1].plot(tms, [100 * (und(t)[i] / Dref[i] - 1) for t in tms], color=COL[i])
for m, f, ls, nm in (('d', div, '-', 'divided'), ('u', und, '--', 'not divided')): axs[2].plot(tms, [100 * (allq(f(t)) / allq(Dref) - 1) for t in tms], 'k' + ls, label=nm)
axs[2].plot(tms, [100 * (1 - t / THREF) for t in tms], 'r:', label='time (background) rejected')
for a, t in zip(axs, ('divided in zenith bins', 'not divided (integrated)', 'all energy bins')): a.set_xlabel('maximum zenith θmax [deg]'); a.set_ylabel('SNR change vs the divided no-cut reference [%]'); a.set_title(f'{t}; reference transit θ < {THREF:g}°'); a.axhline(0, color='k', lw=.5)
axs[0].legend(fontsize=7); axs[2].legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(outdir, f'pointing_divided{SUF}.png'), dpi=95)
open(os.path.join(outdir, f'pointing_divided{SUF}.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
