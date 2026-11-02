"""Effect of the upward-going filter |psi - phi| < theta0 on the SNR with and without dividing the transit in zenith bins.
divided     : SNR^2 = sum over 5-deg zenith bins of max_w A_theta(w)^2 / B_theta(w)   (window optimised per bin, background at the bin's position)
not divided : SNR   = max_w sum_theta A_theta(w) / sqrt(sum_theta B_theta(w))          (whole transit integrated, one common window)
Transit theta < THREF (default 180 deg), without and with a pointing cut theta < 45 deg. E^-2 source (GAMMA=2).
Usage: GAMMA=2 THREF=180 python plot_filter_divided.py signal_dir bkgpsichi_dir outdir"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from psichi_pos_common import *

THREF = float(os.environ.get('THREF', '180')); SUF = '' if THREF == 90 else f'_ref{THREF:g}'
def AB(zcap=None):
    sm, bm = cut(sq5=False, zcap=zcap); ok = ok0 & sm
    H = np.bincount(((iE * NT + iT) * (NA + 1) + iA)[ok], weights=wt[ok], minlength=NE * NT * (NA + 1)).reshape(NE, NT, NA + 1); A = np.cumsum(H, -1)[..., :NA] / SOL[None, :, None]
    Mc = (Np * bm).sum((3, 4, 5, 6)); B = np.stack([np.einsum('pq,pqtk->tk', Mc[i], F) for i in range(NE)]); return A, B, 1 - (Np * bm).sum((1, 2, 3, 4, 5, 6)) / Nt
def snr_div(A, B, tmax): sel = TC < tmax; return np.sqrt((A[:, sel] ** 2 / np.maximum(B[:, sel], 1e-30)).max(2).sum(1))
def snr_und(A, B, tmax):
    sel = TC < tmax; S = A[:, sel].sum(1); Bt = B[:, sel].sum(1); return (S / np.sqrt(np.maximum(Bt, 1e-30))).max(1)
A0, B0, _ = AB(None); base = {(m, t): f(A0, B0, t) for m, f in (('div', snr_div), ('und', snr_und)) for t in (THREF, 45.0)}
th0 = [5, 10, 15, 20, 30, 45, 60, 90, 120, 181]; res = {}
for t0 in th0:
    A, B, rem = AB(t0)
    for tmax in (THREF, 45.0):
        for m, f in (('div', snr_div), ('und', snr_und)): res[(t0, tmax, m)] = (f(A, B, tmax), rem)
allq = lambda v: np.sqrt((v ** 2).sum())
md = []
def table(t, rows, h):
    md.append(f'\n### {t}\n'); md.append('| ' + ' | '.join(h) + ' |'); md.append('|' + '---|' * len(h)); md.extend('| ' + ' | '.join(r) + ' |' for r in rows)
LABS = LAB
for tmax, nm in ((THREF, f'no pointing cut (transit θ < {THREF:g}°)'), (45.0, 'pointing cut θ < 45°')):
    rows = []
    for t0 in th0:
        d = res[(t0, tmax, 'div')][0]; u = res[(t0, tmax, 'und')][0]; b_d = base[('div', tmax)]; b_u = base[('und', tmax)]
        rows.append([f'{t0}°'] + [f'{100 * (d[i] / b_d[i] - 1):+.1f} / {100 * (u[i] / b_u[i] - 1):+.1f}' for i in range(NE)] + [f'{100 * (allq(d) / allq(b_d) - 1):+.1f} / {100 * (allq(u) / allq(b_u) - 1):+.1f}'] + [f'{100 * res[(t0, tmax, "div")][1].mean():.0f}'])
    table(f'{nm}: SNR change from the filter relative to the same analysis without the filter [%], divided / not divided', rows, ['θ0'] + LABS + ['all bins', 'bkg events rejected [%]'])
rows = [[f'{tmax:g}'] + [f'{base[("und", tmax)][i] / base[("div", tmax)][i]:.2f}' for i in range(NE)] + [f'{allq(base[("und", tmax)]) / allq(base[("div", tmax)]):.2f}'] for tmax in (THREF, 45.0)]
table('SNR of the undivided analysis relative to the divided analysis (no filter)', rows, ['transit range [deg]'] + LABS + ['all bins'])
fig, axs = plt.subplots(2, 3, figsize=(18, 9)); 
for row, tmax in enumerate((THREF, 45.0)):
    for i in range(NE):
        axs[row, 0].plot(th0, [100 * (res[(t, tmax, 'div')][0][i] / base[('div', tmax)][i] - 1) for t in th0], color=COL[i], marker='o', ms=3, label=LAB[i])
        axs[row, 1].plot(th0, [100 * (res[(t, tmax, 'und')][0][i] / base[('und', tmax)][i] - 1) for t in th0], color=COL[i], marker='o', ms=3)
    for m, ls, nm in (('div', '-', 'divided in zenith bins'), ('und', '--', 'not divided')):
        axs[row, 2].plot(th0, [100 * (allq(res[(t, tmax, m)][0]) / allq(base[(m, tmax)]) - 1) for t in th0], 'k' + ls, marker='s', ms=3, label=nm)
    axs[row, 2].plot(th0, [100 * res[(t, tmax, 'div')][1].mean() for t in th0], 'r:', label='bkg events rejected')
    for c, t in enumerate(('divided in zenith bins', 'not divided (integrated transit)', 'all energy bins')): axs[row, c].set_title(f'{t}; ' + (f'transit θ < {THREF:g}°' if row == 0 else 'pointing cut θ < 45°')); axs[row, c].set_xscale('log'); axs[row, c].set_xlabel('θ0 [deg]'); axs[row, c].set_ylabel('SNR change from the filter [%]'); axs[row, c].axhline(0, color='k', lw=.5)
axs[0, 0].legend(fontsize=7); axs[0, 2].legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(outdir, f'filter_divided{SUF}.png'), dpi=95)
open(os.path.join(outdir, f'filter_divided{SUF}.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
