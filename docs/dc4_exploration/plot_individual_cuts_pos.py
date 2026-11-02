"""Individual cuts (one at a time) with the background evaluated at the source zenith angle (psichi_pos_common): SNR change, background rejection and ARM, transit theta < 45 deg.
The number-of-hits row uses the sphere-averaged background of the order histogram (the psichi histogram only has an SQ > 5 flag) and is marked as such.
Usage: python plot_individual_cuts_pos.py signal_dir bkgpsichi_dir outdir bkgorder_dir"""
import sys, os, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
odir = sys.argv[4]; sys.argv = sys.argv[:4]
from psichi_pos_common import *

TMAX = 45.0; selT = TC < TMAX
def evaluate(sm, bm):
    ok = ok0 & sm; H = np.bincount(((iE * NT + iT) * (NA + 1) + iA)[ok], weights=wt[ok], minlength=NE * NT * (NA + 1)).reshape(NE, NT, NA + 1)
    A = np.cumsum(H, -1)[..., :NA] / SOL[None, :, None]; Mc = (Np * bm).sum((3, 4, 5, 6)); s2 = np.zeros((NE, NT)); wo = np.zeros(NE); w50 = np.zeros(NE)
    for i in range(NE):
        B = np.einsum('pq,pqtk->tk', Mc[i], F); v = A[i] ** 2 / np.maximum(B, 1e-30); k = v.argmax(1); s2[i] = v[np.arange(NT), k]; wsel = s2[i][selT]
        wo[i] = (W[k][selT] * wsel).sum() / max(wsel.sum(), 1e-30); pool = A[i][selT].sum(0); w50[i] = W[min(np.searchsorted(pool, 0.5 * pool[-1]), NA - 1)] if pool[-1] > 0 else np.nan
    return s2, wo, w50, 1 - (Np * bm).sum((1, 2, 3, 4, 5, 6)) / Nt
# sphere-averaged background for the SQ row (order histogram) -- the transit signal as above
No = sum(np.load(f)['N'] for f in glob.glob(odir + '/*.npz')); PC36 = (np.arange(36) + .5) * 5.0
p36 = np.deg2rad(PC36)[:, None]; A36 = (np.cos(np.maximum(p36 - np.deg2rad(W)[None, :], 0)) - np.cos(np.minimum(p36 + np.deg2rad(W)[None, :], np.pi))) / 2; Nto = No.sum((1, 2, 3, 4, 5, 6))
def evaluate_sq(smin, smax):
    sm = (ev['sq'] >= smin) & (ev['sq'] <= smax); ok = ok0 & sm; H = np.bincount(((iE * NT + iT) * (NA + 1) + iA)[ok], weights=wt[ok], minlength=NE * NT * (NA + 1)).reshape(NE, NT, NA + 1)
    A = np.cumsum(H, -1)[..., :NA] / SOL[None, :, None]; m = ((np.arange(2, 8) >= smin) & (np.arange(2, 8) <= smax)).reshape(1, 1, 1, 6, 1, 1, 1); B = (No * m).sum((2, 3, 4, 5, 6)) @ A36
    s2 = np.zeros((NE, NT)); wo = np.zeros(NE); w50 = np.zeros(NE)
    for i in range(NE):
        v = A[i] ** 2 / np.maximum(B[i], 1e-30)[None, :]; k = v.argmax(1); s2[i] = v[np.arange(NT), k]; wsel = s2[i][selT]; wo[i] = (W[k][selT] * wsel).sum() / max(wsel.sum(), 1e-30)
        pool = A[i][selT].sum(0); w50[i] = W[min(np.searchsorted(pool, 0.5 * pool[-1]), NA - 1)] if pool[-1] > 0 else np.nan
    return s2, wo, w50, 1 - (No * m).sum((1, 2, 3, 4, 5, 6)) / Nto
def cut2(dmin=0, lay=(0, 1, 2, 3), phimin=0, phimax=181, zcap=None):
    sm, bm = cut(dmin=dmin, lay=lay, phimax=phimax, sq5=False, zcap=zcap); sm = sm & (phi >= phimin); bm = bm & (PC >= phimin).reshape(1, 72, 1, 1, 1, 1, 1); return sm, bm
base = evaluate(*cut2()); base_sq = evaluate_sq(2, 7)
def res(r, b):
    return dict(ch=100 * (np.sqrt(r[0][:, selT].sum(1) / b[0][:, selT].sum(1)) - 1), cc=100 * (np.sqrt(r[0][:, selT].sum() / b[0][:, selT].sum()) - 1), rem=r[3], w50=r[2], wopt=r[1])
sweeps = {}
sweeps['Minimum Distance [cm]'] = ([0, 0.5, 1, 2, 3, 4, 6, 8], [res(evaluate(*cut2(dmin=x)), base) for x in [0, 0.5, 1, 2, 3, 4, 6, 8]])
sweeps['Number of hits (sphere-averaged B): SQ ≤ n'] = (['2', '3', '4', '5', '6', '7+'], [res(evaluate_sq(2, n), base_sq) for n in (2, 3, 4, 5, 6, 7)])
sweeps['SQ ≥ n'] = (['2', '3', '4', '5', '6'], [res(evaluate_sq(n, 7), base_sq) for n in (2, 3, 4, 5, 6)])
sweeps['First-hit layer'] = (['all', 'not bottom', 'upper two', 'top only'], [res(evaluate(*cut2(lay=l)), base) for l in [(0, 1, 2, 3), (1, 2, 3), (2, 3), (3,)]])
sweeps['φ < x [deg]'] = ([30, 45, 60, 90, 120, 150, 180], [res(evaluate(*cut2(phimax=x + 0.1)), base) for x in [30, 45, 60, 90, 120, 150, 180]])
sweeps['φ > x [deg]'] = ([0, 15, 30, 45, 60, 90], [res(evaluate(*cut2(phimin=x)), base) for x in [0, 15, 30, 45, 60, 90]])
sweeps['Zenith cap θ0 [deg] (|ψ−φ| < θ0)'] = ([5, 10, 15, 20, 30, 45, 60, 90, 120], [res(evaluate(*cut2(zcap=x)), base) for x in [5, 10, 15, 20, 30, 45, 60, 90, 120]])
sweeps['Psichi margin [deg] (|ψ−φ| < 45° + margin)'] = ([0, 5, 10, 15, 20, 30, 45], [res(evaluate(*cut2(zcap=45 + x)), base) for x in [0, 5, 10, 15, 20, 30, 45]])
groups = [('Minimum Distance [cm]', ['Minimum Distance [cm]']), ('Number of hits', ['Number of hits (sphere-averaged B): SQ ≤ n', 'SQ ≥ n']), ('First-hit layer', ['First-hit layer']), ('φ', ['φ < x [deg]', 'φ > x [deg]']),
          ('Zenith cap θ0 [deg]', ['Zenith cap θ0 [deg] (|ψ−φ| < θ0)']), ('Psichi margin [deg]', ['Psichi margin [deg] (|ψ−φ| < 45° + margin)'])]
fig, axs = plt.subplots(len(groups), 4, figsize=(22, 4.0 * len(groups)))
for g, (title, names) in enumerate(groups):
    for k, nm in enumerate(names):
        xs, out = sweeps[nm]; ls = '-' if k == 0 else '--'; xi = np.arange(len(xs))
        for i in range(NE):
            axs[g, 0].plot(xi, [r['ch'][i] for r in out], color=COL[i], ls=ls, marker='o', ms=3, lw=1.1, label=LAB[i] if (g == 0 and k == 0) else None)
            axs[g, 1].plot(xi, [100 * r['rem'][i] for r in out], color=COL[i], ls=ls, marker='o', ms=3, lw=1.1)
            axs[g, 2].plot(xi, [r['w50'][i] for r in out], color=COL[i], ls=ls, marker='o', ms=3, lw=1.1); axs[g, 3].plot(xi, [r['wopt'][i] for r in out], color=COL[i], ls=ls, marker='o', ms=3, lw=1.1)
        axs[g, 0].plot(xi, [r['cc'] for r in out], 'k', ls=ls, marker='s', ms=3, lw=1.6, label='all bins' if (g == 0 and k == 0) else None)
    for c, yl in enumerate(('S/√B change [%]', 'background rejected [%]', 'ARM core half-width w50 [deg]', 'optimal ARM window (SNR²-weighted) [deg]')):
        a = axs[g, c]; a.set_ylabel(yl); a.set_xlabel(title + (' (dashed: second cut of the pair)' if len(names) > 1 else '')); a.set_xticks(np.arange(len(sweeps[names[0]][0]))); a.set_xticklabels([str(x) for x in sweeps[names[0]][0]], fontsize=8)
        if c == 0: a.axhline(0, color='k', lw=.5)
        if c >= 2: a.set_yscale('log')
axs[0, 0].legend(fontsize=7); fig.suptitle('Individual cuts, transit θ < 45°, background at the source position (number-of-hits row: sphere-averaged background); dashed = second cut of the pair in a row', y=0.995)
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'individual_cuts_pos.png'), dpi=80)
md = ['(energy bins: ' + ', '.join(LAB) + ')']
for nm, (xs, out) in sweeps.items():
    md += [f'\n### {nm}\n', '| value | all-bin S/√B [%] | per bin S/√B [%] | bkg rejected per bin [%] | w50 per bin [deg] | w_opt per bin [deg] |', '|---|---|---|---|---|---|']
    for x, r in zip(xs, out): md.append(f'| {x} | {r["cc"]:+.1f} | ' + ' / '.join(f'{v:+.0f}' for v in r['ch']) + ' | ' + ' / '.join(f'{100 * v:.0f}' for v in r['rem']) + ' | ' + ' / '.join(f'{v:.1f}' for v in r['w50']) + ' | ' + ' / '.join(f'{v:.1f}' for v in r['wopt']) + ' |')
open(os.path.join(outdir, 'individual_cuts_pos.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
