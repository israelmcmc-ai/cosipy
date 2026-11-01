"""SNR change, background rejection and ARM versus the parameter of each individual cut (one cut at a time), for a transit using only theta < theta_max.
Cuts: minimum Distance; number of hits (SQ <= n, SQ >= n); first-hit layers; phi (phi < x, phi > x); zenith cap theta0 of the psichi-consistency cut (|psi - phi| < theta0, signal transit theta < theta0);
psichi margin (|psi - phi| < 45 deg + margin, transit theta < 45 deg).
SNR^2 = sum over 5-deg zenith bins of max_w A(theta,w)^2/B(w) (effective area ~ events/(sin theta dtheta)); each sweep has its own no-cut baseline.
ARM: w50 = half-width containing 50% of the (zenith-weighted) signal events with |ARM| < 30 deg; w_opt = optimal window of the zenith-pooled S/sqrt(B).
Usage: python plot_individual_cuts.py signal_dir bkgorder_dir bkgpsichi_dir outdir"""
import sys, os, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tra_quality import E_EDGES

sigdir, odir, pdir, outdir = sys.argv[1:5]
NE, NA, DA = 6, 60, 0.5; DEDGE = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 20]); W = DA * np.arange(1, NA + 1)
def area(pc): p = np.deg2rad(pc)[:, None]; w = np.deg2rad(W)[None, :]; return (np.cos(np.maximum(p - w, 0)) - np.cos(np.minimum(p + w, np.pi))) / 2
PC36 = (np.arange(36) + .5) * 5.0; PC72 = (np.arange(72) + .5) * 2.5; A36, A72 = area(PC36), area(PC72)
No = sum(np.load(f)['N'] for f in glob.glob(odir + '/*.npz'))        # (E, phi36, D10, SQ6, L1 4, L2 4, dir 2)
Np = sum(np.load(f)['N'] for f in glob.glob(pdir + '/*.npz'))        # (E, phi72, psi72, D10, L1 4, SQ>5 2, dir 2)
LAB = [f'{E_EDGES[i]:.0f}–{E_EDGES[i + 1]:.0f} keV' for i in range(NE)]; COL = plt.cm.viridis(np.linspace(0, 0.9, NE))
ev = {k: np.concatenate([np.load(f)[k] for f in sorted(glob.glob(sigdir + '/*.npz'))]) for k in ('E', 'arm', 'Etrue', 'theta_src', 'phi', 'd01', 'sq', 'cat1', 'z1', 'z2')}
phi = np.rad2deg(ev['phi'].astype(float)); psi = np.rad2deg(np.arccos(np.clip(-(ev['z2'].astype(float) - ev['z1']) / ev['d01'], -1, 1))); dpf = np.abs(psi - phi)
iE = np.digitize(ev['E'], E_EDGES) - 1; iA = np.clip((np.abs(ev['arm']) / DA).astype(int), 0, NA); wt = ev['Etrue'].astype(float) ** -2.1; lay1 = ev['cat1'] // 3
validE = (iE >= 0) & (iE < NE)
def run(cap, sm, Bw, rem):
    """signal: events with true zenith < cap passing sm; Bw (NE, NA) background in the window; rem (NE,) rejection."""
    nt = int(round(cap / 5)); th = np.arange(0, cap + .1, 5.0); tc = 0.5 * (th[1:] + th[:-1]); sol = np.sin(np.deg2rad(tc)) * np.deg2rad(5.0)
    ok = validE & (ev['theta_src'] < cap) & sm; iT = np.clip((ev['theta_src'] / 5).astype(int), 0, nt - 1)
    H = np.bincount(((iE * nt + iT) * (NA + 1) + iA)[ok], weights=wt[ok], minlength=NE * nt * (NA + 1)).reshape(NE, nt, NA + 1)
    A = np.cumsum(H, -1)[..., :NA] / sol[None, :, None]; snr2 = np.zeros(NE); wopt = np.zeros(NE); w50 = np.zeros(NE)
    for i in range(NE):
        B = np.maximum(Bw[i], 1e-30); snr2[i] = (A[i] ** 2 / B[None, :]).max(1).sum(); pool = A[i].sum(0); wopt[i] = W[np.argmax(pool / np.sqrt(B))]
        core = pool[-1]; w50[i] = W[min(np.searchsorted(pool, 0.5 * core), NA - 1)] if core > 0 else np.nan
    return dict(snr2=snr2, wopt=wopt, w50=w50, rem=rem)
Nto, Ntp = No.sum((1, 2, 3, 4, 5, 6)), Np.sum((1, 2, 3, 4, 5, 6))
def order_cut(dmin=0, sq=(2, 7), lay=(0, 1, 2, 3), ph=(0, 181)):
    m = (np.arange(10) >= (np.digitize(dmin, DEDGE) if dmin else 0)).reshape(1, 1, 10, 1, 1, 1, 1) & ((np.arange(2, 8) >= sq[0]) & (np.arange(2, 8) <= sq[1])).reshape(1, 1, 1, 6, 1, 1, 1) \
        & np.isin(np.arange(4), lay).reshape(1, 1, 1, 1, 4, 1, 1) & ((PC36 >= ph[0]) & (PC36 < ph[1])).reshape(1, 36, 1, 1, 1, 1, 1)
    m = np.broadcast_to(m, No.shape); B = (No * m).sum((2, 3, 4, 5, 6)) @ A36; rem = 1 - (No * m).sum((1, 2, 3, 4, 5, 6)) / Nto
    sm = (ev['d01'] >= dmin) & (ev['sq'] >= sq[0]) & (ev['sq'] <= sq[1]) & np.isin(lay1, lay) & (phi >= ph[0]) & (phi < ph[1]); return sm, B, rem
def psichi_cut(limit=None):
    m = np.ones(Np.shape, bool) if limit is None else np.broadcast_to((np.abs(PC72[:, None] - PC72[None, :]) < limit).reshape(1, 72, 72, 1, 1, 1, 1), Np.shape)
    B = (Np * m).sum((2, 3, 4, 5, 6)) @ A72; rem = 1 - (Np * m).sum((1, 2, 3, 4, 5, 6)) / Ntp; sm = np.ones(len(phi), bool) if limit is None else dpf < limit; return sm, B, rem
rows = []; md = []
def sweep(name, xs, labels, fn, cap=45, cap_of=None):
    out = []
    for x in xs:
        c = cap_of(x) if cap_of else cap; sm, B, rem = fn(x); base = run(c, *((np.ones(len(phi), bool),) + (psichi_cut(None)[1:] if name.startswith(('Zenith', 'Psichi')) else order_cut()[1:])))
        r = run(c, sm, B, rem); r['ch'] = 100 * (np.sqrt(r['snr2'] / base['snr2']) - 1); r['cc'] = 100 * (np.sqrt(r['snr2'].sum() / base['snr2'].sum()) - 1); r['remall'] = 100 * (1 - (r['rem'] * (Nto if not name.startswith(('Zenith', 'Psichi')) else Ntp) ).sum() * 0 - 0) if False else None; out.append(r)
    rows.append((name, xs, labels, out))
    return out
sweep('Minimum Distance [cm]', [0, 0.5, 1, 2, 3, 4, 6, 8], ['0', '0.5', '1', '2', '3', '4', '6', '8'], lambda x: order_cut(dmin=x))
sweep('Number of hits: SQ ≤ n (solid), SQ ≥ n (dashed)', [2, 3, 4, 5, 6, 7], ['2', '3', '4', '5', '6', '7+'], lambda x: order_cut(sq=(2, x)))
sweep('SQ ≥ n', [2, 3, 4, 5, 6], ['2', '3', '4', '5', '6'], lambda x: order_cut(sq=(x, 7)))
sweep('First-hit layer', [0, 1, 2, 3], ['all', 'not bottom', 'upper two', 'top only'], lambda x: order_cut(lay=[(0, 1, 2, 3), (1, 2, 3), (2, 3), (3,)][x]))
sweep('Scatter angle: φ < x [deg]', [30, 45, 60, 90, 120, 150, 180], ['30', '45', '60', '90', '120', '150', '180'], lambda x: order_cut(ph=(0, x + 0.1)))
sweep('Scatter angle: φ > x [deg]', [0, 15, 30, 45, 60, 90], ['0', '15', '30', '45', '60', '90'], lambda x: order_cut(ph=(x, 181)))
sweep('Zenith cap θ0 of the psichi cut (|ψ−φ| < θ0, transit θ < θ0) [deg]', [15, 30, 45, 60, 75, 90], ['15', '30', '45', '60', '75', '90'], lambda x: psichi_cut(x), cap_of=lambda x: x)
sweep('Psichi margin [deg] (|ψ−φ| < 45° + margin)', [0, 5, 10, 15, 20, 30, 45], ['0', '5', '10', '15', '20', '30', '45'], lambda x: psichi_cut(45 + x))
# ---- figure
groups = [('Minimum Distance [cm]', ['Minimum Distance [cm]']), ('Number of hits', ['Number of hits: SQ ≤ n (solid), SQ ≥ n (dashed)', 'SQ ≥ n']), ('First-hit layer', ['First-hit layer']),
          ('Scatter angle φ', ['Scatter angle: φ < x [deg]', 'Scatter angle: φ > x [deg]']), ('Zenith cap θ0 [deg]', ['Zenith cap θ0 of the psichi cut (|ψ−φ| < θ0, transit θ < θ0) [deg]']), ('Psichi margin [deg]', ['Psichi margin [deg] (|ψ−φ| < 45° + margin)'])]
byname = {r[0]: r for r in rows}
fig, axs = plt.subplots(len(groups), 4, figsize=(22, 4.0 * len(groups)))
for g, (title, names) in enumerate(groups):
    for k, nm in enumerate(names):
        _, xs, labels, out = byname[nm]; ls = '-' if k == 0 else '--'; xi = np.arange(len(xs))
        for i in range(NE):
            axs[g, 0].plot(xi, [r['ch'][i] for r in out], color=COL[i], ls=ls, marker='o', ms=3, lw=1.1, label=LAB[i] if (g == 0 and k == 0) else None)
            axs[g, 1].plot(xi, [100 * r['rem'][i] for r in out], color=COL[i], ls=ls, marker='o', ms=3, lw=1.1)
            axs[g, 2].plot(xi, [r['w50'][i] for r in out], color=COL[i], ls=ls, marker='o', ms=3, lw=1.1)
            axs[g, 3].plot(xi, [r['wopt'][i] for r in out], color=COL[i], ls=ls, marker='o', ms=3, lw=1.1)
        axs[g, 0].plot(xi, [r['cc'] for r in out], 'k', ls=ls, marker='s', ms=3, lw=1.6, label='all bins' if (g == 0 and k == 0) else None)
    for c, yl in enumerate(('S/√B change [%]', 'background rejected [%]', 'ARM core half-width w50 [deg]', 'optimal ARM window [deg]')):
        a = axs[g, c]; a.set_ylabel(yl); a.set_xlabel(title); a.set_xticks(np.arange(len(byname[names[0]][2]))); a.set_xticklabels(byname[names[0]][2], fontsize=8)
        if c == 0: a.axhline(0, color='k', lw=.5)
        if c >= 2: a.set_yscale('log')
axs[0, 0].legend(fontsize=7); fig.suptitle('Individual cuts (one at a time) for a transit with θ < 45° (zenith-cap row: θ < θ0): S/√B change, background rejection, signal ARM', y=0.995)
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'individual_cuts.png'), dpi=80)
# ---- table
def tbl(nm):
    _, xs, labels, out = byname[nm]; md.append(f'\n### {nm}\n'); md.append('| value | all-bin S/√B [%] | per energy bin S/√B [%] | bkg rejected per bin [%] | w50 per bin [deg] | w_opt per bin [deg] |'); md.append('|---|---|---|---|---|---|')
    for l, r in zip(labels, out): md.append(f'| {l} | {r["cc"]:+.1f} | ' + ' / '.join(f'{v:+.0f}' for v in r['ch']) + ' | ' + ' / '.join(f'{100 * v:.0f}' for v in r['rem']) + ' | ' + ' / '.join(f'{v:.1f}' for v in r['w50']) + ' | ' + ' / '.join(f'{v:.1f}' for v in r['wopt']) + ' |')
for nm in byname: tbl(nm)
open(os.path.join(outdir, 'individual_cuts.md'), 'w').write('(energy bins in order: ' + ', '.join(LAB) + ')\n' + '\n'.join(md) + '\n'); print('\n'.join(md))
