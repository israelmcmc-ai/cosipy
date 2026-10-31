"""Signal |ARM| width and tail fraction versus event characteristics (response simulation, weights E_true^-2.1), within each measured-energy bin.
Core width = half-width containing 50% of the events whose |ARM| < 30 deg (w50); tail = weighted fraction with |ARM| > 30 deg.
Usage: python arm_vs_characteristics.py signal_dir outdir"""
import sys, glob, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tra_quality import E_EDGES
sigdir, outdir = sys.argv[1:3]
D = {}
for f in sorted(glob.glob(sigdir + '/*.npz')):
    z = np.load(f)
    for k in ('E', 'phi', 'z1', 'z2', 'sq', 'theta_src', 'Etrue', 'cat1', 'cat2', 'd01', 'arm'): D.setdefault(k, []).append(z[k])
D = {k: np.concatenate(v) for k, v in D.items()}
w = D['Etrue'] ** -2.1; a = np.abs(D['arm']); ie = np.digitize(D['E'], E_EDGES) - 1
phi = np.rad2deg(D['phi']); dz = D['z2'] - D['z1']
def w50(m):
    x = a[m & (a < 30)]; ww = w[m & (a < 30)]
    if ww.sum() == 0: return np.nan
    o = np.argsort(x); c = np.cumsum(ww[o]); return x[o][np.searchsorted(c, 0.5 * c[-1])]
def tail(m): return w[m & (a >= 30)].sum() / w[m].sum()
chars = {
 'distance [cm]': (D['d01'], [0, 0.5, 1, 2, 4, 8, 1e9], ['<0.5', '0.5-1', '1-2', '2-4', '4-8', '>8']),
 'number of hits': (D['sq'], [1.5, 2.5, 3.5, 4.5, 99], ['2', '3', '4', '5+']),
 'first-hit layer': (D['cat1'] // 3, [-.5, .5, 1.5, 2.5, 3.5], ['bottom', '2', '3', 'top']),
 'second-hit layer': (D['cat2'] // 3, [-.5, .5, 1.5, 2.5, 3.5], ['bottom', '2', '3', 'top']),
 'first-hit depth type': (D['cat1'] % 3, [-.5, .5, 1.5, 2.5], ['bulk', 'mid-plane', 'surface']),
 'second-hit depth type': (D['cat2'] % 3, [-.5, .5, 1.5, 2.5], ['bulk', 'mid-plane', 'surface']),
 'scatter angle phi [deg]': (phi, [0, 30, 60, 90, 120, 180.1], ['<30', '30-60', '60-90', '90-120', '>120']),
 'hit 2 below hit 1 (dz<0)': ((dz < 0).astype(float), [-.5, .5, 1.5], ['no (up)', 'yes (down)']),
 'source polar angle theta [deg] (not observable)': (D['theta_src'], [0, 30, 60, 90, 180.1], ['<30', '30-60', '60-90', '>90']),
}
base50 = np.array([w50(ie == i) for i in range(6)]); baseT = np.array([tail(ie == i) for i in range(6)])
md = ['| characteristic | value | share of events [%] | core width w50 relative to all (mean over energy bins) | w50 range over bins [deg] | tail fraction (|ARM|>30°) relative |', '|---|---|---|---|---|---|']
spread = []; rows = []
for nm, (x, edges, labs) in chars.items():
    r = []
    for j, lab in enumerate(labs):
        m = (x >= edges[j]) & (x < edges[j + 1]); n = np.array([w50(m & (ie == i)) for i in range(6)]); t = np.array([tail(m & (ie == i)) if w[m & (ie == i)].sum() > 0 else np.nan for i in range(6)])
        rel = np.nanmean(n / base50); relt = np.nanmean(t / baseT); share = 100 * w[m & (ie >= 0) & (ie < 6)].sum() / w[(ie >= 0) & (ie < 6)].sum()
        r.append((lab, share, rel, relt, np.nanmin(n), np.nanmax(n)))
        md.append(f'| {nm} | {lab} | {share:.0f} | {rel:.2f} | {np.nanmin(n):.1f}–{np.nanmax(n):.1f} | {relt:.2f} |')
    spread.append((max(q[2] for q in r) - min(q[2] for q in r), nm, r))
spread.sort(reverse=True)
md = ['Ranking of the characteristics by how much they change the core width (largest spread first): ' + ', '.join(f'{nm} ({sp:.2f})' for sp, nm, _ in spread), ''] + md
open(os.path.join(outdir, 'arm_vs_characteristics.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
fig, ax = plt.subplots(figsize=(10, 7)); y = 0; yt = []; yl = []
for sp, nm, r in spread:
    for lab, share, rel, relt, lo, hi in r:
        ax.barh(y, rel - 1, color='C3' if rel > 1 else 'C2'); yt.append(y); yl.append(f'{nm}: {lab}'); y += 1
    y += 0.7
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6); ax.invert_yaxis(); ax.axvline(0, color='k', lw=.5); ax.set_xlabel('core width w50 relative to the average (negative = better ARM)'); fig.tight_layout()
fig.savefig(os.path.join(outdir, 'arm_vs_characteristics.png'), dpi=100)
