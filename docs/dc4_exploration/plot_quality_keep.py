"""(1) S/sqrt(B) when keeping events with AT LEAST n hits; (2) best cut combination that removes >= a given fraction (default 90%) of the background events.
Usage: python plot_quality_keep.py signal_dir bkg_dir outdir [removed_fraction]"""
import sys, os, itertools
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sensitivity_quality as Q
from sensitivity_quality import Cut, fom, combined, NE

sigdir, bkgdir, outdir = sys.argv[1:4]
REM = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
Hfile = [Q.signal_hist([f])[0] for f in Q.signal_files(sigdir)]; Htot = sum(Hfile); Hc = Q.cumulative(Htot)
bN, bS, per = Q.bkg_hist(bkgdir)
base = fom(Cut(), Hc, bN, bS); N0 = base['Bn']
LAB = [f'{Q.E_EDGES[i]:.0f}–{Q.E_EDGES[i + 1]:.0f} keV' for i in range(NE)]; COL = plt.cm.viridis(np.linspace(0, 0.9, NE))
md = []
def table(t, rows, h):
    md.append(f'\n### {t}\n'); md.append('| ' + ' | '.join(h) + ' |'); md.append('|' + '---|' * len(h)); md.extend('| ' + ' | '.join(r) + ' |' for r in rows)
def ev(cut, Hcc=Hc):
    r = fom(cut, Hcc, bN, bS); r['ch'] = 100 * (r['fom'] / base['fom'] - 1); r['cc'] = 100 * (combined(r['fom']) / combined(base['fom']) - 1)
    r['rem'] = 1 - r['Bn'] / N0; r['remtot'] = 1 - r['Bn'].sum() / N0.sum(); r['sig'] = r['Sall'] / base['Sall']; return r

# ---- 1. at least n hits
mins = [2, 3, 4, 5, 6, 7]
R = [ev(Cut(sq_min=n)) for n in mins]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for i in range(NE):
    axs[0].plot(mins, [r['ch'][i] for r in R], color=COL[i], marker='o', ms=3, label=LAB[i])
    axs[1].plot([100 * r['rem'][i] for r in R], [r['ch'][i] for r in R], color=COL[i], marker='o', ms=3)
axs[0].plot(mins, [r['cc'] for r in R], 'k--', marker='s', ms=3, label='all bins'); axs[0].axhline(0, color='k', lw=.5); axs[0].legend(fontsize=7)
axs[0].set_xlabel('minimum number of hits (keep SQ ≥ n)'); axs[0].set_ylabel('change in S/√B [%] (ARM window re-optimised)')
axs[1].set_xlabel('background events removed [%]'); axs[1].set_ylabel('change in S/√B [%]'); axs[1].axhline(0, color='k', lw=.5)
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'keep_min_hits.png'), dpi=100); plt.close(fig)
table('Keep events with at least n hits: change in S/√B [%] (window re-optimised)',
      [[f'SQ ≥ {n}'] + [f'{R[j]["ch"][i]:+.1f}' for i in range(NE)] + [f'{R[j]["cc"]:+.1f}', f'{100 * R[j]["remtot"]:.0f}', f'{100 * R[j]["sig"].mean():.0f}'] for j, n in enumerate(mins)],
      ['cut'] + LAB + ['all bins', 'bkg events removed [%]', 'mean signal kept [%]'])

# ---- 2. grid search with a removal constraint
Ds = [0, 0.5, 1, 2, 3, 4, 6, 8]
SQ = [(n, 7) for n in range(2, 7)] + [(2, m) for m in (3, 4, 5, 6)] + [(n, n) for n in range(3, 7)] + [(3, 4), (3, 5), (4, 5), (4, 6)]
LY = [('all', (0, 1, 2, 3)), ('≥2', (1, 2, 3)), ('≥3', (2, 3)), ('top', (3,)), ('≤3 (no top)', (0, 1, 2)), ('2 or 3', (1, 2))]
DQ = [('none', (), ()), ('mid 1&2', (1,), (1,)), ('mid+surf 1&2', (1, 2), (1, 2))]
combos = list(itertools.product(Ds, SQ, LY, DQ))
def mk(c): return Cut(la_min=c[0], sq_min=c[1][0], sq_max=c[1][1], layers1=c[2][1], excl1=c[3][1], excl2=c[3][2])
name = lambda c: f'D ≥ {c[0]:g} cm, SQ {c[1][0]}–{c[1][1] if c[1][1] < 7 else "7+"}, 1st-hit layer {c[2][0]}, depth: {c[3][0]}'
res = [ev(mk(c)) for c in combos]
rem = np.array([r['rem'] for r in res]); remtot = np.array([r['remtot'] for r in res]); ch = np.array([r['ch'] for r in res]); cc = np.array([r['cc'] for r in res]); sig = np.array([r['sig'] for r in res])
def pct(x): return f'{100 * x:.0f}'
# a) overall constraint (total background events 200-5000 keV), quadrature FoM
ok = remtot >= REM; order = np.argsort(-np.where(ok, cc, -1e9))
table(f'Best combinations removing ≥ {100 * REM:.0f}% of all background events (200-5000 keV), ranked by the all-bin S/√B change',
      [[name(combos[n]), f'{cc[n]:+.1f}', pct(remtot[n]), pct(sig[n].mean())] + [f'{ch[n, i]:+.1f}' for i in range(NE)] for n in order[:10] if ok[n]],
      ['combination', 'all bins [%]', 'bkg removed [%]', 'mean signal kept [%]'] + LAB)
# b) constraint per energy bin
rows = []
for i in range(NE):
    okb = rem[:, i] >= REM; n = int(np.argmax(np.where(okb, ch[:, i], -1e9)))
    rows.append([LAB[i], name(combos[n]) if okb[n] else 'none reaches it', f'{ch[n, i]:+.1f}', pct(rem[n, i]), pct(sig[n, i]), f'{res[n]["w"][i]:g}'])
table(f'Best combination per energy bin removing ≥ {100 * REM:.0f}% of that bin\'s background events', rows, ['bin', 'combination', 'S/√B change [%]', 'bkg removed [%]', 'signal kept [%]', 'optimal w [deg]'])
# c) trade-off curve: best change vs fraction removed
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
levels = np.arange(0.0, 0.995, 0.01); best = [np.where(remtot >= l, cc, -np.inf).max() if (remtot >= l).any() else np.nan for l in levels]
axs[0].scatter(100 * remtot, cc, s=4, alpha=.25); axs[0].plot(100 * levels, best, 'r-', label='best at each removal level'); axs[0].axvline(100 * REM, color='k', ls=':')
axs[0].set_xlabel('background events removed [%] (200-5000 keV)'); axs[0].set_ylabel('all-bin S/√B change [%]'); axs[0].legend(); axs[0].set_ylim(-100, 10)
for i in range(NE):
    bb = [np.where(rem[:, i] >= l, ch[:, i], -np.inf).max() if (rem[:, i] >= l).any() else np.nan for l in levels]; axs[1].plot(100 * levels, bb, color=COL[i], label=LAB[i])
axs[1].axvline(100 * REM, color='k', ls=':'); axs[1].set_xlabel('background events removed in the bin [%]'); axs[1].set_ylabel('best S/√B change [%]'); axs[1].legend(fontsize=7); axs[1].set_ylim(-100, 10)
fig.suptitle('Best S/√B change achievable for a given background reduction (all cut combinations, ARM window re-optimised)'); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'keep_tradeoff.png'), dpi=100); plt.close(fig)
rows = [[f'{100 * l:.0f}%', f'{best[int(round(l * 100))]:+.1f}'] for l in (0.5, 0.7, 0.8, 0.9, 0.95)]
table('Best all-bin S/√B change at a given overall background removal', rows, ['removed ≥', 'best change [%]'])
# d) jackknife error of the winner
n = order[0]; c = combos[n]; jk = []
for h in Hfile:
    Hj = Q.cumulative(Htot - h); b0 = fom(Cut(), Hj, bN, bS); jk.append(100 * (combined(fom(mk(c), Hj, bN, bS)['fom']) / combined(b0['fom']) - 1))
jk = np.array(jk); md.append(f'\nJackknife error of the best all-bin value ({name(c)}): {cc[n]:+.1f} ± {np.sqrt(9 / 10 * ((jk - jk.mean()) ** 2).sum()):.1f} %')
open(os.path.join(outdir, 'keep_summary.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
