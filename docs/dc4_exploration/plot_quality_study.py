"""Figures and summary tables of the S/sqrt(B) vs event-quality-cut study (see sensitivity_quality.py).

Usage: python plot_quality_study.py signal_dir bkg_dir outdir
Writes quality_*.png and quality_summary.md into outdir.
"""
import sys, os, re, itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sensitivity_quality as Q
from sensitivity_quality import Cut, fom, combined, NE, W_GRID

sigdir, bkgdir, outdir = sys.argv[1:4]
os.makedirs(outdir, exist_ok=True)
LAB = [f'{Q.E_EDGES[i]:.0f}–{Q.E_EDGES[i + 1]:.0f} keV' for i in range(NE)]
COL = plt.cm.viridis(np.linspace(0, 0.9, NE))
md = []                                   # lines of the markdown summary

files = Q.signal_files(sigdir)
Hfile = [Q.signal_hist([f])[0] for f in files]
Htot = sum(Hfile)
Hc = Q.cumulative(Htot)
bN, bS, per = Q.bkg_hist(bkgdir)
base = fom(Cut(), Hc, bN, bS)
nsig = len(files)
# jackknife (leave one signal file out) cumulative histograms
Hc_jk = [Q.cumulative(Htot - h) for h in Hfile]


def change(cut, with_err=True):
    """% change of S/sqrt(B) (best ARM window re-optimised) vs the no-cut baseline: per bin (NE) and combined; with jackknife errors."""
    r = fom(cut, Hc, bN, bS)
    ch = 100 * (r['fom'] / base['fom'] - 1); cc = 100 * (combined(r['fom']) / combined(base['fom']) - 1)
    out = dict(r=r, ch=ch, cc=cc, removed=1 - r['Bn'] / np.maximum(base['Bn'], 1e-30), sig_eff=r['Sall'] / base['Sall'])
    if with_err:
        jk = []; jc = []
        for h in Hc_jk:
            b0 = fom(Cut(), h, bN, bS); rj = fom(cut, h, bN, bS)
            jk.append(100 * (rj['fom'] / b0['fom'] - 1)); jc.append(100 * (combined(rj['fom']) / combined(b0['fom']) - 1))
        jk = np.array(jk); jc = np.array(jc)
        out['err'] = np.sqrt((nsig - 1) / nsig * ((jk - jk.mean(0)) ** 2).sum(0)); out['cerr'] = np.sqrt((nsig - 1) / nsig * ((jc - jc.mean()) ** 2).sum())
    return out


def table(title, rows, header):
    md.append(f'\n### {title}\n'); md.append('| ' + ' | '.join(header) + ' |'); md.append('|' + '---|' * len(header))
    for r in rows:
        md.append('| ' + ' | '.join(r) + ' |')


fmt = lambda c, e=None: f'{c:+.1f}' + (f' ± {e:.1f}' if e is not None else '')

# ---------------------------------------------------------------- 0. validation against the Crab study
crab_gain = [16, 25, 43, 56, 38, 8]; crab_w = [10, 6, 4, 3, 3, 8]
gain_win = 100 * (base['fom'] / base['fom_nowindow'] - 1)
sigw = np.array([(Htot[i].sum((0, 1, 2, 3))[:Q.NA] if False else 0) for i in range(NE)])
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
x = np.arange(NE)
axs[0].bar(x - 0.2, gain_win, 0.4, label='this study (sim. signal, sky-averaged)'); axs[0].bar(x + 0.2, crab_gain, 0.4, label='Crab study (real Crab + total BG)')
axs[0].set_xticks(x); axs[0].set_xticklabels(LAB, rotation=30, fontsize=8); axs[0].set_ylabel('S/√B gain from the optimal ARM window [%]'); axs[0].legend(fontsize=8)
axs[1].bar(x - 0.2, base['w'], 0.4, label='this study'); axs[1].bar(x + 0.2, crab_w, 0.4, label='Crab study')
axs[1].set_xticks(x); axs[1].set_xticklabels(LAB, rotation=30, fontsize=8); axs[1].set_ylabel('optimal |ARM| half-width [deg]'); axs[1].legend(fontsize=8)
# signal ARM distribution per bin (no cut)
tot_arm = Htot.sum((1, 2, 3, 4))                     # (NE, NA+1)
for i in range(NE):
    h = tot_arm[i, :Q.NA]; axs[2].step(np.r_[0, W_GRID][:-1] + 0.25, h / h.sum() / Q.DA, where='mid', color=COL[i], label=LAB[i])
axs[2].set_yscale('log'); axs[2].set_xlabel('|ARM| [deg]'); axs[2].set_ylabel('signal density (weighted, per deg, within 30°)'); axs[2].legend(fontsize=7)
fig.suptitle('Validation of the sky-averaged simulation (true-direction ARM) and of the sin(φ) background estimate against the Crab study')
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'quality_validation.png'), dpi=100); plt.close(fig)
table('Validation: gain of the optimal ARM window over no window (no quality cut)',
      [[LAB[i], f'{gain_win[i]:+.0f}', f'{crab_gain[i]:+d}', f'{base["w"][i]:g}', f'{crab_w[i]:g}'] for i in range(NE)],
      ['energy bin', 'this study [%]', 'Crab study [%]', 'optimal w [deg] (this)', 'optimal w [deg] (Crab)'])

# ---------------------------------------------------------------- helper for scans
def scan_plot(cuts, xvals, xlabel, fname, title, xticklabels=None, logx=False):
    res = [change(c) for c in cuts]
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.9))
    for i in range(NE):
        y = np.array([r['ch'][i] for r in res]); e = np.array([r['err'][i] for r in res])
        axs[0].errorbar(xvals, y, e, color=COL[i], marker='o', ms=3, lw=1, capsize=1.5, label=LAB[i])
    axs[0].errorbar(xvals, [r['cc'] for r in res], [r['cerr'] for r in res], color='k', ls='--', marker='s', ms=3, label='all bins (quadrature)')
    axs[0].axhline(0, color='k', lw=0.5); axs[0].set_xlabel(xlabel); axs[0].set_ylabel('change in S/√B [%] (ARM window re-optimised)'); axs[0].legend(fontsize=7)
    if xticklabels: axs[0].set_xticks(xvals); axs[0].set_xticklabels(xticklabels)
    for i in range(NE):
        axs[1].plot([100 * r['removed'][i] for r in res], [r['ch'][i] for r in res], color=COL[i], marker='o', ms=3, lw=1)
    axs[1].axhline(0, color='k', lw=0.5); axs[1].set_xlabel('background events removed [%]'); axs[1].set_ylabel('change in S/√B [%]')
    for i in range(NE):
        axs[2].plot(xvals, [r['r']['w'][i] for r in res], color=COL[i], marker='o', ms=3, lw=1)
    axs[2].set_xlabel(xlabel); axs[2].set_ylabel('optimal |ARM| half-width [deg]')
    if xticklabels: axs[2].set_xticks(xvals); axs[2].set_xticklabels(xticklabels)
    if logx:
        axs[0].set_xscale('log'); axs[2].set_xscale('log')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(os.path.join(outdir, fname), dpi=100); plt.close(fig)
    return res

# ---------------------------------------------------------------- 1. Distance (lever arm LA)
las = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
res_la = scan_plot([Cut(la_min=v) for v in las], las, 'minimum Distance (lever arm LA = minimum consecutive hop) [cm]', 'quality_distance.png',
                   'S/√B vs the minimum Distance cut (ARM window re-optimised), sky-averaged signal vs week-1 background')
sel = [las.index(v) for v in (0.5, 1, 2, 5, 8)]
table('S/√B change vs minimum Distance (LA), ARM window re-optimised [%] (± jackknife over the 10 signal files); last rows: all bins in quadrature and background removed',
      [[f'{las[j]:g} cm'] + [fmt(res_la[j]['ch'][i], res_la[j]['err'][i]) for i in range(NE)] + [fmt(res_la[j]['cc'], res_la[j]['cerr']), f'{100 * res_la[j]["removed"].mean():.0f}'] for j in sel],
      ['cut'] + LAB + ['all bins', 'mean bkg removed [%]'])

# ---------------------------------------------------------------- 2. number of hits
nh = [2, 3, 4, 5, 6, 7]
res_nmax = scan_plot([Cut(sq_max=n) for n in nh], nh, 'maximum number of hits (keep SQ ≤ n)', 'quality_nhits_max.png',
                      'S/√B vs the maximum number of hits (ARM window re-optimised)', xticklabels=['2', '3', '4', '5', '6', '7+ (no cut)'])
res_neq = scan_plot([Cut(sq_eq=n) for n in nh], nh, 'exactly n hits (SQ = n)', 'quality_nhits_exact.png',
                    'S/√B for events with exactly n hits (ARM window re-optimised)', xticklabels=['2', '3', '4', '5', '6', '7+'])
table('S/√B change vs the maximum number of hits (SQ ≤ n) [%]',
      [[f'SQ ≤ {n}'] + [fmt(res_nmax[j]['ch'][i], res_nmax[j]['err'][i]) for i in range(NE)] + [fmt(res_nmax[j]['cc'], res_nmax[j]['cerr']), f'{100 * res_nmax[j]["removed"].mean():.0f}', f'{100 * res_nmax[j]["sig_eff"].mean():.0f}'] for j, n in enumerate(nh[:-1])],
      ['cut'] + LAB + ['all bins', 'mean bkg removed [%]', 'mean signal kept [%]'])
table('S/√B change for events with exactly n hits [%]',
      [[f'SQ = {n}'] + [fmt(res_neq[j]['ch'][i], res_neq[j]['err'][i]) for i in range(NE)] + [fmt(res_neq[j]['cc'], res_neq[j]['cerr'])] for j, n in enumerate(nh)],
      ['cut'] + LAB + ['all bins'])

# ---------------------------------------------------------------- 3. first-hit depth: layer and depth quality
opts = [('all layers', Cut()), ('layers 2-4 (not the bottom)', Cut(layers1=(1, 2, 3))), ('layers 3-4', Cut(layers1=(2, 3))), ('top layer only', Cut(layers1=(3,))),
        ('bottom layer only', Cut(layers1=(0,))), ('layer 2 only', Cut(layers1=(1,))), ('layer 3 only', Cut(layers1=(2,))),
        ('excl. middle-plane 1st hit', Cut(excl1=(1,))), ('excl. middle-plane & surface 1st hit', Cut(excl1=(1, 2))),
        ('excl. middle-plane 1st or 2nd hit', Cut(excl1=(1,), excl2=(1,))), ('excl. middle-plane & surface, 1st or 2nd hit', Cut(excl1=(1, 2), excl2=(1, 2)))]
res_z = [change(c) for _, c in opts]
fig, axs = plt.subplots(1, 2, figsize=(17, 6))
w = 0.8 / NE
for i in range(NE):
    axs[0].barh(np.arange(len(opts)) + i * w - 0.4 + w / 2, [r['ch'][i] for r in res_z], w, xerr=[r['err'][i] for r in res_z], color=COL[i], label=LAB[i])
axs[0].set_yticks(np.arange(len(opts))); axs[0].set_yticklabels([o[0] for o in opts], fontsize=8); axs[0].invert_yaxis(); axs[0].axvline(0, color='k', lw=0.5)
axs[0].set_xlabel('change in S/√B [%] (ARM window re-optimised)'); axs[0].legend(fontsize=7)
axs[1].barh(np.arange(len(opts)), [r['cc'] for r in res_z], 0.5, xerr=[r['cerr'] for r in res_z], color='k', label='all bins')
axs[1].barh(np.arange(len(opts)) + 0.3, [100 * r['removed'].mean() for r in res_z], 0.25, color='C3', label='mean background removed [%]')
axs[1].barh(np.arange(len(opts)) + 0.55, [100 * (1 - r['sig_eff'].mean()) for r in res_z], 0.25, color='C0', label='mean signal removed [%]')
axs[1].set_yticks([]); axs[1].invert_yaxis(); axs[1].axvline(0, color='k', lw=0.5); axs[1].legend(fontsize=8); axs[1].set_xlabel('[%]')
fig.suptitle('S/√B vs the depth (layer, middle-plane / surface flags) of the first and second hit'); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'quality_depth.png'), dpi=100); plt.close(fig)
table('S/√B change vs first-hit layer and depth quality [%]',
      [[opts[j][0]] + [fmt(res_z[j]['ch'][i], res_z[j]['err'][i]) for i in range(NE)] + [fmt(res_z[j]['cc'], res_z[j]['cerr']), f'{100 * res_z[j]["removed"].mean():.0f}', f'{100 * (1 - res_z[j]["sig_eff"].mean()):.0f}'] for j in range(len(opts))],
      ['cut'] + LAB + ['all bins', 'mean bkg removed [%]', 'mean signal removed [%]'])

# ---------------------------------------------------------------- 4. combined cuts: exhaustive grid
LAs = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8]
SQs = [2, 3, 4, 5, 6, 7]
LAY = [('all', (0, 1, 2, 3)), ('>=2', (1, 2, 3)), ('>=3', (2, 3)), ('top', (3,))]
DQ = [('none', (), ()), ('1st mid', (1,), ()), ('1st+2nd mid', (1,), (1,)), ('1st mid+surf', (1, 2), ()), ('1st+2nd mid+surf', (1, 2), (1, 2))]
combos = list(itertools.product(LAs, SQs, LAY, DQ))
tab = np.zeros((len(combos), NE)); comb = np.zeros(len(combos)); rem = np.zeros((len(combos), NE)); wopt = np.zeros((len(combos), NE))
for n, (la, sq, lay, dq) in enumerate(combos):
    r = fom(Cut(la_min=la, sq_max=sq, layers1=lay[1], excl1=dq[1], excl2=dq[2]), Hc, bN, bS)
    tab[n] = 100 * (r['fom'] / base['fom'] - 1); comb[n] = 100 * (combined(r['fom']) / combined(base['fom']) - 1); rem[n] = 1 - r['Bn'] / base['Bn']; wopt[n] = r['w']
name = lambda c: f'LA ≥ {c[0]:g}, SQ ≤ {c[1]}, 1st-hit layer {c[2][0]}, depth: {c[3][0]}'
order = np.argsort(-comb)
table('Best combinations over all 1200 (all bins in quadrature, one common cut for all energies)',
      [[name(combos[n]), f'{comb[n]:+.1f}'] + [f'{tab[n, i]:+.1f}' for i in range(NE)] for n in order[:8]], ['combination', 'all bins'] + LAB)
best_bin = [int(np.argmax(tab[:, i])) for i in range(NE)]
table('Best combination for each energy bin separately (cuts could be energy dependent)',
      [[LAB[i], name(combos[best_bin[i]]), f'{tab[best_bin[i], i]:+.1f}', f'{100 * rem[best_bin[i], i]:.0f}', f'{wopt[best_bin[i], i]:g}'] for i in range(NE)],
      ['energy bin', 'combination', 'S/√B change [%]', 'bkg removed [%]', 'optimal w [deg]'])
# heatmaps LA x max hits, no depth cut
fig, axs = plt.subplots(2, 3, figsize=(17, 8.5))
for i, ax in enumerate(axs.flat):
    M = np.array([[change(Cut(la_min=la, sq_max=sq), with_err=False)['ch'][i] for la in LAs] for sq in SQs])
    lim = np.abs(M).max(); im = ax.pcolormesh(np.arange(len(LAs) + 1) - .5, np.arange(len(SQs) + 1) - .5, M, cmap='RdBu_r', vmin=-lim, vmax=lim)
    j, k = np.unravel_index(M.argmax(), M.shape); ax.plot(k, j, 'k*', ms=12)
    ax.set_xticks(range(len(LAs))); ax.set_xticklabels([f'{v:g}' for v in LAs], fontsize=7); ax.set_yticks(range(len(SQs))); ax.set_yticklabels(['7+' if s == 7 else str(s) for s in SQs])
    ax.set_xlabel('minimum Distance [cm]'); ax.set_ylabel('maximum number of hits'); ax.set_title(f'{LAB[i]}: best {M.max():+.1f}%'); fig.colorbar(im, ax=ax)
fig.suptitle('S/√B change (%) vs Distance and maximum number of hits (ARM window re-optimised, no depth cut)'); fig.tight_layout()
fig.savefig(os.path.join(outdir, 'quality_distance_x_nhits.png'), dpi=95); plt.close(fig)
# Pareto: gain vs background removed, all combos
fig, axs = plt.subplots(2, 3, figsize=(17, 8.5))
for i, ax in enumerate(axs.flat):
    ax.scatter(100 * rem[:, i], tab[:, i], s=3, alpha=0.3, color=COL[i]); ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('background removed [%]'); ax.set_ylabel('S/√B change [%]'); ax.set_title(LAB[i])
fig.suptitle('All 1200 cut combinations: S/√B change vs fraction of background removed'); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'quality_pareto.png'), dpi=95); plt.close(fig)

# ---------------------------------------------------------------- 5. ARM width vs cuts (signal only)
def w68(cut):
    S = cut.apply_sig(Hc).astype(np.float64); inside = S[:, Q.NA - 1]            # events within 30 deg
    return np.array([W_GRID[np.searchsorted(S[i, :Q.NA], 0.68 * inside[i])] for i in range(NE)])
cuts_arm = [('no cut', Cut()), ('LA ≥ 1 cm', Cut(la_min=1)), ('LA ≥ 2 cm', Cut(la_min=2)), ('LA ≥ 4 cm', Cut(la_min=4)), ('SQ ≤ 3', Cut(sq_max=3)), ('SQ = 2', Cut(sq_eq=2)),
            ('excl. middle-plane/surface hits', Cut(excl1=(1, 2), excl2=(1, 2))), ('top layer 1st hit', Cut(layers1=(3,)))]
fig, ax = plt.subplots(figsize=(11, 5))
for j, (nm, c) in enumerate(cuts_arm):
    ax.plot(range(NE), w68(c), marker='o', label=nm)
ax.set_xticks(range(NE)); ax.set_xticklabels(LAB, rotation=20); ax.set_ylabel('|ARM| 68% containment (of events within 30°) [deg]'); ax.legend(fontsize=8)
ax.set_title('Signal ARM width vs quality cut'); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'quality_arm_width.png'), dpi=100); plt.close(fig)
table('Signal |ARM| 68% containment [deg] (of the events within 30 deg)', [[nm] + [f'{v:g}' for v in w68(c)] for nm, c in cuts_arm], ['cut'] + LAB)

# ---------------------------------------------------------------- 6. background composition and retention per component
comps = sorted(per)
cut_ref = [('no cut', Cut()), ('LA ≥ 1 cm', Cut(la_min=1)), ('SQ ≤ 3', Cut(sq_max=3)), ('best combined cut', Cut(la_min=combos[order[0]][0], sq_max=combos[order[0]][1], layers1=combos[order[0]][2][1], excl1=combos[order[0]][3][1], excl2=combos[order[0]][3][2]))]
fig, axs = plt.subplots(1, len(cut_ref), figsize=(5.2 * len(cut_ref), 5.6), sharey=True)
rows = []
for ax, (nm, c) in zip(axs, cut_ref):
    shares = np.array([c.apply_bkg(per[k][1]) for k in comps])            # sum sin(phi) per component, bin
    shares = shares / shares.sum(0)
    bottom = np.zeros(NE)
    for j, k in enumerate(comps):
        ax.bar(range(NE), shares[j], bottom=bottom, label=k); bottom += shares[j]
    ax.set_xticks(range(NE)); ax.set_xticklabels(LAB, rotation=40, fontsize=7); ax.set_title(nm)
    for i in (1, 3, 5):
        rows.append([nm, LAB[i]] + [f'{100 * shares[j, i]:.0f}' for j in range(len(comps))])
axs[0].set_ylabel('share of the background in the ARM window'); axs[-1].legend(fontsize=6, loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'quality_background_composition.png'), dpi=100); plt.close(fig)
table('Share of the background (sum of sin φ) by component [%] for 3 energy bins', rows, ['cut', 'bin'] + comps)
ret = [[k] + [f'{100 * (c.apply_bkg(per[k][0]) / np.maximum(Cut().apply_bkg(per[k][0]), 1)).mean():.0f}' for _, c in cut_ref[1:]] for k in comps]
table('Mean (over energy bins) fraction of each background component kept by the cut [%]', ret, ['component'] + [n for n, _ in cut_ref[1:]])

# ---------------------------------------------------------------- 7. robustness of the best combined cut and of single cuts
bestc = combos[order[0]]
cb = Cut(la_min=bestc[0], sq_max=bestc[1], layers1=bestc[2][1], excl1=bestc[3][1], excl2=bestc[3][2])
rows = []
for lab, kw in (('default: weights E_true^-2.1, all directions', {}), ('weights E_true^-1.5', dict(gamma=1.5)), ('weights E_true^-3', dict(gamma=3.0)),
                ('source within 60 deg of the +z axis', dict(theta_max=60.0))):
    Hv, _ = Q.signal_hist(files, **kw); Hvc = Q.cumulative(Hv); b0 = fom(Cut(), Hvc, bN, bS)
    r1 = fom(Cut(la_min=1), Hvc, bN, bS); r2 = fom(Cut(sq_max=3), Hvc, bN, bS); r3 = fom(cb, Hvc, bN, bS)
    rows.append([lab, f'{100 * (combined(r1["fom"]) / combined(b0["fom"]) - 1):+.1f}', f'{100 * (combined(r2["fom"]) / combined(b0["fom"]) - 1):+.1f}',
                 f'{100 * (combined(r3["fom"]) / combined(b0["fom"]) - 1):+.1f}', ' '.join(f'{v:g}' for v in b0['w'])])
table(f'Robustness (all bins in quadrature, % change): best combined cut = {name(bestc)}', rows,
      ['signal variant', 'LA ≥ 1 cm', 'SQ ≤ 3', 'best combined', 'optimal w of the baseline per bin [deg]'])

open(os.path.join(outdir, 'quality_summary.md'), 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
