"""Effect of a Distance (first lever arm) cut with the ARM window re-optimised for each cut: window, signal and background inside it.
Usage: python plot_quality_arm_opt.py signal_dir bkg_dir outdir"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sensitivity_quality as Q
from sensitivity_quality import Cut, fom, NE

sigdir, bkgdir, outdir = sys.argv[1:4]
Hc = Q.cumulative(Q.signal_hist(Q.signal_files(sigdir))[0]); bN, bS, _ = Q.bkg_hist(bkgdir)
LAB = [f'{Q.E_EDGES[i]:.0f}–{Q.E_EDGES[i + 1]:.0f} keV' for i in range(NE)]; COL = plt.cm.viridis(np.linspace(0, 0.9, NE))
las = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8]
base = fom(Cut(), Hc, bN, bS)
R = [fom(Cut(la_min=v), Hc, bN, bS) for v in las]
w = np.array([r['w'] for r in R]); S = np.array([r['S'] for r in R]) / base['S']; B = np.array([r['B'] for r in R]) / base['B']
F = np.array([r['fom'] for r in R]) / base['fom']
# same cut but keeping the no-cut window (what a fixed-window analysis would see)
iw = [np.searchsorted(Q.W_GRID, base['w'][i]) for i in range(NE)]
Bfix = np.array([[Cut(la_min=v).apply_bkg(bS)[i] * Q.SINW[iw[i]] for i in range(NE)] for v in las]) / base['B']
Sfix = np.array([[Cut(la_min=v).apply_sig(Hc)[i, iw[i]] for i in range(NE)] for v in las]) / base['S']
fig, axs = plt.subplots(1, 4, figsize=(22, 4.8))
for i in range(NE):
    axs[0].plot(las, w[:, i], color=COL[i], marker='o', ms=3, label=LAB[i])
    axs[1].plot(las, S[:, i], color=COL[i], marker='o', ms=3); axs[1].plot(las, Sfix[:, i], color=COL[i], ls=':')
    axs[2].plot(las, B[:, i], color=COL[i], marker='o', ms=3); axs[2].plot(las, Bfix[:, i], color=COL[i], ls=':')
    axs[3].plot(las, 100 * (F[:, i] - 1), color=COL[i], marker='o', ms=3)
axs[0].set_ylabel('optimal |ARM| half-width [deg]'); axs[0].legend(fontsize=7)
axs[1].set_ylabel('signal inside the window / no-cut value'); axs[2].set_ylabel('background inside the window / no-cut value'); axs[2].set_yscale('log')
axs[3].set_ylabel('change in S/√B [%] (own optimal window)'); axs[3].axhline(0, color='k', lw=0.5)
for a in axs: a.set_xlabel('minimum Distance (first-to-second hit) [cm]')
fig.suptitle('Distance cut with the ARM window re-optimised per cut (solid) vs keeping the no-cut window (dotted)'); fig.tight_layout()
fig.savefig(os.path.join(outdir, 'quality_distance_arm_optimised.png'), dpi=100)
sel = [las.index(v) for v in (1, 2, 4)]
L = ['| cut | quantity | ' + ' | '.join(LAB) + ' |', '|' + '---|' * (NE + 2)]
for j in sel:
    for nm, a, f in (('optimal w [deg]', w, '{:g}'), ('signal in window (rel.)', S, '{:.2f}'), ('background in window (rel.)', B, '{:.2f}'), ('bkg in the no-cut window (rel.)', Bfix, '{:.2f}')):
        L.append(f'| D ≥ {las[j]:g} cm | {nm} | ' + ' | '.join(f.format(a[j, i]) for i in range(NE)) + ' |')
open(os.path.join(outdir, 'quality_distance_arm_optimised.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
