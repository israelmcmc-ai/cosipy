"""Signal ARM versus Distance bin with no other cuts (sphere-averaged background, all source directions).
Usage: python plot_arm_vs_distance_nocut.py signal_dir bkgorder_dir outdir"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from order_common import *

DL = ['<0.5', '0.5–1', '1–2', '2–3', '3–4', '4–6', '6–8', '8–12', '12–20', '>20']
w50 = np.full((NE, ND), np.nan); w68 = np.full((NE, ND), np.nan); wopt = np.full((NE, ND), np.nan); tail = np.full((NE, ND), np.nan); share = np.zeros((NE, ND))
for i in range(NE):
    tot = Sc_all[i][..., -1].sum()
    for j in range(ND):
        cum = Sc_all[i][:, j].reshape(-1, NA + 1).sum(0); core = cum[NA - 1]; share[i, j] = cum[-1] / tot
        if core <= 0: continue
        w50[i, j] = W[np.searchsorted(cum[:NA], 0.5 * core)]; w68[i, j] = W[min(np.searchsorted(cum[:NA], 0.68 * core), NA - 1)]; tail[i, j] = 1 - core / cum[-1]
        B = (Nb[i][:, j]).sum((1, 2, 3, 4)) @ AREA; wopt[i, j] = W[np.argmax(cum[:NA] / np.sqrt(np.maximum(B, 1e-30)))]
fig, axs = plt.subplots(1, 4, figsize=(22, 4.8)); x = np.arange(ND)
for a, (v, yl, log) in zip(axs, ((w50, 'core half-width w50 [deg]', True), (w68, 'half-width w68 [deg]', True), (wopt, 'optimal S/√B window [deg]', True), (tail, 'fraction with |ARM| > 30°', False))):
    for i in range(NE): a.plot(x, v[i], color=COL[i], marker='o', ms=3, label=LAB[i])
    a.set_xticks(x); a.set_xticklabels(DL, rotation=45, fontsize=8); a.set_xlabel('Distance [cm]'); a.set_ylabel(yl)
    if log: a.set_yscale('log')
axs[0].legend(fontsize=7); fig.suptitle('Signal ARM vs Distance, no other cuts (all source directions, E^-2.1 source weighting, sphere-averaged background)', fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'arm_vs_distance_nocut.png'), dpi=100)
md = ['| energy bin | quantity | ' + ' | '.join(DL) + ' |', '|' + '---|' * (ND + 2)]
for i in range(NE):
    for nm, v in (('w50 [deg]', w50), ('w_opt [deg]', wopt), ('tail', tail), ('share of events [%]', 100 * share)): md.append(f'| {LAB[i]} | {nm} | ' + ' | '.join(f'{u:.2f}' if nm == 'tail' else f'{u:.1f}' for u in v[i]) + ' |')
open(os.path.join(outdir, 'arm_vs_distance_nocut.md'), 'w').write('\n'.join(md) + '\n')
