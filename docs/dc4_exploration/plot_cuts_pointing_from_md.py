"""Redraw cuts_pointing_ref<N>.png without the all-bins line, from the committed tables (figures_quality/cuts_pointing_summary_ref<N>.md).
Usage: python plot_cuts_pointing_from_md.py [ref=180] [outdir=figures_quality]"""
import sys, os, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ref = sys.argv[1] if len(sys.argv) > 1 else '180'; HERE = os.path.dirname(os.path.abspath(__file__)); outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'figures_quality')
txt = open(os.path.join(HERE, 'figures_quality', f'cuts_pointing_summary_ref{ref}.md')).read()
def section(title_start):
    i = txt.index(title_start); j = txt.find('\n### ', i + 5); return txt[i:j if j > 0 else None]
def rows(sec): return [[c.strip() for c in r.split('|')[1:-1]] for r in sec.splitlines() if r.startswith('| +')]  # the label of the last row contains '|', so numbers are taken from the right
snr_np = np.array([[float(c) for c in r[-7:-1]] for r in rows(section(f'### no pointing cut (transit θ < {ref}°): change of SNR relative to the reference'))])
snr_p = np.array([[float(c) for c in r[-7:-1]] for r in rows(section('### with the pointing cut θ < 45°: change of SNR relative to the reference'))])
ev_np = np.array([[float(c) for c in r[-7:-1]] for r in rows(section(f'### no pointing cut (transit θ < {ref}°): background rejected'))])
tot_p = np.array([[float(c) for c in r[-7:-1]] for r in rows(section('### with the pointing cut θ < 45°: background rejected'))])
LAB = ['200–342 keV', '342–585 keV', '585–1000 keV', '1000–1710 keV', '1710–2924 keV', '2924–5000 keV']; COL = plt.cm.viridis(np.linspace(0, 0.9, 6)); names = ['+ Distance ≥ 1 cm', '+ SQ ≤ 5', '+ upper two layers', '+ φ < 90°', '+ filter |ψ−φ| < 45°']
fig, axs = plt.subplots(2, 2, figsize=(14, 9)); x = np.arange(5)
for col, (s, r, nm, yl) in enumerate(((snr_np, ev_np, 'no pointing cut', 'background rejected [%] (events)'), (snr_p, tot_p, 'pointing cut θ < 45°', 'background rejected [%] (time + events)'))):
    for i in range(6): axs[0, col].plot(x, s[:, i], color=COL[i], marker='o', ms=3, label=LAB[i]); axs[1, col].plot(x, r[:, i], color=COL[i], marker='o', ms=3)
    axs[0, col].set_title(nm + f' (reference transit θ < {ref}°)'); axs[0, col].set_ylabel('SNR change vs reference [%]'); axs[1, col].set_ylabel(yl)
    for k in (0, 1): axs[k, col].set_xticks(x); axs[k, col].set_xticklabels(names, rotation=25, fontsize=8, ha='right')
axs[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(outdir, f'cuts_pointing_ref{ref}_nobins.png'), dpi=100)
