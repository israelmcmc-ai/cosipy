"""Background events rejected by the upward-going filter |psi - phi| < theta0 versus theta0, per energy bin, from the committed table of plot_sort_out.py
(figures_quality/sortout_summary.md, section "Upward-going filter ... no pointing cut"; percentages rounded to integers; event rejection, independent of any pointing cut).
Usage: python plot_filter_rejection.py outdir"""
import sys, os, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

outdir = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__))
txt = open(os.path.join(HERE, 'figures_quality', 'sortout_summary.md')).read(); sec = txt[txt.index('### Upward-going filter'):]; sec = sec[:sec.index('### Pointing cut θ < 15°')]
rows = [r for r in sec.splitlines() if re.match(r'\| [0-9]+° \|', r)]
th0 = np.array([float(r.split('|')[1].strip().rstrip('°')) for r in rows]); th0[th0 > 180] = 180
rej = np.array([[float(c) for c in r.split('|')[9:15]] for r in rows])                    # last 6 columns: rejected [%] per energy bin
LAB = ['200–342 keV', '342–585 keV', '585–1000 keV', '1000–1710 keV', '1710–2924 keV', '2924–5000 keV']; COL = plt.cm.viridis(np.linspace(0, 0.9, 6))
fig, ax = plt.subplots(figsize=(7.5, 5))
for i in range(6): ax.plot(th0, rej[:, i], color=COL[i], marker='o', ms=3, label=LAB[i])
ax.set_xscale('log'); ax.set_xlabel('θ0 [deg]'); ax.set_ylabel('background events rejected [%]'); ax.set_title('Upward-going filter |ψ−φ| < θ0: background rejection'); ax.set_ylim(0, 100); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'filter_rejection.png'), dpi=110)
