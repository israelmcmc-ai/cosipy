import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fits = [  # label, K, sK, idx, sIdx, color, ls
 ("Neural-network response, 24 h\n(George, weighted-pulsation tutorial)", 18.33, 3.60, -2.257, 0.0337, "#2a78d6", "-"),
 ("Relative-histogram response, 24 h\n(new PR #22 notebook)",            83.33, 12.06, -2.464, 0.0256, "#eb6834", "-"),
 ("Neural-network response, 3 h\n(cositools example_crab_fit_normalizing_flows)", 4.787, 2.89, -1.979, 0.0942, "#1baf7a", "-"),
]
E = np.geomspace(200, 5000, 200)
fig, ax = plt.subplots(figsize=(9, 7.4), dpi=150)
bg="#fcfcfb"; fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
for lab,K,sK,g,sg,c,ls in fits:
    F = K*E**g
    e2 = E**2*F
    # 1-sigma band assuming K and index are fully anti-correlated (reported r = -1.00)
    sl = np.abs(sK/K - np.log(E)*sg)
    ax.fill_between(E, e2*np.exp(-sl), e2*np.exp(sl), color=c, alpha=0.18, lw=0)
    ax.plot(E, e2, color=c, lw=2, label=f"{lab}\nK={K:.3g}, index={g:.3f}")
ax.plot(E, E**2*4.94717171*E**-2.0, color='#0b0b0b', ls=':', lw=2, label='Injected (true) spectrum\nK=4.947, index=-2.000')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.1, 100)
ax.set_xlabel("Energy (keV)", color="#52514e"); ax.set_ylabel(r"$E^2\,dN/dE$  (keV cm$^{-2}$ s$^{-1}$)", color="#52514e")
ax.set_title("Fitted Crab power law, DC4 simulation", loc="left", color="#0b0b0b", fontsize=13)
ax.grid(alpha=0.25, lw=0.6); [s.set_visible(False) for k,s in ax.spines.items() if k in ('top','right')]
ax.tick_params(colors="#52514e")
ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0,-0.13), ncol=1)
from matplotlib.ticker import ScalarFormatter, FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{x:g}")); ax.xaxis.set_minor_formatter(FuncFormatter(lambda x,_: f"{x:g}" if x in (300,500,2000,3000) else ""))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{y:g}")); ax.yaxis.set_minor_formatter(FuncFormatter(lambda y,_: ""))
fig.text(0.01,0.005,"Bands: 1σ from fit errors, assuming K–index correlation = −1 (as reported). Power law pivot at 1 keV. Injected spectrum as given in the notebooks.",fontsize=7,color="#52514e")
fig.tight_layout(rect=(0,0.02,1,1)); fig.savefig(__import__('sys').argv[1] if len(__import__('sys').argv)>1 else 'crab_spectra_comparison.png', facecolor=bg)
