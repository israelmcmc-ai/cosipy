"""Reference SNR, pointing cut and upward-going-events filter for a source transiting over the SC zenith (E^-2 spectrum, background at the source position).
1. reference: zenith bins (5 deg) of the transit theta < 90 deg, SNR^2 = sum over bins of max_w A^2/B_theta(w) per energy bin (arbitrary units).
2. pointing cut: only bins theta < theta_max; loss of SNR vs theta_max; background rejection = fraction of the transit time (= background events) not used = 1 - theta_max/90.
3. upward-going-events filter |psi - phi| < theta0 applied to every event; (a) without pointing cut, (b) with pointing cut theta_max, versus the margin theta0 - theta_max.
Usage: GAMMA=2 python plot_sort_out.py signal_dir bkgpsichi_dir outdir"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from psichi_pos_common import *

THREF = 90.0; tsel = lambda tmax: TC < tmax
def run(zcap=None):
    s2, rem = snr2(*cut(sq5=False, zcap=zcap)); return s2, rem
ref, _ = run(None)                                                                  # (NE, NT) SNR^2 per zenith bin
def snr(s2, tmax): return np.sqrt(s2[:, tsel(tmax)].sum(1))
R = snr(ref, THREF); RA = np.sqrt((R ** 2).sum())
md = []
def table(t, rows, h):
    md.append(f'\n### {t}\n'); md.append('| ' + ' | '.join(h) + ' |'); md.append('|' + '---|' * len(h)); md.extend('| ' + ' | '.join(r) + ' |' for r in rows)
LABS = LAB
# ---- 1. reference
table('Reference SNR (transit θ < 90°, E⁻² spectrum, arbitrary units; relative to the 200–342 keV bin)', [['SNR'] + [f'{v / R[0]:.2f}' for v in R] + [f'{RA / R[0]:.2f}']], ['quantity'] + LABS + ['all bins (quadrature)'])
tm_list = [5 * k for k in range(1, 19)]; cum = np.array([[np.sqrt(ref[i, tsel(t)].sum() / ref[i, tsel(THREF)].sum()) for t in tm_list] for i in range(NE)])
table('Share of the reference SNR reached with the bins θ < θmax (SNR(θmax)/SNR(90°))', [[f'{t}°'] + [f'{v:.2f}' for v in cum[:, j]] for j, t in enumerate(tm_list) if t in (10, 15, 20, 30, 45, 60, 75, 90)], ['θmax'] + LABS)
# ---- 2. pointing cut
point = {t: np.sqrt(ref[:, tsel(t)].sum(1) / ref[:, tsel(THREF)].sum(1)) for t in tm_list}; pointA = {t: np.sqrt(ref[:, tsel(t)].sum() / ref[:, tsel(THREF)].sum()) for t in tm_list}
table('Pointing cut: loss of SNR [%] vs maximum zenith (reference: no cut, θ < 90°) and background rejected (= transit time not used)', [[f'{t}°'] + [f'{100 * (1 - point[t][i]):.0f}' for i in range(NE)] + [f'{100 * (1 - pointA[t]):.0f}', f'{100 * (1 - t / THREF):.0f}'] for t in tm_list if t in (10, 15, 20, 30, 45, 60, 75)], ['θmax'] + LABS + ['all bins', 'background rejected [%]'])
# ---- 3a. upward-going filter without pointing cut
th0 = [5, 10, 15, 20, 30, 45, 60, 75, 90, 120, 150, 181]; fa = {}
for t0 in th0:
    s2, rem = run(t0); fa[t0] = (np.sqrt(s2[:, tsel(THREF)].sum(1) / ref[:, tsel(THREF)].sum(1)), np.sqrt(s2[:, tsel(THREF)].sum() / ref[:, tsel(THREF)].sum()), rem)
table('Upward-going filter |ψ−φ| < θ0, no pointing cut (transit θ < 90°): SNR change [%] and background events rejected [%] per energy bin', [[f'{t0}°'] + [f'{100 * (fa[t0][0][i] - 1):+.1f}' for i in range(NE)] + [f'{100 * (fa[t0][1] - 1):+.1f}'] + [f'{100 * fa[t0][2][i]:.0f}' for i in range(NE)] for t0 in th0], ['θ0'] + [f'SNR {l}' for l in LABS] + ['SNR all bins'] + [f'rej {l}' for l in LABS])
# ---- 3b. with a pointing cut, versus the margin
tmaxs = [15, 30, 45, 60, 90]; margins = [-10, -5, 0, 5, 10, 15, 20, 30, 45, 60]; fb = {}
for tm in tmaxs:
    for m in margins:
        t0 = tm + m
        if t0 <= 0: continue
        s2, rem = run(min(t0, 181)); sel = tsel(tm)
        fb[(tm, m)] = dict(vs_ref=np.sqrt(s2[:, sel].sum(1) / ref[:, tsel(THREF)].sum(1)), vs_ref_all=np.sqrt(s2[:, sel].sum() / ref[:, tsel(THREF)].sum()), vs_point=np.sqrt(s2[:, sel].sum(1) / ref[:, sel].sum(1)), vs_point_all=np.sqrt(s2[:, sel].sum() / ref[:, sel].sum()), rem=rem, tfrac=tm / THREF)
for tm in tmaxs:
    rows = [[f'{m:+d}°', f'{tm + m}°'] + [f'{100 * (fb[(tm, m)]["vs_point"][i] - 1):+.1f}' for i in range(NE)] + [f'{100 * (fb[(tm, m)]["vs_point_all"] - 1):+.1f}', f'{100 * (fb[(tm, m)]["vs_ref_all"] - 1):+.1f}', f'{100 * (1 - (1 - fb[(tm, m)]["tfrac"]) * 0 - fb[(tm, m)]["tfrac"] * (1 - fb[(tm, m)]["rem"].mean())):.0f}'] for m in margins if (tm, m) in fb]
    table(f'Pointing cut θ < {tm}° plus the upward-going filter, versus the margin θ0 − θmax: SNR change relative to the pointing cut alone [%], all bins relative to the pointing cut alone / to the reference, and total background rejected (time and filter) [%]', rows, ['margin', 'θ0'] + [f'{l}' for l in LABS] + ['all bins vs pointing cut', 'all bins vs reference', 'total bkg rejected [%] (mean over bins)'])
# ---- figures
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
ax = axs[0]; xe = np.arange(NE); ax.bar(xe, R / R[0], color=COL); ax.set_xticks(xe); ax.set_xticklabels(LABS, rotation=25); ax.set_ylabel('reference SNR (relative to 200–342 keV)'); ax.set_title('1. Reference SNR vs energy (θ<90°, E⁻²)')
for i in range(NE): axs[1].plot(tm_list, [100 * (1 - point[t][i]) for t in tm_list], color=COL[i], label=LABS[i])
axs[1].plot(tm_list, [100 * (1 - pointA[t]) for t in tm_list], 'k--', label='all bins'); axs[1].plot(tm_list, [100 * (1 - t / THREF) for t in tm_list], 'r:', label='background rejected (time)')
axs[1].set_xlabel('maximum zenith θmax [deg]'); axs[1].set_ylabel('[%]'); axs[1].set_title('2. Pointing cut: SNR loss (solid) and background rejection (red)'); axs[1].legend(fontsize=7)
for i in range(NE): axs[2].plot(th0, [100 * (fa[t][0][i] - 1) for t in th0], color=COL[i], marker='o', ms=3, label=f'SNR change {LABS[i]}'); axs[2].plot(th0, [100 * fa[t][2][i] for t in th0], color=COL[i], ls=':')
axs[2].set_xscale('log'); axs[2].set_xlabel('θ0 [deg]'); axs[2].set_ylabel('SNR change (solid) / background rejected (dotted) [%]'); axs[2].set_title('3a. Upward-going filter, no pointing cut'); axs[2].axhline(0, color='k', lw=.5)
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'sortout_1_2_3a.png'), dpi=95); plt.close(fig)
fig, axs = plt.subplots(2, 3, figsize=(18, 9.5)); CT = plt.cm.plasma(np.linspace(0, 0.85, len(tmaxs)))
for j, (tm, c) in enumerate(zip(tmaxs, CT)):
    ms = [m for m in margins if (tm, m) in fb]
    axs[0, 0].plot(ms, [100 * (fb[(tm, m)]['vs_point_all'] - 1) for m in ms], color=c, marker='o', ms=3, label=f'θmax = {tm}°')
    axs[0, 1].plot(ms, [100 * (fb[(tm, m)]['vs_ref_all'] - 1) for m in ms], color=c, marker='o', ms=3, label=f'θmax = {tm}°')
    axs[0, 2].plot(ms, [100 * (1 - (tm / THREF) * (1 - fb[(tm, m)]['rem'].mean())) for m in ms], color=c, marker='o', ms=3, label=f'θmax = {tm}°')
    axs[1, 0].plot(ms, [100 * fb[(tm, m)]['rem'].mean() for m in ms], color=c, marker='o', ms=3)
    for i in (0, 2, 5): axs[1, 1].plot(ms, [100 * (fb[(tm, m)]['vs_point'][i] - 1) for m in ms], color=c, ls=['-', '--', ':'][(0, 2, 5).index(i)], lw=1)
axs[0, 0].set_ylabel('all-bin SNR change vs the pointing cut alone [%]'); axs[0, 1].set_ylabel('all-bin SNR change vs the reference (θ<90°, no filter) [%]'); axs[0, 2].set_ylabel('total background rejected (time + filter) [%]')
axs[1, 0].set_ylabel('background events rejected by the filter [%]'); axs[1, 1].set_ylabel('SNR change vs pointing cut alone [%]: 200–342 (solid), 585–1000 (dashed), 2.9–5 MeV (dotted)')
for a in axs.flat[:5]: a.set_xlabel('margin θ0 − θmax [deg]'); a.axhline(0, color='k', lw=.5); a.axvline(0, color='k', lw=.5, ls=':')
axs[0, 0].legend(fontsize=8); axs[1, 2].axis('off'); fig.suptitle('3b. Upward-going filter combined with the pointing cut vs the margin θ0 − θmax'); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'sortout_3b.png'), dpi=95)
open(os.path.join(outdir, 'sortout_summary.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
