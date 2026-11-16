"""Gains from discarding 2-hit events with a small reconstructed scatter angle (SQ == 2 and phi < phi0, default 30 deg).
Transit analysis (divided in 5 deg zenith bins, background at the source position, E^-2, reference transit theta < 180 deg) without / with a 45 deg pointing cut, and the all-sky average
(sphere-averaged background, no zenith division). Usage: GAMMA=2 python plot_sq2.py signal_dir bkgsq_dir outdir"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sq_common as C
from sq_common import NE, NT, NA, TC, LAB, COL, W, PC, iso, ev, phi, iE, iA, wt, ok0, Np

outdir = sys.argv[3]; THREF = 180.0
def transit(kw, tmax):
    s2, rem = C.snr2(*C.cut(**kw)); return np.sqrt(s2[:, C.TC < tmax].sum(1)), rem
def allsky(kw):
    sm, bm = C.cut(**kw); ok = ok0 & sm; H = np.bincount((iE * (NA + 1) + iA)[ok], weights=wt[ok], minlength=NE * (NA + 1)).reshape(NE, NA + 1); S = np.cumsum(H, -1)[:, :NA]
    Mp = (Np * bm).sum((2, 3, 4, 5, 6)); B = Mp @ iso; return (S / np.sqrt(np.maximum(B, 1e-30))).max(1), 1 - (Np * bm).sum((1, 2, 3, 4, 5, 6)) / Np.sum((1, 2, 3, 4, 5, 6))
def kept(kw, tmax=None):
    sm, _ = C.cut(**kw); sel = ok0 & (np.ones(len(phi), bool) if tmax is None else ev['theta_src'] < tmax); return np.array([wt[sel & sm & (iE == i)].sum() / wt[sel & (iE == i)].sum() for i in range(NE)])
ref_t, _ = transit(dict(sq5=False), THREF); ref_t45 = ref_t; ref_a, _ = allsky(dict(sq5=False))
md = []
def table(t, rows, h):
    md.append(f'\n### {t}\n'); md.append('| ' + ' | '.join(h) + ' |'); md.append('|' + '---|' * len(h)); md.extend('| ' + ' | '.join(r) + ' |' for r in rows)
H = ['cut'] + LAB + ['all bins']; q = lambda v: np.sqrt((v ** 2).sum()); f1 = lambda v: f'{v:+.1f}'
# ---- 1. the cut alone, versus phi0
phis = [10, 15, 20, 25, 30, 40, 50, 60, 90]; res1 = {}
for p0 in phis:
    r, rem = transit(dict(sq5=False, drop2=p0), THREF); a, rema = allsky(dict(sq5=False, drop2=p0)); res1[p0] = (100 * (r / ref_t - 1), 100 * (q(r) / q(ref_t) - 1), 100 * rem, 100 * (a / ref_a - 1), 100 * (q(a) / q(ref_a) - 1), 100 * kept(dict(sq5=False, drop2=p0)))
table('Discarding 2-hit events with φ < φ0 alone: SNR change [%] (transit θ < 180°, divided / all-sky average) and background events rejected [%]',
      [[f'φ0 = {p}°'] + [f'{res1[p][0][i]:+.1f} / {res1[p][3][i]:+.1f}' for i in range(NE)] + [f'{res1[p][1]:+.1f} / {res1[p][4]:+.1f}'] for p in phis], H)
table('... background events rejected [%] / signal kept [%] (2-hit events with φ < φ0 discarded)', [[f'φ0 = {p}°'] + [f'{res1[p][2][i]:.1f} / {100 - 0:.0f}'.split(' / ')[0] + f' / {res1[p][5][i]:.0f}' for i in range(NE)] + [''] for p in phis], H)
# comparison: other hit multiplicities
rows = []
for nm, kw in (('2-hit & φ < 30°', dict(sq5=False, drop2=30)), ('3-5-hit & φ < 30°', dict(sq5=False, drop2=30, dropsq=3)), ('any hits & φ < 30° (φ > 30° only)', dict(sq5=False, phimax=181))):
    if nm.startswith('any'):
        sm, bm = C.cut(sq5=False); m = bm & ~(PC < 30).reshape(1, 72, 1, 1, 1, 1, 1); sm = sm & (phi >= 30); s2, rem = C.snr2(sm, m); r = np.sqrt(s2.sum(1)); rem = rem
    else: r, rem = transit(kw, THREF)
    rows.append([nm] + [f1(100 * (r[i] / ref_t[i] - 1)) + f' ({100 * rem[i]:.0f}%)' for i in range(NE)] + [f1(100 * (q(r) / q(ref_t) - 1))])
table('Which events to discard below φ = 30° (SNR change, transit θ < 180°; in brackets the background rejected)', rows, H)
# ---- 2. on top of other cuts
sets = [('no other cut', dict(sq5=False)), ('D ≥ 1 cm, SQ ≤ 5, upper two layers', dict(dmin=1, lay=(2, 3))), ('D ≥ 1 cm, SQ ≤ 5, upper two layers, filter 45°', dict(dmin=1, lay=(2, 3), zcap=45)),
        ('D ≥ 1 cm, SQ ≤ 5, upper two layers, φ < 90°, filter 45°', dict(dmin=1, lay=(2, 3), phimax=90, zcap=45)), ('D ≥ 1 cm, SQ ≤ 5, upper two layers, φ < 90°', dict(dmin=1, lay=(2, 3), phimax=90))]
for tmax, nm in ((THREF, 'no pointing cut (transit θ < 180°)'), (45.0, 'pointing cut θ < 45°')):
    rows = []; rows_rej = []
    for sn, kw in sets:
        a, rem0 = transit(kw, tmax); b, rem1 = transit(dict(kw, drop2=30), tmax); sk0 = kept(kw, tmax if tmax < 180 else None); sk1 = kept(dict(kw, drop2=30), tmax if tmax < 180 else None)
        rows.append([sn, 'without the 2-hit cut'] + [f1(100 * (a[i] / ref_t[i] - 1)) for i in range(NE)] + [f1(100 * (q(a) / q(ref_t) - 1))]); rows.append([sn, '+ discard 2-hit φ < 30°'] + [f1(100 * (b[i] / ref_t[i] - 1)) for i in range(NE)] + [f1(100 * (q(b) / q(ref_t) - 1))])
        rows.append([sn, 'effect of the 2-hit cut'] + [f1(100 * (b[i] / a[i] - 1)) for i in range(NE)] + [f1(100 * (q(b) / q(a) - 1))])
        rows_rej.append([sn, f'{100 * rem0.mean():.0f} → {100 * rem1.mean():.0f}'] + [f'{100 * rem0[i]:.0f} → {100 * rem1[i]:.0f}' for i in range(NE)] + [f'{100 * sk0.mean():.0f} → {100 * sk1.mean():.0f}'])
    table(f'{nm}: SNR change relative to the reference (no cuts, θ < 180°) [%]', rows, ['cut set', 'variant'] + LAB + ['all bins'])
    table(f'{nm}: background events rejected [%] and signal kept (mean) [%] without → with the 2-hit cut', rows_rej, ['cut set', 'bkg rejected (mean)'] + LAB + ['signal kept (mean)'])
rows = []
for sn, kw in sets:
    a, _ = allsky(kw); b, _ = allsky(dict(kw, drop2=30)); rows.append([sn] + [f1(100 * (b[i] / a[i] - 1)) for i in range(NE)] + [f1(100 * (q(b) / q(a) - 1))])
table('All-sky average (no zenith division, sphere-averaged background): effect of discarding 2-hit φ < 30° on the SNR [%]', rows, H)
# ---- 3. who are these events
rows = []
for i in range(NE):
    sel = ok0 & (iE == i); sub = sel & (ev['sq'] == 2) & (phi < 30); rest = sel & ~sub; ar = np.abs(ev['arm'])
    def w50(m): x = ar[m & (ar < 30)]; ww = wt[m & (ar < 30)]; o = np.argsort(x); c = np.cumsum(ww[o]); return x[o][np.searchsorted(c, 0.5 * c[-1])]
    bsub = Np[i][:, :, :, :, 0, :].sum(); btot = Np[i].sum(); b30 = Np[i][:C.np.searchsorted(PC, 30), :, :, :, 0, :].sum()
    rows.append([LAB[i], f'{100 * wt[sub].sum() / wt[sel].sum():.1f}', f'{100 * b30 / btot:.1f}', f'{w50(sub):.1f}', f'{w50(rest):.1f}', f'{100 * (ar[sub] > 30).mean():.0f}', f'{100 * (ar[rest] > 30).mean():.0f}'])
table('The 2-hit, φ < 30° events: share of the signal and of the background events, signal ARM core width w50 [deg] (these events / all other events), and fraction with |ARM| > 30° [%] (these / others)', rows, ['bin', 'signal share [%]', 'background share [%]', 'w50 these', 'w50 others', 'tail these [%]', 'tail others [%]'])
# ---- figure
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i in range(NE):
    axs[0].plot(phis, [res1[p][0][i] for p in phis], color=COL[i], marker='o', ms=3, label=LAB[i]); axs[1].plot(phis, [res1[p][2][i] for p in phis], color=COL[i], marker='o', ms=3)
axs[0].plot(phis, [res1[p][1] for p in phis], 'k--', label='all bins'); axs[0].set_ylabel('SNR change [%] (transit θ < 180°)'); axs[1].set_ylabel('background events rejected [%]')
for a in axs[:2]: a.set_xlabel('φ0 [deg] (discard 2-hit events with φ < φ0)'); a.axhline(0, color='k', lw=.5)
axs[0].legend(fontsize=7); x = np.arange(len(sets)); wd = 0.8 / NE
for i in range(NE):
    ef = []
    for sn, kw in sets: a, _ = transit(kw, THREF); b, _ = transit(dict(kw, drop2=30), THREF); ef.append(100 * (b[i] / a[i] - 1))
    axs[2].bar(x + (i - NE / 2 + .5) * wd, ef, wd, color=COL[i])
axs[2].set_xticks(x); axs[2].set_xticklabels([s[0].replace(', ', ',\n') for s in sets], fontsize=6); axs[2].set_ylabel('effect of the 2-hit φ < 30° cut on the SNR [%]'); axs[2].axhline(0, color='k', lw=.5)
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'sq2_small_phi.png'), dpi=95)
open(os.path.join(outdir, 'sq2_small_phi.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
