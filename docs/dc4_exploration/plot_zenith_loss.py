"""Loss of sensitivity from using only the time a source is within a given zenith angle (45 deg) of a transit over the instrument zenith (+z).
Per zenith bin (5 deg) the response simulation (far-field isotropic beam: effective area ~ detected events / (sin(theta) dtheta)) gives the signal
inside every |ARM| window; the background rate does not depend on the source position. A source crossing the zenith at constant angular speed spends
the same time in every theta bin, so the signal and background of a bin scale with dt, S/sqrt(B) with sqrt(dt) and the bins add in quadrature (unbinned approximation):
  SNR^2 = sum_theta  max_w [A(theta, w)^2 / B(w)]  (A = effective area x flux within the window, ARM window optimised per zenith bin and energy bin).
Loss(theta_max) = 1 - SNR(theta < 45) / SNR(theta < theta_max).
Usage: python plot_zenith_loss.py signal_dir bkgorder_dir outdir"""
import sys, os, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from order_common import *

TH = np.arange(0, 180.1, 5.0); NT = len(TH) - 1; TC = 0.5 * (TH[1:] + TH[:-1])
SOL = np.sin(np.deg2rad(TC)) * np.deg2rad(5.0)                                          # solid-angle weight per bin (x 2 pi)
cuts = [('no cut', dict(), lambda z: np.ones(len(z['E']), bool)),
        ('Distance ≥ 1 cm, SQ ≤ 5, upper two layers, φ < 90°, second hit below first', dict(dmin=1, sq=(2, 5), lay=(2, 3), phimax=90, down=1),
         lambda z: (z['d01'] >= 1) & (z['sq'] <= 5) & ((z['cat1'] // 3) >= 2) & (np.rad2deg(z['phi']) < 90) & (z['z2'] < z['z1'])),
        ('SQ ≤ 5, upper two layers, φ < 90°, second hit below first (no Distance cut)', dict(sq=(2, 5), lay=(2, 3), phimax=90, down=1),
         lambda z: (z['sq'] <= 5) & ((z['cat1'] // 3) >= 2) & (np.rad2deg(z['phi']) < 90) & (z['z2'] < z['z1']))]
files = sorted(glob.glob(sigdir + '/*.npz')); keys = ('E', 'arm', 'Etrue', 'theta_src', 'phi', 'd01', 'sq', 'cat1', 'z1', 'z2')
ev = {k: np.concatenate([np.load(f)[k] for f in files]) for k in keys}
iE = np.digitize(ev['E'], E_EDGES) - 1; iT = np.clip(np.digitize(ev['theta_src'], TH) - 1, 0, NT - 1); iA = np.clip((np.abs(ev['arm']) / DA).astype(int), 0, NA); wt = ev['Etrue'].astype(float) ** -2.1
res = {}
for nm, kw, f in cuts:
    ok = (iE >= 0) & (iE < NE) & f(ev); flat = ((iE * NT + iT) * (NA + 1) + iA)[ok]
    H = np.bincount(flat, weights=wt[ok], minlength=NE * NT * (NA + 1)).reshape(NE, NT, NA + 1); Sc_ = np.cumsum(H, -1)[..., :NA]          # signal inside |ARM| < W[k]
    A = Sc_ / SOL[None, :, None]                                                         # effective area x flux inside the window (arbitrary common unit)
    m = rect(**kw); B = np.stack([(Nb[i] * m[i]).sum((1, 2, 3, 4, 5)) @ AREA for i in range(NE)])                  # (NE, NA) background in the window (arbitrary unit, same for all theta)
    snr2 = np.zeros((NE, NT)); wopt = np.zeros((NE, NT))
    for i in range(NE):
        v = A[i] ** 2 / np.maximum(B[i], 1e-30)[None, :]; k = v.argmax(1); snr2[i] = v[np.arange(NT), k]; wopt[i] = W[k]
    res[nm] = dict(snr2=snr2, wopt=wopt, A=A)
md = []
def table(t, rows, h):
    md.append(f'\n### {t}\n'); md.append('| ' + ' | '.join(h) + ' |'); md.append('|' + '---|' * len(h)); md.extend('| ' + ' | '.join(r) + ' |' for r in rows)
LIM = 45.0; sel = TC < LIM
for nm in res:
    s2 = res[nm]['snr2']; rows = []
    for tmax in (90, 120, 180):
        al = TC < tmax; loss = 1 - np.sqrt(s2[:, sel].sum(1) / s2[:, al].sum(1)); lall = 1 - np.sqrt(s2[:, sel].sum() / s2[:, al].sum())
        rows.append([f'θ < {tmax}°'] + [f'{100 * v:.0f}' for v in loss] + [f'{100 * lall:.0f}'])
    table(f'{nm}: sensitivity loss [%] from using only the time within {LIM:g}° of the zenith (quadrature sum over zenith bins), relative to all times with θ < θmax', rows, ['reference'] + LAB + ['all energies'])
# no-cut vs cut: ratio of the within-45 SNR to the no-cut SNR (to see the cut dependence)
rows = []
base2 = res['no cut']['snr2']
for nm in res:
    s2 = res[nm]['snr2']; rows.append([nm] + [f'{100 * (np.sqrt(s2[i, sel].sum() / base2[i, TC < 90].sum()) - 1):+.1f}' for i in range(NE)])
table('Change of the SNR within 45° of the zenith (with the cut) relative to the no-cut SNR over θ < 90° [%]', rows, ['cut'] + LAB)
# share of SNR^2 by zenith bin (no cut), cumulative
rows = []
for tm in (15, 30, 45, 60, 75, 90, 120, 180):
    c = base2[:, TC < tm].sum(1) / base2[:, TC < 180].sum(1); rows.append([f'θ < {tm}°'] + [f'{100 * v:.0f}' for v in c])
table('No cut: cumulative share of SNR² (all zenith angles = 100%)', rows, ['range'] + LAB)
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i in range(NE):
    a = res['no cut']['A'][i]; k = np.argmax(a[:, :].sum(0) if False else (base2[i][:, None] * 0 + a).max(0)); 
    ea = (res['no cut']['A'][i][:, -1]); axs[0].plot(TC, ea / ea[TC < 20].mean(), color=COL[i], label=LAB[i])                              # effective area in the widest window, relative to its 0-20 deg mean
    axs[1].plot(TC, base2[i] / base2[i].max(), color=COL[i])
    for nm, ls in zip(res, ('-', '--', ':')):
        s2 = res[nm]['snr2'][i]; axs[2].plot(np.arange(15, 181, 5), [np.sqrt(s2[TC < t].sum() / res['no cut']['snr2'][i][TC < 90].sum()) for t in np.arange(15, 181, 5)], color=COL[i], ls=ls, lw=1)
axs[0].set_xlabel('source zenith angle θ [deg]'); axs[0].set_ylabel('effective area in the widest window (rel. to θ < 20°)'); axs[0].legend(fontsize=7)
axs[1].set_xlabel('θ [deg]'); axs[1].set_ylabel('SNR² per zenith bin (rel. to the maximum)'); axs[2].set_xlabel('maximum zenith angle used [deg]'); axs[2].set_ylabel('SNR relative to the no-cut value for θ < 90°')
for a in axs: a.axvline(45, color='k', lw=.5)
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'zenith_transit.png'), dpi=100)
open(os.path.join(outdir, 'zenith_loss.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
