"""Psichi zenith-cap cut (|psi - phi| < theta0) applied to a WHOLE transit (all sources up to theta_max), compared with no cut over the same transit.
SNR^2 = sum over 5-deg zenith bins of max_w A(theta,w)^2/B(w) (see plot_zenith_loss.py); sources outside the cap lose signal, so the SNR first rises (background rejection) and then falls.
Usage: python plot_zenith_cap_total.py signal_dir bkgpsichi_dir outdir"""
import sys, os, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tra_quality import E_EDGES

sigdir, pdir, outdir = sys.argv[1:4]
NE, NA, DA = 6, 60, 0.5; W = DA * np.arange(1, NA + 1); PC = (np.arange(72) + .5) * 2.5; p = np.deg2rad(PC)[:, None]; w = np.deg2rad(W)[None, :]
AREA = (np.cos(np.maximum(p - w, 0)) - np.cos(np.minimum(p + w, np.pi))) / 2
Np = sum(np.load(f)['N'] for f in glob.glob(pdir + '/*.npz')); Nt = Np.sum((1, 2, 3, 4, 5, 6))
LAB = [f'{E_EDGES[i]:.0f}–{E_EDGES[i + 1]:.0f} keV' for i in range(NE)]; COL = plt.cm.viridis(np.linspace(0, 0.9, NE))
ev = {k: np.concatenate([np.load(f)[k] for f in sorted(glob.glob(sigdir + '/*.npz'))]) for k in ('E', 'arm', 'Etrue', 'theta_src', 'phi', 'z1', 'z2', 'd01')}
phi = np.rad2deg(ev['phi'].astype(float)); psi = np.rad2deg(np.arccos(np.clip(-(ev['z2'].astype(float) - ev['z1']) / ev['d01'], -1, 1))); dpf = np.abs(psi - phi)
iE = np.digitize(ev['E'], E_EDGES) - 1; iA = np.clip((np.abs(ev['arm']) / DA).astype(int), 0, NA); wt = ev['Etrue'].astype(float) ** -2.1; ok0 = (iE >= 0) & (iE < NE)
TH = np.arange(0, 180.1, 5.0); NT = 36; TC = 0.5 * (TH[1:] + TH[:-1]); SOL = np.sin(np.deg2rad(TC)) * np.deg2rad(5.0); iT = np.clip((ev['theta_src'] / 5).astype(int), 0, NT - 1)
def snr2(cap, margin=0.0):
    keep = np.ones(len(phi), bool) if cap is None else dpf < cap + margin; ok = ok0 & keep
    H = np.bincount(((iE * NT + iT) * (NA + 1) + iA)[ok], weights=wt[ok], minlength=NE * NT * (NA + 1)).reshape(NE, NT, NA + 1); A = np.cumsum(H, -1)[..., :NA] / SOL[None, :, None]
    m = np.ones(Np.shape, bool) if cap is None else np.broadcast_to((np.abs(PC[:, None] - PC[None, :]) < cap + margin).reshape(1, 72, 72, 1, 1, 1, 1), Np.shape)
    B = (Np * m).sum((2, 3, 4, 5, 6)) @ AREA; rem = 1 - (Np * m).sum((1, 2, 3, 4, 5, 6)) / Nt; out = np.zeros((NE, NT))
    for i in range(NE): out[i] = (A[i] ** 2 / np.maximum(B[i], 1e-30)[None, :]).max(1)
    return out, rem, (wt * (ok & (ev['theta_src'] < 45))).sum() / (wt * (ok0 & (ev['theta_src'] < 45))).sum()
base, _, _ = snr2(None); caps = [5, 10, 15, 20, 25, 30, 45, 60, 75, 90, 120, 150]
md = []; res = {}
for tmax in (45, 90, 180):
    sel = TC < tmax; rows = []; res[tmax] = []
    for c in caps:
        s, rem, sk = snr2(c); ch = 100 * (np.sqrt(s[:, sel].sum(1) / base[:, sel].sum(1)) - 1); cc = 100 * (np.sqrt(s[:, sel].sum() / base[:, sel].sum()) - 1); res[tmax].append((ch, cc))
        rows.append([f'{c}°'] + [f'{v:+.0f}' for v in ch] + [f'{cc:+.1f}', f'{100 * rem.mean():.0f}', f'{100 * sk:.0f}'])
    md.append(f'\n### Zenith-cap cut applied to every event, transit θ < {tmax}° (change in SNR vs no cut over the same transit [%])\n'); md.append('| θ0 | ' + ' | '.join(LAB) + ' | all bins | mean bkg rejected [%] | signal kept for sources θ<45° [%] |'); md.append('|' + '---|' * (NE + 4)); md += ['| ' + ' | '.join(r) + ' |' for r in rows]
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for ax, tmax in zip(axs, (45, 90, 180)):
    for i in range(NE): ax.plot(caps, [r[0][i] for r in res[tmax]], color=COL[i], marker='o', ms=3, label=LAB[i])
    ax.plot(caps, [r[1] for r in res[tmax]], 'k--', marker='s', ms=3, label='all bins'); ax.axhline(0, color='k', lw=.5); ax.set_xscale('log'); ax.set_xlabel('zenith cap θ0 [deg]'); ax.set_ylabel('change in SNR [%]'); ax.set_title(f'whole transit θ < {tmax}° (cut applied to every event)')
axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'zenith_cap_total.png'), dpi=100)
open(os.path.join(outdir, 'zenith_cap_total.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
