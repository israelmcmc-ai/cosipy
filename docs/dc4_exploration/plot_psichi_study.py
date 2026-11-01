"""Cut on psichi consistent with a source within 45 deg of the SC zenith (+z), as an alternative / addition to the hit-order cut.
The psichi direction p (= minus the scattered direction) has polar angle psi from +z; the source lies on the ring at angle phi from p, whose smallest zenith angle is |psi - phi|.
An event is 'consistent with a source within 45 deg of the zenith' if |psi - phi| < 45 deg + margin (the margin absorbs the ARM error).
Sensitivity = a transit over the zenith (theta < 45 deg only): 5 deg zenith bins, SNR^2 = sum over bins of A(theta,w)^2/B(w), window optimised per bin
(see plot_zenith_loss.py); the signal is the response simulation with true source zenith angle < 45 deg, the background the week-1 .tra events (bkg_psichi.py).
Usage: python plot_psichi_study.py signal_dir bkgpsichi_dir outdir"""
import sys, os, glob, itertools
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tra_quality import E_EDGES

sigdir, bkgdir, outdir = sys.argv[1:4]
NE, NP, ND, NL, NA, DA = 6, 72, 10, 4, 60, 0.5; DEDGE = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 20]); W = DA * np.arange(1, NA + 1)
PC = (np.arange(NP) + .5) * 2.5; p_ = np.deg2rad(PC)[:, None]; w_ = np.deg2rad(W)[None, :]
AREA = (np.cos(np.maximum(p_ - w_, 0)) - np.cos(np.minimum(p_ + w_, np.pi))) / 2
Nb = sum(np.load(f)['N'] for f in glob.glob(bkgdir + '/*.npz'))                  # (NE, phi, psi, D, L1, SQ>5, below)
LAB = [f'{E_EDGES[i]:.0f}–{E_EDGES[i + 1]:.0f} keV' for i in range(NE)]
ev = {k: np.concatenate([np.load(f)[k] for f in sorted(glob.glob(sigdir + '/*.npz'))]) for k in ('E', 'arm', 'Etrue', 'theta_src', 'phi', 'd01', 'sq', 'cat1', 'z1', 'z2')}
phi = np.rad2deg(ev['phi'].astype(float)); psi = np.rad2deg(np.arccos(np.clip(-(ev['z2'].astype(float) - ev['z1']) / ev['d01'], -1, 1))); dpf = np.abs(psi - phi)
iE = np.digitize(ev['E'], E_EDGES) - 1; near = ev['theta_src'] < 45; TH = np.arange(0, 45.1, 5.0); NT = len(TH) - 1; TC = 0.5 * (TH[1:] + TH[:-1]); SOL = np.sin(np.deg2rad(TC)) * np.deg2rad(5.0)
iT = np.clip(np.digitize(ev['theta_src'], TH) - 1, 0, NT - 1); iA = np.clip((np.abs(ev['arm']) / DA).astype(int), 0, NA); wt = ev['Etrue'].astype(float) ** -2.1
below = ev['z2'] < ev['z1']
def smask(dmin=0, lay=(0, 1, 2, 3), phimax=181, sq5=True, down=None, margin=None):
    m = (ev['d01'] >= dmin) & np.isin(ev['cat1'] // 3, lay) & (phi < phimax)
    if sq5: m &= ev['sq'] <= 5
    if down: m &= below
    if margin is not None: m &= dpf < 45 + margin
    return m
def bmask(dmin=0, lay=(0, 1, 2, 3), phimax=181, sq5=True, down=None, margin=None):
    m = (np.arange(ND) >= (np.digitize(dmin, DEDGE) if dmin else 0)).reshape(1, 1, 1, ND, 1, 1, 1) & np.isin(np.arange(NL), lay).reshape(1, 1, 1, 1, NL, 1, 1) & (PC < phimax).reshape(1, NP, 1, 1, 1, 1, 1)
    if sq5: m = m & (np.arange(2) == 0).reshape(1, 1, 1, 1, 1, 2, 1)
    if down: m = m & (np.arange(2) == 1).reshape(1, 1, 1, 1, 1, 1, 2)
    if margin is not None: m = m & (np.abs(PC[:, None] - PC[None, :]) < 45 + margin).reshape(1, NP, NP, 1, 1, 1, 1)
    return np.broadcast_to(m, Nb.shape)
Ntot = Nb.sum((1, 2, 3, 4, 5, 6))
def evaluate(**kw):
    ok = (iE >= 0) & (iE < NE) & near & smask(**kw); H = np.bincount(((iE * NT + iT) * (NA + 1) + iA)[ok], weights=wt[ok], minlength=NE * NT * (NA + 1)).reshape(NE, NT, NA + 1)
    A = np.cumsum(H, -1)[..., :NA] / SOL[None, :, None]; m = bmask(**kw); Bp = (Nb * m).sum((2, 3, 4, 5, 6))                        # (NE, phi)
    B = Bp @ AREA; snr2 = np.zeros(NE)
    for i in range(NE): snr2[i] = (A[i] ** 2 / np.maximum(B[i], 1e-30)[None, :]).max(1).sum()
    return snr2, 1 - (Nb * m).sum() / Nb.sum(), (wt * ((iE >= 0) & (iE < NE) & near & smask(**kw))).sum() / (wt * ((iE >= 0) & (iE < NE) & near)).sum()
S0, _, _ = evaluate(sq5=False)
def chg(s2): return 100 * (np.sqrt(s2 / S0) - 1), 100 * (np.sqrt(s2.sum() / S0.sum()) - 1)
md = []
def table(t, rows, h):
    md.append(f'\n### {t}\n'); md.append('| ' + ' | '.join(h) + ' |'); md.append('|' + '---|' * len(h)); md.extend('| ' + ' | '.join(r) + ' |' for r in rows)
def row(nm, kw):
    s2, rem, sk = evaluate(**kw); ch, cc = chg(s2); return [nm, f'{100 * rem:.0f}', f'{100 * sk:.0f}'] + [f'{v:+.1f}' for v in ch] + [f'{cc:+.1f}']
H = ['cut', 'bkg removed [%]', 'signal (θ<45°) kept [%]'] + LAB + ['all bins']
# 1. psichi consistency alone
table('Psichi consistent with a source within 45° of the zenith (|ψ − φ| < 45° + margin), alone: S/√B change for the θ < 45° transit [%]',
      [row(f'margin {m}°', dict(sq5=False, margin=m)) for m in (0, 5, 10, 15, 20, 30)] + [row('hit order only (second hit below first)', dict(sq5=False, down=1)), row('psichi margin 10° + hit order', dict(sq5=False, margin=10, down=1))], H)
# 2. on top of the base cuts
base = dict(dmin=1, lay=(2, 3), phimax=90)
rows = [row('base: D ≥ 1 cm, SQ ≤ 5, upper two layers, φ < 90°', base), row('base + hit order', dict(base, down=1))] + [row(f'base + psichi margin {m}°', dict(base, margin=m)) for m in (0, 5, 10, 15, 20)] + [row('base + psichi margin 10° + hit order', dict(base, margin=10, down=1))]
table('Added to the base cuts', rows, H)
base2 = dict(lay=(2, 3), phimax=90)
rows = [row('SQ ≤ 5, upper two layers, φ < 90° (no Distance cut)', base2), row('+ hit order', dict(base2, down=1))] + [row(f'+ psichi margin {m}°', dict(base2, margin=m)) for m in (0, 5, 10, 20)]
table('Added to the cuts without the Distance requirement', rows, H)
# 3. most background removed for a limited loss
grid = []
for dmin, lay, ph, opt in itertools.product((0, 1, 2), ((0, 1, 2, 3), (1, 2, 3), (2, 3), (3,)), (90, 181), ('none', 'order', 'psichi0', 'psichi5', 'psichi10', 'psichi20', 'both10')):
    kw = dict(dmin=dmin, lay=lay, phimax=ph)
    if opt == 'order': kw['down'] = 1
    elif opt.startswith('psichi'): kw['margin'] = int(opt[6:])
    elif opt == 'both10': kw.update(margin=10, down=1)
    s2, rem, sk = evaluate(**kw); grid.append((rem, chg(s2), sk, opt, kw))
names = {'none': 'neither', 'order': 'hit order', 'psichi0': 'psichi margin 0°', 'psichi5': 'psichi margin 5°', 'psichi10': 'psichi margin 10°', 'psichi20': 'psichi margin 20°', 'both10': 'psichi 10° + hit order'}
desc = lambda kw: f'Distance ≥ {kw["dmin"]} cm, SQ ≤ 5, first-hit layer ' + {(0, 1, 2, 3): 'all', (1, 2, 3): '≥2', (2, 3): 'upper two', (3,): 'top'}[kw['lay']] + f', φ < {kw["phimax"]}°' + (f', psichi margin {kw["margin"]}°' if 'margin' in kw else '') + (', second hit below first' if kw.get('down') else '')
for loss in (10, 20, 30):
    rows = []
    for opt in names:
        c = sorted([g for g in grid if g[3] == opt and g[1][1] >= -loss], key=lambda g: -g[0])[:1]
        rows += [[names[opt], desc(g[4]), f'{100 * g[0]:.0f}', f'{100 * g[2]:.0f}', f'{g[1][1]:+.1f}'] + [f'{v:+.1f}' for v in g[1][0]] for g in c]
    table(f'Most background removed with an all-bin S/√B loss of at most {loss}% (θ < 45° transit), by the extra cut used', rows, ['extra cut', 'cuts', 'bkg removed [%]', 'signal kept [%]', 'all bins [%]'] + LAB)
fig, ax = plt.subplots(figsize=(7.5, 5))
for opt, c in zip(('none', 'order', 'psichi10', 'both10'), ('C0', 'C1', 'C2', 'C3')):
    g = sorted([x for x in grid if x[3] == opt], key=lambda x: x[0]); xs = np.array([x[0] * 100 for x in g]); ys = np.array([x[1][1] for x in g]); env = np.maximum.accumulate(ys[::-1])[::-1]
    ax.scatter(xs, ys, s=6, color=c, alpha=.3); ax.plot(xs, env, color=c, label=names[opt])
ax.set_xlabel('background events removed [%]'); ax.set_ylabel('all-bin S/√B change, θ < 45° transit [%]'); ax.set_ylim(-60, 5); ax.legend(); ax.axhline(0, color='k', lw=.5); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'psichi_tradeoff.png'), dpi=100)
open(os.path.join(outdir, 'psichi_summary.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
