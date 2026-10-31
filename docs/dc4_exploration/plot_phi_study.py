"""Sensitivity versus background removed for cuts on the Compton scatter angle phi (alone and combined with Distance, number of hits, first-hit layer).
Signal: response simulation (E_true^-2.1); background: week-1 .tra events (bkg_phi.py). Background inside the |ARM| < w window uses the exact
isotropic-source fraction (cos(max(phi-w,0)) - cos(min(phi+w,pi)))/2 of the sphere per event (valid also for small phi).
Usage: python plot_phi_study.py signal_dir bkgphi_dir outdir"""
import sys, os, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tra_quality import E_EDGES

sigdir, bkgdir, outdir = sys.argv[1:4]
NE, NP, ND, NS, NL, NA, DA = 6, 72, 10, 6, 4, 60, 0.5
DEDGE = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 20]); W = DA * np.arange(1, NA + 1)
PC = (np.arange(NP) + 0.5) * 2.5                                                           # bin centres [deg]
def area(w):                                                                               # (NP, len(w)) sphere fraction in the window
    p = np.deg2rad(PC)[:, None]; ww = np.deg2rad(w)[None, :]
    return (np.cos(np.maximum(p - ww, 0)) - np.cos(np.minimum(p + ww, np.pi))) / 2
AREA = area(W)
H = np.zeros((NE, NP, ND, NS, NL, NA + 1)); files = sorted(glob.glob(sigdir + '/*.npz')); Hf = []
for f in files:
    z = np.load(f); w = z['Etrue'].astype(float) ** -2.1; iE = np.digitize(z['E'], E_EDGES) - 1
    iP = np.clip((np.rad2deg(z['phi'].astype(float)) / 2.5).astype(int), 0, NP - 1); iD = np.digitize(z['d01'], DEDGE)
    iS = np.clip(z['sq'].astype(int), 2, 7) - 2; iL = z['cat1'] // 3; iA = np.clip((np.abs(z['arm']) / DA).astype(int), 0, NA)
    ok = (iE >= 0) & (iE < NE); flat = ((((iE * NP + iP) * ND + iD) * NS + iS) * NL + iL) * (NA + 1) + iA
    h = np.bincount(flat[ok], weights=w[ok], minlength=H.size).reshape(H.shape); Hf.append(h)
Hs = sum(Hf); Sc = np.cumsum(Hs, -1)                                                       # Sc[..., k] = signal with |ARM| < W[k] (last = all)
Nb = sum(np.load(f)['N'] for f in glob.glob(bkgdir + '/*.npz'))                            # (NE, NP, ND, NS, NL)
LAB = [f'{E_EDGES[i]:.0f}–{E_EDGES[i + 1]:.0f} keV' for i in range(NE)]; COL = plt.cm.viridis(np.linspace(0, 0.9, NE))
Ntot = Nb.sum((1, 2, 3, 4))

def fom_mask(m, Sc_=Sc):
    """m: boolean mask (NE, NP, ND, NS, NL). Returns per-bin best S/sqrt(B), w, signal kept, background removed."""
    out = []
    for i in range(NE):
        S = (Sc_[i][..., :NA] * m[i][..., None]).sum((0, 1, 2, 3)); B = np.einsum('pdsl,pk->k', Nb[i] * m[i], AREA)
        f = np.where(B > 0, S / np.sqrt(np.maximum(B, 1e-30)), 0); k = f.argmax()
        out.append((f[k], W[k], Sc_[i][..., -1][m[i]].sum() / Sc_[i][..., -1].sum(), 1 - (Nb[i] * m[i]).sum() / Ntot[i]))
    return np.array(out)
full = np.ones(Nb.shape, bool); B0 = fom_mask(full)
def change(ev, b=B0): return 100 * (ev[:, 0] / b[:, 0] - 1), 100 * (np.sqrt((ev[:, 0] ** 2).sum()) / np.sqrt((b[:, 0] ** 2).sum()) - 1)
k0 = np.searchsorted(W, B0[:, 1])

# ---- 1. simple cuts on phi alone
edges = np.arange(0, 180.1, 2.5)
up, lo = [], []                                    # phi < x / phi > x
for x in edges[1:-1]:
    m = np.broadcast_to((PC < x)[None, :, None, None, None], Nb.shape); up.append(fom_mask(m))
    m = np.broadcast_to((PC > x)[None, :, None, None, None], Nb.shape); lo.append(fom_mask(m))
# ---- 2. optimal selections: rank cells by signal/background ratio (Poisson regression with offset) and keep until a fraction is removed
def groups(shape_only=False):
    p, d, s, l = np.meshgrid(np.arange(NP) // 4, np.arange(ND), np.arange(NS), np.arange(NL), indexing='ij'); return p, d, s, l
gp, gd, gs, gl = groups()
def oh(x, n): return np.eye(n)[x.ravel()]
def inter(a, na, b, nb): return (oh(a, na)[:, :, None] * oh(b, nb)[:, None, :]).reshape(a.size, -1)
npg = NP // 4
def design(use_phi, use_rest):
    cols = [np.ones((gp.size, 1))]
    if use_phi: cols += [oh(gp, npg)]
    if use_rest: cols += [oh(gd, ND), oh(gs, NS), oh(gl, NL), inter(gd, ND, gs, NS), inter(gd, ND, gl, NL), inter(gs, NS, gl, NL)]
    if use_phi and use_rest: cols += [inter(gp, npg, gd, ND), inter(gp, npg, gs, NS), inter(gp, npg, gl, NL)]
    X = np.hstack(cols); return X[:, X.any(0)]
def fit(X, S, b, ridge=1.0):
    S = S.ravel(); b = b.ravel(); ok = b > 0
    beta = np.zeros(X.shape[1]); beta[0] = np.log(S[ok].sum() / b[ok].sum()); P = ridge * np.eye(X.shape[1]); P[0, 0] = 0
    for _ in range(80):
        mu = b[ok] * np.exp(np.clip(X[ok] @ beta, -30, 30)); g = X[ok].T @ (S[ok] - mu) - P @ beta; Hm = (X[ok] * mu[:, None]).T @ X[ok] + P + 1e-8 * np.eye(X.shape[1])
        st = np.linalg.solve(Hm, g); beta += st
        if np.abs(st).max() < 1e-7: break
    return (X @ beta).reshape(Nb.shape[1:])
def select(score, n, frac):
    o = np.argsort(-score.ravel(), kind='stable'); c = np.cumsum(n.ravel()[o]); keep = np.zeros(score.size, bool); keep[o[c <= (1 - frac) * n.sum()]] = True; return keep.reshape(score.shape)
def classify(use_phi, use_rest, frac, Sc_=Sc):
    X = design(use_phi, use_rest); m = np.zeros(Nb.shape, bool)
    for i in range(NE):
        S0 = Sc_[i][..., k0[i]]; b0 = Nb[i] * AREA[:, k0[i]][:, None, None, None]
        m[i] = select(fit(X, S0, b0), Nb[i], frac)
    return m
levels = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
methods = {'phi only (ranked bins)': (True, False), 'Distance, hits, first layer (no phi)': (False, True), 'phi + Distance, hits, first layer': (True, True)}
res = {nm: [change(fom_mask(classify(*f, lv))) for lv in levels] for nm, f in methods.items()}
# jackknife / held-out check for the full classifier at 90%
Hh = [sum(Hf[j] for j in range(a, 10, 2)) for a in (0, 1)]; Sh = [np.cumsum(h, -1) for h in Hh]; ho = []
for a in (0, 1):
    m = classify(True, True, 0.9, Sh[a]); ev = fom_mask(m, Sh[1 - a]); b = fom_mask(full, Sh[1 - a]); ho.append(change(ev, b)[1])
# ---- outputs
md = []
def table(t, rows, h):
    md.append(f'\n### {t}\n'); md.append('| ' + ' | '.join(h) + ' |'); md.append('|' + '---|' * len(h)); md.extend('| ' + ' | '.join(r) + ' |' for r in rows)
for nm in methods:
    table(f'{nm}: S/√B change [%] at a given background removal (every energy bin removes the fraction)', [[f'{100 * lv:.0f}%'] + [f'{v:+.1f}' for v in res[nm][j][0]] + [f'{res[nm][j][1]:+.1f}'] for j, lv in enumerate(levels)], ['removed'] + LAB + ['all bins'])
table('All bins in quadrature, comparison', [[nm] + [f'{res[nm][j][1]:+.1f}' for j in range(len(levels))] for nm in methods], ['method'] + [f'{100 * lv:.0f}%' for lv in levels])
md.append(f'\nHeld-out check of the full classifier at 90% (train on 5 signal files, test on the other 5): {np.mean(ho):+.1f}% vs {res["phi + Distance, hits, first layer"][3][1]:+.1f}% in-sample.')
# simple cuts table: best change vs removed for phi < x and phi > x
def tab_simple(lst, name, sel):
    rows = []
    for j, x in enumerate(edges[1:-1]):
        if x in sel:
            ev = lst[j]; ch, cc = change(ev); rows.append([f'{name} {x:g}°', f'{100 * (1 - (ev[:, 3].mean() if False else 0)):.0f}' if False else f'{100 * ev[:, 3].mean():.0f}', f'{100 * ev[:, 2].mean():.0f}'] + [f'{v:+.1f}' for v in ch] + [f'{cc:+.1f}'])
    return rows
table('Simple cuts on phi alone', tab_simple(up, 'phi <', [30, 45, 60, 90, 120, 150]) + tab_simple(lo, 'phi >', [15, 30, 45, 60, 90]), ['cut', 'mean bkg removed [%]', 'mean signal kept [%]'] + LAB + ['all bins'])
# phi profile
fig, axs = plt.subplots(2, 3, figsize=(17, 8.5))
for i, ax in enumerate(axs.flat):
    sp = Hs[i][..., -1].sum((1, 2, 3)); sp = sp / sp.sum(); bp = Nb[i].sum((1, 2, 3)); bp = bp / bp.sum()
    ax.step(PC, bp, where='mid', label='background'); ax.step(PC, sp, where='mid', label='signal (all ARM)')
    sw = Hs[i][..., :NA][..., k0[i]].sum((1, 2, 3)); ax.step(PC, sw / sw.sum(), where='mid', label=f'signal in the optimal window (|ARM|<{B0[i, 1]:g}°)')
    ax.set_title(LAB[i]); ax.set_xlabel('scatter angle φ [deg]'); ax.set_ylabel('fraction per 2.5°'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'phi_profile.png'), dpi=100); plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for nm, c in zip(methods, ('C0', 'C1', 'C3')): axs[0].plot(np.array(levels) * 100, [r[1] for r in res[nm]], color=c, marker='o', label=nm)
for lst, nm, ls in ((up, 'φ < x', '--'), (lo, 'φ > x', ':')):
    axs[0].plot([100 * e[:, 3].mean() for e in lst], [change(e)[1] for e in lst], 'k' + ls, lw=1, label='simple cut ' + nm)
axs[0].set_xlim(0, 100); axs[0].set_ylim(-100, 20); axs[0].axhline(0, color='k', lw=.5); axs[0].set_xlabel('background events removed [%]'); axs[0].set_ylabel('all-bin S/√B change [%]'); axs[0].legend(fontsize=7)
for i in range(NE): axs[1].plot(np.array(levels) * 100, [r[0][i] for r in res['phi only (ranked bins)']], color=COL[i], marker='o', ms=3, label=LAB[i])
axs[1].set_title('φ only (ranked bins), per energy bin'); axs[1].set_xlabel('background removed [%]'); axs[1].set_ylabel('S/√B change [%]'); axs[1].legend(fontsize=7); axs[1].axhline(0, color='k', lw=.5)
for i in range(NE): axs[2].plot(np.array(levels) * 100, [r[0][i] for r in res['phi + Distance, hits, first layer']], color=COL[i], marker='o', ms=3)
axs[2].set_title('φ + Distance, hits, first layer, per energy bin'); axs[2].set_xlabel('background removed [%]'); axs[2].axhline(0, color='k', lw=.5)
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'phi_tradeoff.png'), dpi=100); plt.close(fig)
open(os.path.join(outdir, 'phi_summary.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))

# ---- 4. most events removed for a limited loss of sensitivity (LOSS env, default 20%)
import itertools
LOSS = float(os.environ.get('LOSS', 20)); md2 = []
def table2(t, rows, h):
    md2.append(f'\n### {t}\n'); md2.append('| ' + ' | '.join(h) + ' |'); md2.append('|' + '---|' * len(h)); md2.extend('| ' + ' | '.join(r) + ' |' for r in rows)
rows = []
for nm, f in methods.items():
    best = None
    for lv in np.arange(0.40, 0.99, 0.01):
        ev = fom_mask(classify(*f, lv)); ch, cc = change(ev)
        if cc >= -LOSS: best = (lv, ev, ch, cc)
    if best: lv, ev, ch, cc = best; rows.append([nm, f'{100 * lv:.0f}', f'{100 * ev[:, 2].mean():.0f}', f'{cc:+.1f}'] + [f'{v:+.1f}' for v in ch])
table2(f'Largest per-bin background removal with an all-bin S/√B loss of at most {LOSS:g}% (ranked-cell selections)', rows, ['method', 'bkg removed [%]', 'signal kept [%]', 'all bins [%]'] + LAB)
D_IDX = [0, 2, 3, 5]; SQMIN = [2, 3, 4]; SQMAX = [7, 5]; LAYS = [('all', (0, 1, 2, 3)), ('≥2', (1, 2, 3)), ('top', (3,))]; PLO = [0, 20, 40]; PHI = [60, 90, 180]
Dn = ['0', '0.5', '1', '2', '3', '4', '6', '8', '12', '20']; res_r = []
for dmin, smin, smax, lay, plo, phi_ in itertools.product(D_IDX, SQMIN, SQMAX, LAYS, PLO, PHI):
    if smax < smin: continue
    sh = (1, NP, ND, NS, NL)
    m = ((PC >= plo) & (PC < phi_)).reshape(1, NP, 1, 1, 1) & (np.arange(ND) >= dmin).reshape(1, 1, ND, 1, 1) & ((np.arange(2, 8) >= smin) & (np.arange(2, 8) <= smax)).reshape(1, 1, 1, NS, 1) & np.isin(np.arange(NL), lay[1]).reshape(1, 1, 1, 1, NL)
    m = np.broadcast_to(m, Nb.shape); ev = fom_mask(m); ch, cc = change(ev)
    res_r.append((1 - (Nb * m).sum() / Nb.sum(), cc, ev, ch, f'Distance ≥ {Dn[dmin]} cm, SQ {smin}–{smax if smax < 7 else "7+"}, first-hit layer {lay[0]}, {plo}° ≤ φ < {phi_}°'))
ok = sorted([r for r in res_r if r[1] >= -LOSS], key=lambda r: -r[0])
table2(f'Rectangular cuts (Distance, number of hits, first-hit layer, φ window), most background removed with an all-bin loss of at most {LOSS:g}%',
       [[r[4], f'{100 * r[0]:.0f}', f'{100 * r[2][:, 2].mean():.0f}', f'{r[1]:+.1f}'] + [f'{v:+.1f}' for v in r[3]] for r in ok[:8]], ['cut', 'bkg removed [%]', 'mean signal kept [%]', 'all bins [%]'] + LAB)
open(os.path.join(outdir, 'phi_constraint_summary.md'), 'w').write('\n'.join(md2) + '\n'); print('\n'.join(md2))
