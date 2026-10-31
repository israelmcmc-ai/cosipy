"""Hit-ordering study: can we keep only events whose second hit is BELOW the first (dz < 0) to reduce the event count without losing much S/sqrt(B)?
Signal: response simulation (E_true^-2.1); background: week-1 .tra (bkg_order.py). Cells: (phi 5 deg, Distance bin, SQ, first-hit layer, second-hit layer, direction).
Background in the |ARM| < w window: exact isotropic-source sphere fraction per event. Window re-optimised for every selection.
Usage: python plot_order_study.py signal_dir bkgorder_dir outdir"""
import sys, os, glob, itertools
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tra_quality import E_EDGES

sigdir, bkgdir, outdir = sys.argv[1:4]
NE, NP, ND, NS, NL, NA, DA = 6, 36, 10, 6, 4, 60, 0.5; DEDGE = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 20]); W = DA * np.arange(1, NA + 1)
PC = (np.arange(NP) + .5) * 5.0; p = np.deg2rad(PC)[:, None]; ww = np.deg2rad(W)[None, :]
AREA = (np.cos(np.maximum(p - ww, 0)) - np.cos(np.minimum(p + ww, np.pi))) / 2
CS = (NP, ND, NS, NL, NL, 2)                                                                  # cell shape per energy bin
files = sorted(glob.glob(sigdir + '/*.npz')); Hs = [np.zeros((NE,) + CS + (NA + 1,), np.float32) for _ in (0, 1)]
for n, f in enumerate(files):
    z = np.load(f); w = z['Etrue'].astype(float) ** -2.1; iE = np.digitize(z['E'], E_EDGES) - 1
    iP = np.clip((np.rad2deg(z['phi'].astype(float)) / 5).astype(int), 0, NP - 1); iD = np.digitize(z['d01'], DEDGE); iS = np.clip(z['sq'].astype(int), 2, 7) - 2
    i1 = z['cat1'] // 3; i2 = z['cat2'] // 3; dr = (z['z2'] < z['z1']).astype(int); iA = np.clip((np.abs(z['arm']) / DA).astype(int), 0, NA); ok = (iE >= 0) & (iE < NE)
    flat = ((((((iE * NP + iP) * ND + iD) * NS + iS) * NL + i1) * NL + i2) * 2 + dr) * (NA + 1) + iA
    Hs[n % 2] += np.bincount(flat[ok], weights=w[ok], minlength=Hs[0].size).reshape(Hs[0].shape).astype(np.float32)
Sc = [np.cumsum(h, -1) for h in Hs]; Sc_all = Sc[0] + Sc[1]; del Hs
comp = {os.path.basename(f)[:-4]: np.load(f)['N'] for f in sorted(glob.glob(bkgdir + '/*.npz'))}; Nb = sum(comp.values())
LAB = [f'{E_EDGES[i]:.0f}–{E_EDGES[i + 1]:.0f} keV' for i in range(NE)]; COL = plt.cm.viridis(np.linspace(0, 0.9, NE))
Ntot = Nb.sum((1, 2, 3, 4, 5, 6))
def fom_mask(m, S_=Sc_all):
    """m: bool (NE,)+CS; returns array per bin: fom, w, signal kept, bkg removed."""
    out = []
    for i in range(NE):
        mf = m[i].reshape(-1).astype(np.float32); S = mf @ S_[i].reshape(-1, NA + 1)[:, :NA]; Bp = (Nb[i] * m[i]).sum((1, 2, 3, 4, 5)); B = Bp @ AREA
        f = np.where(B > 0, S / np.sqrt(np.maximum(B, 1e-30)), 0); k = f.argmax()
        out.append((f[k], W[k], (mf @ S_[i].reshape(-1, NA + 1)[:, -1]) / S_[i][..., -1].sum(), 1 - Bp.sum() / Ntot[i]))
    return np.array(out)
FULL = np.ones(Nb.shape, bool); B0 = fom_mask(FULL)
def change(ev, b=B0): return 100 * (ev[:, 0] / b[:, 0] - 1), 100 * (np.sqrt((ev[:, 0] ** 2).sum()) / np.sqrt((b[:, 0] ** 2).sum()) - 1)
ax_ = lambda shape_i: np.arange(shape_i)
def rect(dmin=0, sq=(2, 7), lay=(0, 1, 2, 3), phimax=180.1, down=None):
    m = (np.arange(ND) >= np.digitize(dmin, DEDGE)).reshape(1, 1, ND, 1, 1, 1, 1) if dmin else np.ones((1, 1, ND, 1, 1, 1, 1), bool)
    m = m & ((np.arange(2, 8) >= sq[0]) & (np.arange(2, 8) <= sq[1])).reshape(1, 1, 1, NS, 1, 1, 1) & np.isin(np.arange(NL), lay).reshape(1, 1, 1, 1, NL, 1, 1) & (PC < phimax).reshape(1, NP, 1, 1, 1, 1, 1)
    if down is not None: m = m & (np.arange(2) == down).reshape(1, 1, 1, 1, 1, 1, 2)
    return np.broadcast_to(m, Nb.shape)
md = []
def table(t, rows, h):
    md.append(f'\n### {t}\n'); md.append('| ' + ' | '.join(h) + ' |'); md.append('|' + '---|' * len(h)); md.extend('| ' + ' | '.join(r) + ' |' for r in rows)

# ---- 1. how many events have the second hit below the first, background vs signal
sd = Sc_all[..., -1]                                                           # (NE,)+CS signal total
rows = []
for i in range(NE):
    s_dn = sd[i][..., 1].sum() / sd[i].sum(); b_dn = Nb[i][..., 1].sum() / Nb[i].sum(); rows.append([LAB[i], f'{100 * s_dn:.0f}', f'{100 * b_dn:.0f}'])
table('Fraction of events whose second hit is below the first [%]', rows, ['bin', 'signal', 'background'])
rows = [[k, f'{100 * v[..., 1].sum() / v.sum():.0f}'] for k, v in comp.items()]
table('... per background component (all energies)', rows, ['component', 'second hit below first [%]'])
rows = []
for j, dl in enumerate(['<0.5', '0.5–1', '1–2', '2–3', '3–4', '4–6', '6–8', '8–12', '12–20', '20+']):
    s = sd[:, :, j].sum((0, 1, 2, 3, 4)); b = Nb[:, :, j].sum((0, 1, 2, 3, 4)); rows.append([dl, f'{100 * s[1] / s.sum():.0f}', f'{100 * b[1] / b.sum():.0f}'])
table('... versus Distance [cm]', rows, ['Distance', 'signal [%]', 'background [%]'])
# ---- 2. the ordering cut alone and on top of other cuts
bases = [('no other cut', dict()), ('Distance ≥ 1 cm', dict(dmin=1)), ('first hit in the upper two layers', dict(lay=(2, 3))), ('first hit in the top layer', dict(lay=(3,))),
         ('Distance ≥ 1 cm, SQ ≤ 5, upper two layers, φ < 90°', dict(dmin=1, sq=(2, 5), lay=(2, 3), phimax=90)), ('SQ ≥ 3', dict(sq=(3, 7)))]
rows = []; extra = []
for nm, kw in bases:
    a = fom_mask(rect(**kw)); b = fom_mask(rect(down=1, **kw)); ca, cca = change(a); cb, ccb = change(b)
    rows.append([nm, 'any order', f'{100 * a[:, 3].mean():.0f}', f'{100 * a[:, 2].mean():.0f}', f'{cca:+.1f}'] + [f'{v:+.1f}' for v in ca])
    rows.append([nm, '+ second hit below first', f'{100 * b[:, 3].mean():.0f}', f'{100 * b[:, 2].mean():.0f}', f'{ccb:+.1f}'] + [f'{v:+.1f}' for v in cb])
table('Adding the ordering cut (second hit below the first) to other cuts', rows, ['other cuts', 'ordering', 'bkg removed [%]', 'signal kept [%]', 'all bins [%]'] + LAB)

# ---- 3. rectangular search: most background removed for a given sensitivity loss, with and without the ordering cut
Ds = [0, 1, 2, 4]; SQs = [(2, 7), (2, 5), (3, 7), (3, 5)]; LYs = [('all', (0, 1, 2, 3)), ('≥2', (1, 2, 3)), ('upper two', (2, 3)), ('top', (3,))]; PHs = [90, 180.1]
grid = []
for dmin, sq, ly, ph, dn in itertools.product(Ds, SQs, LYs, PHs, (None, 1)):
    ev = fom_mask(rect(dmin, sq, ly[1], ph, dn)); ch, cc = change(ev)
    grid.append((100 * (1 - (Nb * rect(dmin, sq, ly[1], ph, dn)).sum() / Nb.sum()), cc, ev, ch, f'Distance ≥ {dmin} cm, SQ {sq[0]}–{sq[1] if sq[1] < 7 else "7+"}, first-hit layer {ly[0]}, φ < {ph:g}°' + (', second hit below first' if dn else '')))
for loss in (10, 20, 30):
    for dn in (False, True):
        c = sorted([g for g in grid if g[1] >= -loss and (' below' in g[4]) == dn], key=lambda g: -g[0])[:1]
    rows = []
    for dn in (False, True):
        c = sorted([g for g in grid if g[1] >= -loss and (' below' in g[4]) == dn], key=lambda g: -g[0])[:1]
        rows += [[('with ordering cut' if dn else 'without ordering cut'), g[4], f'{g[0]:.0f}', f'{100 * g[2][:, 2].mean():.0f}', f'{g[1]:+.1f}'] + [f'{v:+.1f}' for v in g[3]] for g in c]
    table(f'Rectangular cuts: most background removed with an all-bin S/√B loss of at most {loss}%', rows, ['', 'cut', 'bkg removed [%]', 'signal kept [%]', 'all bins [%]'] + LAB)

# ---- 4. cell classifier with and without the direction feature
npg = NP // 3; gp, gd, gs, g1, g2, gr = np.meshgrid(np.arange(NP) // 3, np.arange(ND), np.arange(NS), np.arange(NL), np.arange(NL), np.arange(2), indexing='ij')
def oh(x, n): return np.eye(n, dtype=np.float32)[x.ravel()]
def inter(a, na, b, nb): return (oh(a, na)[:, :, None] * oh(b, nb)[:, None, :]).reshape(a.size, -1)
def design(use_dir):
    c = [np.ones((gp.size, 1), np.float32), oh(gp, npg), oh(gd, ND), oh(gs, NS), oh(g1, NL), oh(g2, NL), inter(gp, npg, gd, ND), inter(gp, npg, g1, NL), inter(gd, ND, gs, NS), inter(gd, ND, g1, NL), inter(gd, ND, g2, NL), inter(gs, NS, g1, NL), inter(g1, NL, g2, NL)]
    if use_dir: c += [oh(gr, 2), inter(gr, 2, gd, ND), inter(gr, 2, g1, NL), inter(gr, 2, g2, NL), inter(gr, 2, gs, NS), inter(gr, 2, gp, npg)]
    X = np.hstack(c); return X[:, X.any(0)]
def fit(X, S, b, ridge=1.0):
    S = S.ravel().astype(np.float64); b = b.ravel(); ok = b > 0; Xo = X[ok].astype(np.float64); So, bo = S[ok], b[ok]
    beta = np.zeros(X.shape[1]); beta[0] = np.log(So.sum() / bo.sum()); P = ridge * np.eye(X.shape[1]); P[0, 0] = 0
    for _ in range(40):
        mu = bo * np.exp(np.clip(Xo @ beta, -30, 30)); g = Xo.T @ (So - mu) - P @ beta; Hm = (Xo * mu[:, None]).T @ Xo + P + 1e-8 * np.eye(X.shape[1]); st = np.linalg.solve(Hm, g); beta += st
        if np.abs(st).max() < 1e-6: break
    return (X.astype(np.float64) @ beta).reshape(CS)
k0 = np.searchsorted(W, B0[:, 1])
def scores(use_dir, S_):
    X = design(use_dir); return [fit(X, S_[i][..., k0[i]], Nb[i] * AREA[:, k0[i]].reshape(NP, 1, 1, 1, 1, 1)) for i in range(NE)]
def select(score, n, frac):
    o = np.argsort(-score.ravel(), kind='stable'); c = np.cumsum(n.ravel()[o]); keep = np.zeros(score.size, bool); keep[o[c <= (1 - frac) * n.sum()]] = True; return keep.reshape(score.shape)
levels = np.round(np.arange(0.40, 0.985, 0.01), 2); show = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
curves = {}; sel_cache = {}
for nm, use_dir in (('classifier without ordering', False), ('classifier with ordering', True)):
    sc = scores(use_dir, Sc_all); cur = []
    for lv in levels:
        m = np.stack([select(sc[i], Nb[i], lv) for i in range(NE)]); ev = fom_mask(m); cur.append((lv,) + change(ev) + (ev,))
    curves[nm] = cur
table('Classifier trade-off: all-bin S/√B change [%] at a given background removal (each energy bin)', [[nm] + [f'{[c for c in cur if abs(c[0] - lv) < 1e-9][0][2]:+.1f}' for lv in show] for nm, cur in curves.items()], ['classifier'] + [f'{100 * lv:.0f}%' for lv in show])
for loss in (10, 20, 30):
    rows = []
    for nm, cur in curves.items():
        ok = [c for c in cur if c[2] >= -loss]; c = max(ok, key=lambda c: c[0]) if ok else None
        if c: rows.append([nm, f'{100 * c[0]:.0f}', f'{100 * c[3][:, 2].mean():.0f}', f'{c[2]:+.1f}'] + [f'{v:+.1f}' for v in c[1]])
    table(f'Classifier: most background removed with an all-bin loss of at most {loss}%', rows, ['classifier', 'bkg removed [%]', 'signal kept [%]', 'all bins [%]'] + LAB)
# held-out check of the classifier with ordering at 75%
ho = []
for a in (0, 1):
    sc = scores(True, Sc[a]); m = np.stack([select(sc[i], Nb[i], 0.75) for i in range(NE)]); ev = fom_mask(m, Sc[1 - a]); ho.append(change(ev, fom_mask(FULL, Sc[1 - a]))[1])
md.append(f'\nHeld-out check of the classifier with ordering at 75% removal: {np.mean(ho):+.1f}% (trained on half of the signal files, tested on the other half) vs {[c for c in curves["classifier with ordering"] if abs(c[0] - 0.75) < 1e-9][0][2]:+.1f}% in-sample.')
fig, ax = plt.subplots(figsize=(7, 5))
for nm, cur in curves.items(): ax.plot([100 * c[0] for c in cur], [c[2] for c in cur], label=nm)
ax.set_xlabel('background events removed [%]'); ax.set_ylabel('all-bin S/√B change [%]'); ax.axhline(0, color='k', lw=.5); ax.set_ylim(-80, 5); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'order_tradeoff.png'), dpi=100)
open(os.path.join(outdir, 'order_summary.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
