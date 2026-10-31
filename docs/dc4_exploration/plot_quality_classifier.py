"""Likelihood-ratio classifier on the quality variables (Distance, number of hits, depth category of hit 1 and 2) and the best cut
at a given background removal.

Per energy bin a Poisson regression S_cell ~ b_cell * exp(X beta) models the signal-to-background ratio of every cell of the
(Distance bin, SQ, cat1, cat2) grid (S = simulated signal inside the baseline-optimal ARM window, b = background sum sin(phi)); one-hot main effects
plus pairwise interactions, small ridge penalty. Cells are ranked by the predicted ratio and kept (best first) until the requested
fraction of the background events is removed; S/sqrt(B) is then evaluated with the ARM window re-optimised.
Overfitting check: train on 5 of the 10 signal files and evaluate on the other 5 (both ways).
Usage: python plot_quality_classifier.py signal_dir bkg_dir outdir [removed_fraction]"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sensitivity_quality as Q
from sensitivity_quality import Cut, fom, combined, NE, NA, SINW

sigdir, bkgdir, outdir = sys.argv[1:4]
REM = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
files = Q.signal_files(sigdir)
Hfile = [Q.signal_hist([f])[0] for f in files]
Htot = sum(Hfile); Hc_all = Q.cumulative(Htot)
bN, bS, _ = Q.bkg_hist(bkgdir)
base = fom(Cut(), Hc_all, bN, bS)
LAB = [f'{Q.E_EDGES[i]:.0f}–{Q.E_EDGES[i + 1]:.0f} keV' for i in range(NE)]; COL = plt.cm.viridis(np.linspace(0, 0.9, NE))
EDG = np.array([0, 2, 4, 8, 12, 16, 24, 32, 48, 80, 120]); ND = len(EDG) - 1           # distance bins (0.25 cm units): 0-.5,.5-1,1-2,2-3,3-4,4-6,6-8,8-12,12-20,20+
DLAB = ['0–0.5', '0.5–1', '1–2', '2–3', '3–4', '4–6', '6–8', '8–12', '12–20', '20+']
def coarse(A):       # (NE, NL, ...) -> (NE, ND, ...)
    return np.stack([A[:, EDG[j]:EDG[j + 1]].sum(1) for j in range(ND)], 1)
bNd, bSd = coarse(bN), coarse(bS)                                                     # (NE, ND, NS, NC, NC)
def sig_coarse(Hc): return coarse(Hc.astype(np.float64))                              # (NE, ND, NS, NC, NC, NA+1) cumulative in |ARM|
NS, NC = bNd.shape[2], bNd.shape[3]
# design matrix over (D, S, c1, c2) cells
d, s, c1, c2 = np.meshgrid(np.arange(ND), np.arange(NS), np.arange(NC), np.arange(NC), indexing='ij')
l1, l2, t1, t2 = c1 // 3, c2 // 3, c1 % 3, c2 % 3
def oh(x, n): return np.eye(n)[x.ravel()]
def inter(a, na, b, nb): return (oh(a, na)[:, :, None] * oh(b, nb)[:, None, :]).reshape(a.size, -1)
X = np.hstack([np.ones((d.size, 1)), oh(d, ND), oh(s, NS), oh(l1, 4), oh(l2, 4), oh(t1, 3), oh(t2, 3),
               inter(l1, 4, l2, 4), inter(d, ND, s, NS), inter(d, ND, l1, 4), inter(s, NS, l1, 4), inter(t1, 3, l1, 4), inter(t2, 3, l2, 4), inter(d, ND, l2, 4)])
X = X[:, X.any(0)]; shp = d.shape
def fit_poisson(S, b, ridge=1.0, it=60):
    """S ~ Poisson(b * exp(X beta)); returns the log-ratio for every cell (also the cells with b = 0)."""
    S = S.ravel(); b = b.ravel(); ok = b > 0
    beta = np.zeros(X.shape[1]); beta[0] = np.log(S[ok].sum() / b[ok].sum())
    P = ridge * np.eye(X.shape[1]); P[0, 0] = 0
    for _ in range(it):
        mu = b[ok] * np.exp(np.clip(X[ok] @ beta, -30, 30))
        g = X[ok].T @ (S[ok] - mu) - P @ beta; H = (X[ok] * mu[:, None]).T @ X[ok] + P + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g); beta += step
        if np.abs(step).max() < 1e-7: break
    return (X @ beta).reshape(shp)
def select(score, bnd, frac):
    """keep the best-scoring cells until a fraction `frac` of the background events is removed."""
    o = np.argsort(-score.ravel(), kind='stable'); cum = np.cumsum(bnd.ravel()[o]); keep = np.zeros(score.size, bool)
    keep[o[cum <= (1 - frac) * bnd.sum()]] = True; return keep.reshape(score.shape)
def evaluate(masks, Hd):
    """per bin: best S/sqrt(B) over the ARM window for the cell masks; returns fom, w, sig kept, bkg removed."""
    out = []
    for i in range(NE):
        S = np.einsum('dscq,dscqk->k', masks[i].astype(float), Hd[i])[:NA]; B = (masks[i] * bSd[i]).sum() * SINW
        f = np.where(B > 0, S / np.sqrt(np.maximum(B, 1e-30)), 0); k = f.argmax()
        out.append((f[k], Q.W_GRID[k], Hd[i][..., -1][masks[i]].sum() / Hd[i][..., -1].sum(), 1 - (masks[i] * bNd[i]).sum() / bNd[i].sum()))
    return np.array(out)
def baseline(Hd): return evaluate([np.ones((ND, NS, NC, NC), bool)] * NE, Hd)
def train_select(Hd_train, k0, frac):
    return [select(fit_poisson(Hd_train[i][..., k0[i]], bSd[i]), bNd[i], frac) for i in range(NE)]
k0 = np.searchsorted(Q.W_GRID, base['w'])
Hd_all = sig_coarse(Hc_all); B0 = baseline(Hd_all)
def pct(ev, b): return 100 * (ev[:, 0] / b[:, 0] - 1), 100 * (combined(ev[:, 0]) / combined(b[:, 0]) - 1)

md = []
def table(t, rows, h):
    md.append(f'\n### {t}\n'); md.append('| ' + ' | '.join(h) + ' |'); md.append('|' + '---|' * len(h)); md.extend('| ' + ' | '.join(r) + ' |' for r in rows)
# ---- 1. headline: classifier at REM, in-sample and held-out
masks = train_select(Hd_all, k0, REM); ev = evaluate(masks, Hd_all); ch, cc = pct(ev, B0)
half = [list(range(0, 10, 2)), list(range(1, 10, 2))]; Hh = [sig_coarse(Q.cumulative(sum(Hfile[j] for j in h))) for h in half]
test = []
for a in (0, 1):
    m = train_select(Hh[a], k0, REM); e = evaluate(m, Hh[1 - a]); b = baseline(Hh[1 - a]); test.append(pct(e, b))
cht = np.mean([t[0] for t in test], 0); cct = np.mean([t[1] for t in test])
jk = []
for h in Hfile:
    Hj = sig_coarse(Q.cumulative(Htot - h)); jk.append(pct(evaluate(masks, Hj), baseline(Hj))[1])
jk = np.array(jk); err = np.sqrt(9 / 10 * ((jk - jk.mean()) ** 2).sum())
table(f'Likelihood-ratio classifier removing {100 * REM:.0f}% of the background events in every energy bin (S/√B change, window re-optimised)',
      [['trained and evaluated on all signal files'] + [f'{v:+.1f}' for v in ch] + [f'{cc:+.1f} ± {err:.1f}'],
       ['trained on 5 files, evaluated on the other 5'] + [f'{v:+.1f}' for v in cht] + [f'{cct:+.1f}'],
       ['best rectangular cut (D ≥ 1 cm, SQ ≥ 3, top layer; overall 90%)'] + ['-59.2', '-32.1', '-24.4', '-27.3', '-30.4', '-33.0', '-41.1']],
      ['selection'] + LAB + ['all bins'])
table('Result of the classifier cut per bin', [[LAB[i], f'{100 * ev[i, 3]:.1f}', f'{100 * ev[i, 2]:.1f}', f'{ev[i, 1]:g}'] for i in range(NE)], ['bin', 'bkg removed [%]', 'signal kept [%]', 'optimal w [deg]'])
# ---- 2. what the classifier keeps (share of kept signal / background by feature)
rows = []
for nm, idx, n, lab in (('first-hit layer', l1, 4, ['bottom', '2', '3', 'top']), ('second-hit layer', l2, 4, ['bottom', '2', '3', 'top']), ('SQ', s, NS, ['2', '3', '4', '5', '6', '7+']), ('Distance [cm]', d, ND, DLAB)):
    ks = np.array([[(masks[i] * Hd_all[i][..., -1])[idx == v].sum() for v in range(n)] for i in range(NE)]).sum(0); ks /= ks.sum()
    kb = np.array([[(masks[i] * bNd[i])[idx == v].sum() for v in range(n)] for i in range(NE)]).sum(0); kb /= kb.sum()
    a = np.array([[(Hd_all[i][..., -1])[idx == v].sum() for v in range(n)] for i in range(NE)]).sum(0); a /= a.sum()
    rows += [[nm, lab[v], f'{100 * a[v]:.0f}', f'{100 * ks[v]:.0f}', f'{100 * kb[v]:.0f}'] for v in range(n)]
table('Composition of the events kept by the classifier (shares in %)', rows, ['variable', 'value', 'all signal', 'kept signal', 'kept background'])
# ---- 3. trade-off curve
levels = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]; curve = []
for f in levels:
    m = train_select(Hd_all, k0, f); e = evaluate(m, Hd_all); curve.append(pct(e, B0))
table('Classifier: S/√B change vs background removed (per bin, quadrature)', [[f'{100 * f:.0f}%'] + [f'{v:+.1f}' for v in curve[j][0]] + [f'{curve[j][1]:+.1f}'] for j, f in enumerate(levels)], ['removed'] + LAB + ['all bins'])
fig, ax = plt.subplots(figsize=(7, 5)); lv = np.array(levels) * 100
for i in range(NE): ax.plot(lv, [c[0][i] for c in curve], color=COL[i], marker='o', ms=3, label=LAB[i])
ax.plot(lv, [c[1] for c in curve], 'k--', marker='s', label='all bins'); ax.plot([50, 70, 80, 90, 95], [-9.2, -20.3, -33.7, -41.1, -52.7], 'r:', marker='x', label='best rectangular cut (all bins)')
ax.axhline(0, color='k', lw=.5); ax.set_xlabel('background events removed [%]'); ax.set_ylabel('change in S/√B [%]'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'classifier_tradeoff.png'), dpi=100); plt.close(fig)
open(os.path.join(outdir, 'classifier_summary.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
