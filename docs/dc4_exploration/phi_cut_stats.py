"""Detailed statistics of one rectangular cut using the phi histograms (bkg_phi.py) and the response simulation.
Usage: python phi_cut_stats.py signal_dir bkgphi_dir dmin_cm sq_max layers(comma, 0=bottom..3=top) phi_max_deg [phi_min_deg]"""
import sys, glob, os
import numpy as np
from tra_quality import E_EDGES
sigdir, bkgdir = sys.argv[1:3]; dmin = float(sys.argv[3]); sqmax = int(sys.argv[4]); lay = [int(x) for x in sys.argv[5].split(',')]; phimax = float(sys.argv[6]); phimin = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
NE, NP, ND, NS, NL, NA, DA = 6, 72, 10, 6, 4, 60, 0.5; DEDGE = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 20]); W = DA * np.arange(1, NA + 1); PC = (np.arange(NP) + .5) * 2.5
p = np.deg2rad(PC)[:, None]; ww = np.deg2rad(W)[None, :]; AREA = (np.cos(np.maximum(p - ww, 0)) - np.cos(np.minimum(p + ww, np.pi))) / 2
dlo = np.digitize(dmin, DEDGE)                                                      # first Distance bin kept (cuts must sit on bin edges)
assert dmin == 0 or dmin in DEDGE, 'Distance cut must be one of 0.5,1,2,3,4,6,8,12,20'
m5 = ((PC >= phimin) & (PC < phimax)).reshape(NP, 1, 1, 1) & (np.arange(ND) >= dlo).reshape(1, ND, 1, 1) & (np.arange(2, 8) <= sqmax).reshape(1, 1, NS, 1) & np.isin(np.arange(NL), lay).reshape(1, 1, 1, NL)
Hf = []
for f in sorted(glob.glob(sigdir + '/*.npz')):
    z = np.load(f); w = z['Etrue'].astype(float) ** -2.1; iE = np.digitize(z['E'], E_EDGES) - 1
    iP = np.clip((np.rad2deg(z['phi'].astype(float)) / 2.5).astype(int), 0, NP - 1); iD = np.digitize(z['d01'], DEDGE); iS = np.clip(z['sq'].astype(int), 2, 7) - 2; iL = z['cat1'] // 3
    iA = np.clip((np.abs(z['arm']) / DA).astype(int), 0, NA); ok = (iE >= 0) & (iE < NE)
    Hf.append(np.bincount(((((iE * NP + iP) * ND + iD) * NS + iS) * NL + iL)[ok] * (NA + 1) + iA[ok], weights=w[ok], minlength=NE * NP * ND * NS * NL * (NA + 1)).reshape(NE, NP, ND, NS, NL, NA + 1))
comp = {os.path.basename(f)[:-4]: np.load(f)['N'] for f in sorted(glob.glob(bkgdir + '/*.npz'))}; Nb = sum(comp.values())
def evaluate(H, N):
    Sc = np.cumsum(H, -1); out = []
    for i in range(NE):
        sel = lambda a: a[i][m5]
        S_all = Sc[i][..., -1].sum(); S_k = np.array([(Sc[i][..., k] * m5).sum() for k in range(NA)]); Bk = np.einsum('pdsl,pk->k', N[i] * m5, AREA); B0k = np.einsum('pdsl,pk->k', N[i], AREA)
        S0k = np.array([Sc[i][..., k].sum() for k in range(NA)]); f = np.where(Bk > 0, S_k / np.sqrt(np.maximum(Bk, 1e-30)), 0); f0 = S0k / np.sqrt(B0k); k = f.argmax(); k0 = f0.argmax()
        out.append(dict(f=f[k], f0=f0[k0], w=W[k], w0=W[k0], sig_kept=(Sc[i][..., -1] * m5).sum() / S_all, bkg_kept=(N[i] * m5).sum() / N[i].sum(), Nkept=(N[i] * m5).sum(), N0=N[i].sum(), Sin=S_k[k] / S0k[k0], Bin=Bk[k] / B0k[k0]))
    return out
ev = evaluate(sum(Hf), Nb); tot = lambda key: np.sqrt(sum(e[key] ** 2 for e in ev))
Hh = [sum(Hf[j] for j in range(a, 10, 2)) for a in (0, 1)]; jk = []
for j in range(10):
    Hj = sum(Hf) - Hf[j]; e = evaluate(Hj, Nb); jk.append(100 * (np.sqrt(sum(x['f'] ** 2 for x in e)) / np.sqrt(sum(x['f0'] ** 2 for x in e)) - 1))
jk = np.array(jk); err = np.sqrt(9 / 10 * ((jk - jk.mean()) ** 2).sum())
print(f'Cut: Distance >= {dmin:g} cm, SQ <= {sqmax}, first-hit layer in {lay} (0 = bottom), {phimin:g} <= phi < {phimax:g} deg')
print('| bin | S/√B change [%] | bkg events kept [%] | bkg removed [%] | signal kept [%] | w no cut -> w cut [deg] | signal in window rel. | background in window rel. |'); print('|---|---|---|---|---|---|---|---|')
for i, e in enumerate(ev):
    print(f'| {E_EDGES[i]:.0f}–{E_EDGES[i+1]:.0f} keV | {100*(e["f"]/e["f0"]-1):+.1f} | {100*e["bkg_kept"]:.1f} | {100*(1-e["bkg_kept"]):.1f} | {100*e["sig_kept"]:.1f} | {e["w0"]:g} -> {e["w"]:g} | {e["Sin"]:.2f} | {e["Bin"]:.2f} |')
print(f'| all bins | {100*(tot("f")/tot("f0")-1):+.1f} ± {err:.1f} | {100*sum(e["Nkept"] for e in ev)/sum(e["N0"] for e in ev):.1f} | {100*(1-sum(e["Nkept"] for e in ev)/sum(e["N0"] for e in ev)):.1f} | {100*np.mean([e["sig_kept"] for e in ev]):.1f} (mean) | | | |')
print('\nbackground events kept by component [%] (week 1, 200-5000 keV): ' + ', '.join(f'{k} {100*(v*m5[None]).sum()/v.sum():.0f}' for k, v in comp.items()))
print('week-1 background events in 200-5000 keV: %.3g, kept %.3g' % (sum(e['N0'] for e in ev), sum(e['Nkept'] for e in ev)))
