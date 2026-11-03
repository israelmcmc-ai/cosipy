"""Background statistics check of the cumulative-cut SNR change per energy bin: background event counts through the cuts, expected background events inside the optimal ARM window,
and a Poisson bootstrap of the background histogram (cell counts resampled with Poisson(N)) for the per-bin SNR change (reference transit theta < 180 deg, E^-2, no pointing cut).
Usage: GAMMA=2 python stats_check_bkg.py signal_dir bkgpsichi_dir outdir [n_boot=12]"""
import sys, os, time
import numpy as np
import psichi_pos_common as C

nboot = int(sys.argv[4]) if len(sys.argv) > 4 else 12; outdir = sys.argv[3]
Np0 = C.Np.copy(); NE, NT = C.NE, C.NT; LAB = C.LAB
steps = [('no cuts', dict(sq5=False)), ('+ Distance ≥ 1 cm', dict(dmin=1, sq5=False)), ('+ SQ ≤ 5', dict(dmin=1)), ('+ upper two layers', dict(dmin=1, lay=(2, 3))), ('+ φ < 90°', dict(dmin=1, lay=(2, 3), phimax=90)), ('+ filter |ψ−φ| < 45°', dict(dmin=1, lay=(2, 3), phimax=90, zcap=45))]
def run(Np):
    C.Np = Np; C.Nt = Np.sum((1, 2, 3, 4, 5, 6)); ref = C.snr2(*C.cut(sq5=False))[0]; out = []
    for nm, kw in steps[1:]: out.append(100 * (np.sqrt(C.snr2(*C.cut(**kw))[0].sum(1) / ref.sum(1)) - 1))
    return np.array(out)
t0 = time.time(); central = run(Np0); print('one pass', time.time() - t0, 's', flush=True)
rng = np.random.default_rng(1); boots = np.array([run(rng.poisson(Np0).astype(float)) for _ in range(nboot)]); err = boots.std(0, ddof=1)
C.Np = Np0
# counts: background events per energy bin through the cuts
md = ['| bin | ' + ' | '.join(nm for nm, _ in steps) + ' |', '|' + '---|' * (len(steps) + 1)]
for i in range(NE):
    cnt = [(Np0[i] * C.cut(**kw)[1][i]).sum() for _, kw in steps]; md.append(f'| {LAB[i]} | ' + ' | '.join(f'{c:.3g}' for c in cnt) + ' |')
tab1 = '\n'.join(md)
md = ['| bin | cumulative cut | SNR change [%] | bootstrap error [%] |', '|---|---|---|---|']
for i in range(NE):
    for k, (nm, _) in enumerate(steps[1:]): md.append(f'| {LAB[i]} | {nm} | {central[k, i]:+.2f} | {err[k, i]:.2f} |')
tab2 = '\n'.join(md)
# expected background events inside the optimal window at theta < 45 for the full cut set (per energy bin)
rows = ['| bin | expected background events in the optimal window, θ<45°, no cuts | with all five cuts |', '|---|---|---|']
for i in range(NE):
    vals = []
    for kw in (steps[0][1], steps[-1][1]):
        sm, bm = C.cut(**kw); Mc = (Np0 * bm).sum((3, 4, 5, 6))[i]; B = np.einsum('pq,pqtk->tk', Mc, C.F)
        ok = C.ok0 & sm; H = np.bincount(((C.iE * C.NT + C.iT) * (C.NA + 1) + C.iA)[ok], weights=C.wt[ok], minlength=NE * C.NT * (C.NA + 1)).reshape(NE, C.NT, C.NA + 1)
        A = (np.cumsum(H, -1)[..., :C.NA] / C.SOL[None, :, None])[i]; k = (A ** 2 / np.maximum(B, 1e-30)).argmax(1); sel = C.TC < 45; vals.append(B[np.arange(C.NT), k][sel].sum())
    rows.append(f'| {LAB[i]} | {vals[0]:.3g} | {vals[1]:.3g} |')
tab3 = '\n'.join(rows)
open(os.path.join(outdir, 'stats_check_bkg.md'), 'w').write('### Background events per energy bin through the cumulative cuts (week 1)\n\n' + tab1 + '\n\n### Per-bin SNR change with Poisson-bootstrap error\n\n' + tab2 + '\n\n### Expected background events in the optimal window\n\n' + tab3 + '\n'); print(tab1); print(tab3); print(tab2)
np.savez(os.path.join(outdir, 'stats_check_bkg.npz'), central=central, err=err)
