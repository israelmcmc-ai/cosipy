"""Shared loader/helpers of the hit-ordering study (signal cells, background cells, fom_mask, rect). Importing it reads sys.argv[1:4] = signal_dir bkgorder_dir outdir.
Hit-ordering study: can we keep only events whose second hit is BELOW the first (dz < 0) to reduce the event count without losing much S/sqrt(B)?
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
