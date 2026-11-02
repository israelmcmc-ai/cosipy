"""Shared loader/helpers (position-dependent background F, snr2, cut). Importing it reads sys.argv[1:4] = signal_dir bkgpsichi_dir outdir.
Psichi-cut studies with the background evaluated AT THE SOURCE POSITION instead of averaged over the sphere.
For a source at zenith angle theta_s and an event with psichi polar angle psi and scatter angle phi, the angle between the source and the psichi direction is alpha with
cos(alpha) = cos(theta_s) cos(psi) + sin(theta_s) sin(psi) cos(dchi); for an azimuthally symmetric background (dchi uniform) the probability that |alpha - phi| < w is
(arccos(u1) - arccos(u2))/pi with u = (cos(bound) - cos(theta_s) cos(psi))/(sin(theta_s) sin(psi)). This replaces the isotropic-source factor sin(w) sin(phi) of the other studies.
Signal: response simulation (E_true^-2.1, effective area ~ events/(sin theta dtheta)); SNR^2 = sum over 5-deg zenith bins of max_w A^2/B_theta(w).
Usage: python plot_psichi_pos.py signal_dir bkgpsichi_dir outdir"""
import sys, os, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tra_quality import E_EDGES

sigdir, pdir, outdir = sys.argv[1:4]
NE, NA, DA, NT = 6, 60, 0.5, 36; DEDGE = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 20]); W = DA * np.arange(1, NA + 1); PC = (np.arange(72) + .5) * 2.5
TH = np.arange(0, 180.1, 5.0); TC = 0.5 * (TH[1:] + TH[:-1]); SOL = np.sin(np.deg2rad(TC)) * np.deg2rad(5.0)
Np = sum(np.load(f)['N'] for f in glob.glob(pdir + '/*.npz'))                  # (E, phi72, psi72, D10, L1 4, SQ>5 2, below 2)
# ---- position-dependent window probability F[phi, psi, theta, w]
ph = np.deg2rad(PC)[:, None, None, None]; ps = np.deg2rad(PC)[None, :, None, None]; ts = np.deg2rad(TC)[None, None, :, None]; ww = np.deg2rad(W)[None, None, None, :]
lo = np.maximum(ph - ww, 0); hi = np.minimum(ph + ww, np.pi); den = np.sin(ts) * np.sin(ps)
u1 = np.clip((np.cos(hi) - np.cos(ts) * np.cos(ps)) / den, -1, 1); u2 = np.clip((np.cos(lo) - np.cos(ts) * np.cos(ps)) / den, -1, 1)
F = (np.arccos(u1) - np.arccos(u2)) / np.pi; del ph, ps, ts, ww, lo, hi, den, u1, u2
# sanity: the sphere average of F (weighted by sin theta) should equal the isotropic factor (cos(max(phi-w,0)) - cos(min(phi+w,pi)))/2
iso = (np.cos(np.maximum(np.deg2rad(PC)[:, None] - np.deg2rad(W)[None, :], 0)) - np.cos(np.minimum(np.deg2rad(PC)[:, None] + np.deg2rad(W)[None, :], np.pi))) / 2
avg = (F * (np.sin(np.deg2rad(TC)) * np.deg2rad(5.0) / 2)[None, None, :, None]).sum(2)                           # (phi, psi, w)
print('sanity (sphere average of the position-dependent window fraction vs the isotropic factor, phi=60 deg, w=5 deg): %.4f vs %.4f' % (avg[24, :, 9].mean(), iso[24, 9]))
LAB = [f'{E_EDGES[i]:.0f}–{E_EDGES[i + 1]:.0f} keV' for i in range(NE)]; COL = plt.cm.viridis(np.linspace(0, 0.9, NE))
ev = {k: np.concatenate([np.load(f)[k] for f in sorted(glob.glob(sigdir + '/*.npz'))]) for k in ('E', 'arm', 'Etrue', 'theta_src', 'phi', 'd01', 'sq', 'cat1', 'z1', 'z2')}
phi = np.rad2deg(ev['phi'].astype(float)); psi = np.rad2deg(np.arccos(np.clip(-(ev['z2'].astype(float) - ev['z1']) / ev['d01'], -1, 1))); dpf = np.abs(psi - phi); below = ev['z2'] < ev['z1']
iE = np.digitize(ev['E'], E_EDGES) - 1; iA = np.clip((np.abs(ev['arm']) / DA).astype(int), 0, NA); wt = (ev['Etrue'].astype(float) / 1000.0) ** -float(os.environ.get('GAMMA', '2.1')); ok0 = (iE >= 0) & (iE < NE); iT = np.clip((ev['theta_src'] / 5).astype(int), 0, NT - 1)
def cut(dmin=0, lay=(0, 1, 2, 3), phimax=181, sq5=True, down=None, zcap=None):
    """returns signal event mask and background cell mask; zcap: |psi - phi| < zcap (no extra margin)."""
    sm = (ev['d01'] >= dmin) & np.isin(ev['cat1'] // 3, lay) & (phi < phimax)
    m = (np.arange(10) >= (np.digitize(dmin, DEDGE) if dmin else 0)).reshape(1, 1, 1, 10, 1, 1, 1) & np.isin(np.arange(4), lay).reshape(1, 1, 1, 1, 4, 1, 1) & (PC < phimax).reshape(1, 72, 1, 1, 1, 1, 1)
    if sq5: sm &= ev['sq'] <= 5; m = m & (np.arange(2) == 0).reshape(1, 1, 1, 1, 1, 2, 1)
    if down: sm &= below; m = m & (np.arange(2) == 1).reshape(1, 1, 1, 1, 1, 1, 2)
    if zcap is not None: sm &= dpf < zcap; m = m & (np.abs(PC[:, None] - PC[None, :]) < zcap).reshape(1, 72, 72, 1, 1, 1, 1)
    return sm, np.broadcast_to(m, Np.shape)
Nt = Np.sum((1, 2, 3, 4, 5, 6))
def snr2(sm, bm):
    """(NE, NT) SNR^2 per zenith bin, background removed per energy bin."""
    ok = ok0 & sm; H = np.bincount(((iE * NT + iT) * (NA + 1) + iA)[ok], weights=wt[ok], minlength=NE * NT * (NA + 1)).reshape(NE, NT, NA + 1)
    A = np.cumsum(H, -1)[..., :NA] / SOL[None, :, None]; Mc = (Np * bm).sum((3, 4, 5, 6)); out = np.zeros((NE, NT))
    for i in range(NE):
        B = np.einsum('pq,pqtk->tk', Mc[i], F); out[i] = (A[i] ** 2 / np.maximum(B, 1e-30)).max(1)
    return out, 1 - (Np * bm).sum((1, 2, 3, 4, 5, 6)) / Nt
