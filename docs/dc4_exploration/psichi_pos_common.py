"""Shared loader and helpers for the studies that evaluate the background AT THE SOURCE POSITION.

Importing this module (``from psichi_pos_common import *``) reads the command line
``sys.argv[1:4] = signal_dir  bkgpsichi_dir  outdir`` and loads everything the studies need.

Why a position-dependent background
-----------------------------------
For an event with scatter angle ``phi`` and psichi direction ``p`` (the sky direction pointing back along the scattered
photon, so the source lies on a ring at angle ``phi`` from ``p``), the event falls inside the |ARM| < w window of a
source at zenith angle ``theta_s`` only if |alpha - phi| < w, where ``alpha`` is the angle between the source and ``p``:

    cos(alpha) = cos(theta_s) cos(psi) + sin(theta_s) sin(psi) cos(dchi)

``psi`` is the polar angle of ``p`` from the spacecraft +z axis and ``dchi`` the azimuth difference between ``p`` and the
source.  For an azimuthally symmetric background (``dchi`` uniform) the probability of falling inside the window is

    P = (arccos(u1) - arccos(u2)) / pi,     u = (cos(bound) - cos(theta_s) cos(psi)) / (sin(theta_s) sin(psi))

with ``bound`` the lower / upper edge of the allowed ``alpha`` range.  This replaces the sphere-averaged factor
sin(w) sin(phi) used by the "all-sky average" studies.

Signal and SNR
--------------
Signal = the response simulation (isotropic far-field beam, true energy re-weighted to a power law E_true^-GAMMA,
GAMMA from the environment, default 2.1).  The effective area at zenith angle theta is proportional to
(detected events) / (sin(theta) dtheta).  For a source transiting the zenith at constant angular speed every 5 deg
zenith bin is observed for the same time, so the bins are independent and add in quadrature:

    SNR^2 = sum over zenith bins of  max_w  A_theta(w)^2 / B_theta(w)

(see ``snr2``; the window w is optimised separately for every zenith bin and energy bin).

Names exported for the study scripts
------------------------------------
Binning:  NE, NA, DA, NT, DEDGE, W, PC, TH, TC, SOL;  Labels: LAB, COL
Background: Np (histogram), Nt (its total per energy bin), F (window probability), iso (sphere-averaged factor)
Signal:   ev, phi, psi, dpf, below, iE, iA, iT, wt, ok0
Functions: cut(...), snr2(...)
"""
import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tra_quality import E_EDGES

sigdir, pdir, outdir = sys.argv[1:4]

# ---------------------------------------------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------------------------------------------
NE = 6                                     # measured-energy bins (E_EDGES: 6 log bins, 200-5000 keV)
NA, DA = 60, 0.5                           # |ARM| window grid: NA half-widths in steps of DA deg (0.5 ... 30 deg)
NT = 36                                    # zenith-angle bins of 5 deg (0 ... 180 deg)
W = DA * np.arange(1, NA + 1)              # ARM window half-widths [deg]
DEDGE = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 20])      # Distance bin edges [cm] -> 10 Distance bins
PC = (np.arange(72) + 0.5) * 2.5           # bin centres [deg] of the 72 bins of 2.5 deg used for phi and psi in the background histogram
TH = np.arange(0, 180.1, 5.0)              # zenith bin edges [deg]
TC = 0.5 * (TH[1:] + TH[:-1])              # zenith bin centres [deg]
SOL = np.sin(np.deg2rad(TC)) * np.deg2rad(5.0)         # solid angle / 2 pi of every zenith bin (converts counts to effective area)
LAB = [f'{E_EDGES[i]:.0f}–{E_EDGES[i + 1]:.0f} keV' for i in range(NE)]
COL = plt.cm.viridis(np.linspace(0, 0.9, NE))          # one colour per energy bin for the plots

# ---------------------------------------------------------------------------------------------------------------
# Background: week-1 .tra events summed over all components (written by bkg_psichi.py)
# ---------------------------------------------------------------------------------------------------------------
# Np axes: (energy bin, phi [72 x 2.5 deg], psi [72 x 2.5 deg], Distance bin [10], first-hit layer [4],
#           SQ > 5 flag [2], second-hit-below-first flag [2])
Np = sum(np.load(f)['N'] for f in glob.glob(pdir + '/*.npz'))

# ---------------------------------------------------------------------------------------------------------------
# Position-dependent window probability F[phi, psi, theta_s, w]
# ---------------------------------------------------------------------------------------------------------------
phi_b = np.deg2rad(PC)[:, None, None, None]            # broadcast axes: phi, psi, theta_s, w
psi_b = np.deg2rad(PC)[None, :, None, None]
theta_b = np.deg2rad(TC)[None, None, :, None]
w_b = np.deg2rad(W)[None, None, None, :]
alpha_lo = np.maximum(phi_b - w_b, 0)                  # allowed range of the source-to-psichi-direction angle alpha ...
alpha_hi = np.minimum(phi_b + w_b, np.pi)              # ... for |alpha - phi| < w
den = np.sin(theta_b) * np.sin(psi_b)
u1 = np.clip((np.cos(alpha_hi) - np.cos(theta_b) * np.cos(psi_b)) / den, -1, 1)
u2 = np.clip((np.cos(alpha_lo) - np.cos(theta_b) * np.cos(psi_b)) / den, -1, 1)
F = (np.arccos(u1) - np.arccos(u2)) / np.pi            # fraction of azimuths dchi for which the event is inside the window
del phi_b, psi_b, theta_b, w_b, alpha_lo, alpha_hi, den, u1, u2

# Sphere-averaged window probability (the "isotropic" factor of the all-sky studies): fraction of the sphere within w of the ring
iso = (np.cos(np.maximum(np.deg2rad(PC)[:, None] - np.deg2rad(W)[None, :], 0))
       - np.cos(np.minimum(np.deg2rad(PC)[:, None] + np.deg2rad(W)[None, :], np.pi))) / 2

# Sanity check: averaging F over the sphere of source positions (weight sin(theta) dtheta / 2) must reproduce ``iso``.
avg = (F * (np.sin(np.deg2rad(TC)) * np.deg2rad(5.0) / 2)[None, None, :, None]).sum(2)                           # (phi, psi, w)
print('sanity (sphere average of the position-dependent window fraction vs the isotropic factor, phi=60 deg, w=5 deg): %.4f vs %.4f'
      % (avg[24, :, 9].mean(), iso[24, 9]))

# ---------------------------------------------------------------------------------------------------------------
# Signal: response-simulation events (written by resp_signal.py, one .npz per simulation file)
# ---------------------------------------------------------------------------------------------------------------
signal_keys = ('E', 'arm', 'Etrue', 'theta_src', 'phi', 'd01', 'sq', 'cat1', 'z1', 'z2')
ev = {k: np.concatenate([np.load(f)[k] for f in sorted(glob.glob(sigdir + '/*.npz'))]) for k in signal_keys}

phi = np.rad2deg(ev['phi'].astype(float))              # reconstructed Compton scatter angle [deg]
# Polar angle psi of the psichi direction from the spacecraft +z axis: psichi = -(scattered direction), and the
# z component of the scattered direction is (z2 - z1) / |hit2 - hit1|.
psi = np.rad2deg(np.arccos(np.clip(-(ev['z2'].astype(float) - ev['z1']) / ev['d01'], -1, 1)))
dpf = np.abs(psi - phi)                                # |psi - phi|: smallest zenith angle reached by the event's ring (upward-going filter variable)
below = ev['z2'] < ev['z1']                            # second hit below the first hit

iE = np.digitize(ev['E'], E_EDGES) - 1                 # measured-energy bin of every event
ok0 = (iE >= 0) & (iE < NE)                            # events inside the 200-5000 keV range
iA = np.clip((np.abs(ev['arm']) / DA).astype(int), 0, NA)           # |ARM| bin (the last bin, NA, holds |ARM| > NA * DA deg)
iT = np.clip((ev['theta_src'] / 5).astype(int), 0, NT - 1)           # zenith bin of the true source direction
wt = (ev['Etrue'].astype(float) / 1000.0) ** -float(os.environ.get('GAMMA', '2.1'))   # re-weights the flat simulated spectrum to E^-GAMMA


# ---------------------------------------------------------------------------------------------------------------
# Cuts
# ---------------------------------------------------------------------------------------------------------------
def cut(dmin=0, lay=(0, 1, 2, 3), phimax=181, sq5=True, down=None, zcap=None):
    """Event-quality cuts, applied consistently to the signal events and to the background histogram.

    dmin    minimum Distance |hit2 - hit1| [cm]
    lay     allowed first-hit layers (0 = bottom ... 3 = top)
    phimax  keep events with phi < phimax [deg]
    sq5     keep events with at most 5 hits
    down    if true, keep only events whose second hit is below the first
    zcap    upward-going filter: keep events with |psi - phi| < zcap [deg], i.e. whose Compton ring reaches
            within zcap of the spacecraft zenith

    Returns (sm, bm): ``sm`` a boolean mask over the signal events and ``bm`` a boolean mask with the shape of ``Np``.
    """
    # --- signal events
    sm = (ev['d01'] >= dmin) & np.isin(ev['cat1'] // 3, lay) & (phi < phimax)
    # --- background cells; every axis of Np gets its own broadcast shape
    distance_ok = np.arange(10) >= (np.digitize(dmin, DEDGE) if dmin else 0)
    m = (distance_ok.reshape(1, 1, 1, 10, 1, 1, 1)
         & np.isin(np.arange(4), lay).reshape(1, 1, 1, 1, 4, 1, 1)
         & (PC < phimax).reshape(1, 72, 1, 1, 1, 1, 1))
    if sq5:
        sm &= ev['sq'] <= 5
        m = m & (np.arange(2) == 0).reshape(1, 1, 1, 1, 1, 2, 1)        # SQ > 5 flag == 0
    if down:
        sm &= below
        m = m & (np.arange(2) == 1).reshape(1, 1, 1, 1, 1, 1, 2)       # second-hit-below flag == 1
    if zcap is not None:                                                # upward-going filter
        sm &= dpf < zcap
        m = m & (np.abs(PC[:, None] - PC[None, :]) < zcap).reshape(1, 72, 72, 1, 1, 1, 1)   # |psi - phi| < zcap on the bin centres
    return sm, np.broadcast_to(m, Np.shape)


Nt = Np.sum((1, 2, 3, 4, 5, 6))                        # total background events per energy bin (denominator of the rejection)


# ---------------------------------------------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------------------------------------------
def snr2(sm, bm):
    """SNR^2 of a zenith transit for the cut defined by the masks ``sm`` (signal events) and ``bm`` (background cells).

    Returns (snr2, rejected):
      snr2      (NE, NT) array, SNR^2 of every zenith bin and energy bin; sum over the bins of interest and take the square root
      rejected  (NE,) fraction of the background events removed by the cut
    """
    ok = ok0 & sm
    # Signal counts per (energy, zenith, |ARM| bin), cumulated over |ARM| -> signal inside |ARM| < W[k]; divided by the zenith-bin
    # solid angle to get a quantity proportional to the effective area.
    counts = np.bincount(((iE * NT + iT) * (NA + 1) + iA)[ok], weights=wt[ok], minlength=NE * NT * (NA + 1)).reshape(NE, NT, NA + 1)
    A = np.cumsum(counts, -1)[..., :NA] / SOL[None, :, None]            # (NE, NT, NA)
    # Background cells surviving the cut, summed over everything except phi and psi, then folded with the window probability.
    surviving = (Np * bm).sum((3, 4, 5, 6))                            # (NE, phi, psi)
    out = np.zeros((NE, NT))
    for i in range(NE):
        B = np.einsum('pq,pqtk->tk', surviving[i], F)                   # (NT, NA): background inside |ARM| < W[k] for a source in zenith bin t
        out[i] = (A[i] ** 2 / np.maximum(B, 1e-30)).max(1)              # best window for every zenith bin
    return out, 1 - (Np * bm).sum((1, 2, 3, 4, 5, 6)) / Nt
