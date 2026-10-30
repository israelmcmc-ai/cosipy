"""S/sqrt(B) vs event quality cuts (Distance = lever arm LA, number of hits, depth of the first / second hit), with the ARM window optimised.

Signal: isotropic far-field simulations with true direction (signal.npz files of resp_signal.py), weighted with E_true^-GAMMA (Crab-like).
        ARM per event = angle(true direction, scattered direction) - Compton angle.
Background: week-1 .tra histograms of bkg_quality.py; for a source direction uniformly distributed on the sphere the expected number of background
        events with |ARM| < w is B(w) = sin(w) * sum(sin(phi)) (the fraction of the sphere within +-w of the event circle), so no direction is needed.
Only percentage changes of S/sqrt(B) relative to the no-cut baseline (same ARM optimisation) are meaningful.
"""
import glob
import os
import numpy as np
from tra_quality import E_EDGES

NL, NS, NC, NE = 120, 6, 12, 6
DL = 0.25                      # cm, LA bin
DA = 0.5                       # deg, ARM bin
NA = 60                        # ARM bins 0..30 deg
GAMMA = 2.1
W_GRID = DA * np.arange(1, NA + 1)                    # window half-widths [deg]
SINW = np.sin(np.deg2rad(W_GRID))


def signal_files(sigdir):
    return sorted(glob.glob(os.path.join(sigdir, '*.npz')))


def signal_hist(files, gamma=GAMMA, theta_max=None):
    """Weighted signal histogram S[E, LA, SQ, cat1, cat2, |ARM| bin] (float32) and sum of squared weights per energy bin."""
    H = np.zeros((NE, NL, NS, NC, NC, NA + 1), dtype=np.float32)      # last ARM bin = |ARM| > 30 deg
    w2 = np.zeros(NE)
    for f in files:
        z = np.load(f)
        E, arm = z['E'], np.abs(z['arm']); w = (z['Etrue'].astype(np.float64) / 1000.0) ** (-gamma)
        ok = np.ones(len(arm), bool)
        if theta_max is not None:
            ok &= z['theta_src'] < theta_max
        iE = np.digitize(E, E_EDGES) - 1
        iL = np.clip((z['LA'] / DL).astype(int), 0, NL - 1)
        iS = np.clip(z['sq'].astype(int), 2, 7) - 2
        iA = np.clip((arm / DA).astype(int), 0, NA)
        flat = ((((iE * NL + iL) * NS + iS) * NC + z['cat1']) * NC + z['cat2']) * (NA + 1) + iA
        H += np.bincount(flat[ok], weights=w[ok], minlength=H.size).reshape(H.shape).astype(np.float32)
        for i in range(NE):
            m = ok & (iE == i); w2[i] += (w[m] ** 2).sum()
    return H, w2


def bkg_hist(bkgdir, components=None):
    N = np.zeros((NE, NL, NS, NC, NC)); S = np.zeros_like(N)
    per = {}
    for f in sorted(glob.glob(os.path.join(bkgdir, '*.npz'))):
        name = os.path.basename(f)[:-4]
        if components is not None and name not in components:
            continue
        z = np.load(f); per[name] = (z['N'], z['SINPHI']); N += z['N']; S += z['SINPHI']
    return N, S, per


class Cut:
    """A cut on the discrete grid: la_min [cm], sq_max, allowed layers of the first hit, depth types excluded for hit 1 / hit 2
    (types: 1 = middle plane, 2 = within 0.1 cm of a surface), sq_eq (exact number of hits)."""
    def __init__(self, la_min=0.0, sq_max=7, layers1=(0, 1, 2, 3), excl1=(), excl2=(), sq_eq=None, layers2=(0, 1, 2, 3)):
        l = np.arange(NL) * DL >= la_min - 1e-9
        s = np.arange(2, 8) <= sq_max
        if sq_eq is not None:
            s &= np.arange(2, 8) == sq_eq
        def catmask(layers, excl):
            c = np.arange(NC); return (np.isin(c // 3, layers)) & (~np.isin(c % 3, excl))
        self.mL, self.mS = l.astype(float), s.astype(float)
        self.m1, self.m2 = catmask(layers1, excl1).astype(float), catmask(layers2, excl2).astype(float)

    def apply_bkg(self, A):          # A: (NE, NL, NS, NC, NC) -> (NE,)
        return np.einsum('elscd,l,s,c,d->e', A, self.mL, self.mS, self.m1, self.m2, optimize=True)

    def apply_sig(self, Hc):         # Hc: (NE, NL, NS, NC, NC, NA) cumulative in the last axis -> (NE, NA)
        return np.einsum('elscdk,l,s,c,d->ek', Hc, self.mL, self.mS, self.m1, self.m2, optimize=True)


def cumulative(H):
    return np.cumsum(H, axis=-1, dtype=np.float32)


def fom(cut, Hc, bN, bS):
    """Per energy bin: best S/sqrt(B) over the ARM window, the window, S and B at the optimum, the background events kept."""
    Sfull = cut.apply_sig(Hc).astype(np.float64)              # (NE, NA + 1): window |ARM| < w for w in W_GRID, last = all events
    S = Sfull[:, :NA]
    Bs = cut.apply_bkg(bS)                                    # sum sin(phi)
    Bn = cut.apply_bkg(bN)
    B = Bs[:, None] * SINW[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        f = S / np.sqrt(B)
    f[~np.isfinite(f)] = 0
    k = f.argmax(1)
    i = np.arange(NE)
    return dict(fom=f[i, k], w=W_GRID[k], S=S[i, k], B=B[i, k], Bn=Bn, Bs=Bs, Sall=Sfull[:, -1],
                fom_nowindow=Sfull[:, -1] / np.sqrt(np.maximum(Bn, 1e-30)))


def combined(fe):
    return np.sqrt((np.asarray(fe) ** 2).sum())
