"""Plots of S/sqrt(B) vs Distance and ARM cuts from sensitivity.py histograms.

All numbers are percentage changes of S/sqrt(B) relative to the no-cut baseline
(no Distance cut, no ARM cut) in the same energy bin; "all bins" combines the
six bins in quadrature. Signal = Crab, background = DC4 total background.

Usage: python sensitivity_plots.py sens_crab.npz sens_bkg.npz outdir
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W_GRID = np.array([0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30])   # ARM half-widths [deg]


class Sens:
    def __init__(self, crab, bkg):
        s, b = np.load(crab), np.load(bkg)
        self.S, self.B = s['H'], b['H']
        self.e, self.d, self.a = s['e_edges'], s['d_edges'], s['a_edges']
        self.ac = 0.5 * (self.a[1:] + self.a[:-1])

    def counts(self, H, ie, dmin, w):
        """Events in energy bin ie with Distance >= dmin and |ARM| < w (w=None: no ARM cut)."""
        sel_d = self.d[:-1] >= dmin - 1e-9
        sel_a = np.ones(len(self.ac), bool) if w is None else np.abs(self.ac) < w - 1e-9
        return H[ie][np.ix_(sel_d, sel_a)].sum()

    def fom(self, ie, dmin=0., w=None):
        s, b = self.counts(self.S, ie, dmin, w), self.counts(self.B, ie, dmin, w)
        return s / np.sqrt(b) if b > 0 else np.nan

    def pct(self, ie, dmin=0., w=None):
        return 100 * (self.fom(ie, dmin, w) / self.fom(ie) - 1)

    def combined_pct(self, dmin=0., w=None):
        tot = lambda **k: np.sqrt(sum(self.fom(i, **k) ** 2 for i in range(len(self.e) - 1)))
        return 100 * (tot(dmin=dmin, w=w) / tot() - 1)

    def fwhm(self, ie):
        p = self.S[ie].sum(0); m = np.abs(self.ac) < 30
        return np.ptp(self.ac[m][p[m] > p[m].max() / 2])


def labels(s):
    return [f'{s.e[i]:.0f}–{s.e[i + 1]:.0f} keV' for i in range(len(s.e) - 1)]


def main(crab, bkg, outdir):
    os.makedirs(outdir, exist_ok=True)
    s = Sens(crab, bkg)
    nE = len(s.e) - 1
    lab = labels(s)
    cols = plt.cm.viridis(np.linspace(0, 0.9, nE))
    D_GRID = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10])

    # 1. ARM distributions
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for i, ax in enumerate(axs.flat):
        for H, name, c in ((s.S, 'Crab (signal)', 'C3'), (s.B, 'background', 'C0')):
            for dmin, ls in ((0, '-'), (1, '--')):
                sel = s.d[:-1] >= dmin - 1e-9
                p = H[i][sel].sum(0)[1:-1]
                ax.step(s.ac[1:-1], p / p.sum() / 0.25, where='mid', color=c, ls=ls,
                        label=f'{name}, Distance ≥ {dmin:g} cm')
        ax.set_yscale('log'); ax.set_title(lab[i]); ax.set_xlim(-30, 30)
        ax.set_xlabel('ARM [deg]'); ax.set_ylabel('fraction per deg (within ±30°)')
    axs[0, 0].legend(fontsize=7)
    fig.suptitle('ARM distributions relative to the Crab position (unit area within ±30°)')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'arm_distributions.png'), dpi=100); plt.close(fig)

    # 2. S/sqrt(B) vs minimum distance
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for k, (ax, mode) in enumerate(zip(axs, ('none', 'fwhm'))):
        for i in range(nE):
            w = None if mode == 'none' else s.fwhm(i)
            ax.plot(D_GRID, [100 * (s.fom(i, d, w) / s.fom(i, 0, w) - 1) for d in D_GRID], 'o-', color=cols[i], label=lab[i], ms=3)
        w_all = None
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('minimum Distance between 1st and 2nd hit [cm]')
        ax.set_ylabel('change in S/√B [%]')
        ax.set_title('no ARM cut' if mode == 'none' else 'with ARM window ±FWHM of the Crab ARM (per bin)')
    axs[0].legend(fontsize=8)
    fig.suptitle('Sensitivity vs Distance cut (relative to no Distance cut, same ARM selection)')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'sensitivity_vs_distance.png'), dpi=100); plt.close(fig)

    # 3. S/sqrt(B) vs ARM window half-width
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, dmin in zip(axs, (0, 1)):
        for i in range(nE):
            ax.plot(W_GRID, [100 * (s.fom(i, dmin, w) / s.fom(i) - 1) for w in W_GRID], 'o-', color=cols[i], label=lab[i], ms=3)
        ax.plot(W_GRID, [s.combined_pct(dmin, w) for w in W_GRID], 'k--', label='all bins (quadrature)')
        ax.set_xscale('log'); ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('ARM window half-width |ARM| < w [deg]')
        ax.set_ylabel('change in S/√B [%] vs no cuts')
        ax.set_title(f'Distance ≥ {dmin:g} cm')
    axs[0].legend(fontsize=8)
    fig.suptitle('Sensitivity vs ARM window (baseline: no Distance cut, no ARM cut)')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'sensitivity_vs_arm_window.png'), dpi=100); plt.close(fig)

    # 4. 2-D (Distance cut x ARM window) maps of % change
    fig, axs = plt.subplots(2, 4, figsize=(20, 8.5))
    best = []
    for i, ax in enumerate(axs.flat[:nE + 1]):
        f = (lambda d, w: s.pct(i, d, w)) if i < nE else (lambda d, w: s.combined_pct(d, w))
        M = np.array([[f(d, w) for d in D_GRID] for w in W_GRID])
        j, k = np.unravel_index(np.nanargmax(M), M.shape)
        best.append((lab[i] if i < nE else 'all bins', W_GRID[j], D_GRID[k], M[j, k]))
        lim = np.nanmax(np.abs(M))
        im = ax.pcolormesh(np.arange(len(D_GRID) + 1) - .5, np.arange(len(W_GRID) + 1) - .5, M, cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.plot(k, j, 'k*', ms=14)
        ax.set_xticks(range(len(D_GRID))); ax.set_xticklabels([f'{d:g}' for d in D_GRID], fontsize=7)
        ax.set_yticks(range(len(W_GRID))); ax.set_yticklabels([f'{w:g}' for w in W_GRID], fontsize=7)
        ax.set_xlabel('min Distance [cm]'); ax.set_ylabel('ARM half-width [deg]')
        ax.set_title(f'{best[-1][0]}: best {M[j, k]:+.0f}% (★)')
        fig.colorbar(im, ax=ax, label='change in S/√B [%]')
    axs.flat[-1].axis('off')
    fig.suptitle('S/√B change vs Distance cut and ARM window (baseline: no cuts)')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'sensitivity_distance_x_arm.png'), dpi=90); plt.close(fig)

    # table
    with open(os.path.join(outdir, 'summary.txt'), 'w') as f:
        f.write('bin | Crab events | bkg events | Crab ARM FWHM [deg] | best ARM half-width | best min Distance | S/sqrtB change at best\n')
        for i, (name, w, d, g) in enumerate(best):
            if i < nE:
                f.write(f'{name} | {int(s.S[i].sum())} | {int(s.B[i].sum())} | {s.fwhm(i):.1f} | {w:g} | {d:g} | {g:+.0f}%\n')
            else:
                f.write(f'{name} | | | | {w:g} | {d:g} | {g:+.0f}%\n')
    print(open(os.path.join(outdir, 'summary.txt')).read())


if __name__ == '__main__':
    main(*sys.argv[1:4])
