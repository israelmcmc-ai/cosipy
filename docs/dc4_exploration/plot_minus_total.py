"""Subtract a component's *total* spacecraft-frame distribution from its psichi matrix.

The template T is the SC-frame psichi map summed over all energies and phi (of the
given mode, or of all events for the SC-frame plot). For each (energy, phi) slice it
is scaled so its total counts match the slice, and subtracted:
    residual = slice - T * (sum slice / sum T)
  * SC frame:    residual in SC coordinates.
  * Earth frame: per survey mode, T (mode events only) is rotated into the Earth
                 frame with the per-sample attitude (see plot_decouple_full.py)
                 and subtracted from that mode's Earth-frame slice.

Usage: python plot_minus_total.py hist.npz orientation.fits outdir [smooth_fwhm_deg]
"""
import sys
import os
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_dc4 as P
import plot_decouple_full as F


def matrix(z, residual, right, title, path, fwhm, equator=None):
    """residual(e0, e1, p0, p1) -> map to draw (diverging) for each matrix panel."""
    nr, nc = len(P.E_SLICES), len(P.PHI_SLICES)
    fig = plt.figure(figsize=(3.4 * nc, 2.1 * nr + 0.8))
    for r, (e0, e1) in enumerate(P.E_SLICES):
        for c, (p0, p1) in enumerate(P.PHI_SLICES):
            ax = fig.add_subplot(nr, nc, r * nc + c + 1, projection='mollweide')
            res = P.smooth(residual(e0, e1, p0, p1), fwhm)
            lim = np.percentile(np.abs(res), 99.5)
            if lim < 1e-9:
                lim = 1
                ax.text(0, 0, 'no events', ha='center', va='center', fontsize=9)
            mesh = P.render_map(ax, res, right, vmin=-lim, vmax=lim, cmap='RdBu_r')
            if equator is not None:
                ax.plot(*P.great_circle_xy(*equator, right), color='k', lw=0.7, ls='--')
            fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8,
                         fraction=0.05).ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(f'φ = {np.rad2deg(z["phi_edges"][p0]):.0f}–{np.rad2deg(z["phi_edges"][p1]):.0f}°', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{z["e_edges"][e0]:.0f}–{z["e_edges"][e1]:.0f} keV', fontsize=10, labelpad=14)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=90, bbox_inches='tight')
    plt.close(fig)


def main(npz, orifile, outdir, fwhm=4.0):
    os.makedirs(outdir, exist_ok=True)
    z = np.load(npz)
    nside = int(z['nside'])
    md = float(z['min_dist']) if 'min_dist' in z else 0.0
    cut = f' [Distance ≥ {md:g} cm]' if md > 0 else ''
    sm = f', Gaussian-smoothed (FWHM {fwhm:g}°)' if fwhm else ''
    sl = lambda h, e0, e1, p0, p1: h[e0:e1, p0:p1].sum(axis=(0, 1)).astype(float)

    # --- spacecraft frame (all events)
    h = z['spacecraft']; T = h.sum(axis=(0, 1)).astype(float)
    matrix(z, lambda *s: sl(h, *s) - T * sl(h, *s).sum() / T.sum(), False,
           f'Spacecraft frame: slice minus scaled total (∑E ∑φ) distribution, counts/pixel{sm}{cut}',
           os.path.join(outdir, 'psichi_maps_spacecraft_minus_total.png'), fwhm)

    # --- Earth frame, per survey mode
    ori = fits.open(orifile)[1].data
    slices = [(e0, e1, p0, p1) for (e0, e1) in P.E_SLICES for (p0, p1) in P.PHI_SLICES]
    for name in ('north', 'south'):
        hs = z[f'spacecraft_{name}']; he = z[f'earth_{name}']
        Tm = hs.sum(axis=(0, 1)).astype(float)
        alpha = {s: sl(he, *s).sum() / Tm.sum() for s in slices}
        models = F.rotation_models(z, ori, name, nside,
                                   templates={s: (alpha[s] * Tm).astype(np.float32) for s in slices})
        zd = tuple(z[f'zdir_{name}'])
        matrix(z, lambda *s, m=models, he=he: sl(he, *s) - m[s], True,
               f'Survey {name}, Earth frame: slice minus rotated scaled total SC distribution, counts/pixel{sm}{cut}'
               '\n(dashed = spacecraft equator)',
               os.path.join(outdir, f'psichi_maps_earth_minus_total_sc_survey_{name}.png'), fwhm, equator=zd)
        print(name, 'done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 4.0)
