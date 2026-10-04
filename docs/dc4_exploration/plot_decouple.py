"""Approximately separate Earth-fixed from spacecraft-fixed structure in psichi.

For each survey mode independently (the z axis is ~fixed in the Earth frame
within a mode):
  1. take the spacecraft-frame psichi map of that mode's events,
  2. average it over spacecraft azimuth chi -> counts/pixel as a function of psi,
  3. paint that axisymmetric map onto the Earth (alt-az) frame about the mode's
     mean z axis, and
  4. subtract it from the Earth-frame map of the same events.
What is left is Earth-frame structure not explained by a chi-symmetric
spacecraft response (plus any chi-dependent spacecraft structure, which is
smeared by the changing azimuth of the Earth frame).

Usage: python plot_decouple.py hist.npz outdir [smooth_fwhm_deg]
(hist.npz from bin_dc4.py)
"""
import sys
import os
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_dc4 as P


def psi_profile(sc_map):
    """Azimuthal (chi) average of a HEALPix map: returns (ring colatitudes, mean per ring)."""
    nside = hp.npix2nside(len(sc_map))
    theta, _ = hp.pix2ang(nside, np.arange(len(sc_map)))
    ring_theta, inv = np.unique(np.round(theta, 9), return_inverse=True)
    mean = np.bincount(inv, weights=sc_map) / np.bincount(inv)
    return ring_theta, mean


def earth_model(sc_map, zdir_azalt):
    """Axisymmetric (about z) map built from the chi-average of sc_map, in Earth coords."""
    nside = hp.npix2nside(len(sc_map))
    ring_theta, mean = psi_profile(sc_map)
    v = np.array(hp.pix2vec(nside, np.arange(len(sc_map))))   # (lon, colat) = (az, 90-alt)
    zv = P.vec_lonlat(*zdir_azalt)
    psi = np.arccos(np.clip(zv @ v, -1, 1))
    return np.interp(psi, ring_theta, mean)


def decouple_maps(h_earth, h_sc, zdir, e_edges, phi_edges, outdir, fwhm, name, cut,
                  model_fn=None, what='χ-averaged spacecraft map', fname='psichi_maps_earth_minus_sc_survey_'):
    """model_fn(sc_slice_map, (e0, e1, p0, p1)) -> Earth-frame model; default = chi-average."""
    nr, nc = len(P.E_SLICES), len(P.PHI_SLICES)
    fig = plt.figure(figsize=(3.4 * nc, 2.1 * nr + 0.8))
    for r, (e0, e1) in enumerate(P.E_SLICES):
        for c, (p0, p1) in enumerate(P.PHI_SLICES):
            ax = fig.add_subplot(nr, nc, r * nc + c + 1, projection='mollweide')
            earth = h_earth[e0:e1, p0:p1].sum(axis=(0, 1)).astype(float)
            sc = h_sc[e0:e1, p0:p1].sum(axis=(0, 1)).astype(float)
            mod = model_fn(sc, (e0, e1, p0, p1)) if model_fn else earth_model(sc, zdir)
            res = P.smooth(earth - mod, fwhm)
            lim = np.percentile(np.abs(res), 99.5)
            if lim < 1e-9:
                lim = 1
                ax.text(0, 0, 'no events', ha='center', va='center', fontsize=9)
            mesh = P.render_map(ax, res, True, vmin=-lim, vmax=lim, cmap='RdBu_r')
            fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8,
                         fraction=0.05).ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(f'φ = {np.rad2deg(phi_edges[p0]):.0f}–{np.rad2deg(phi_edges[p1]):.0f}°', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{e_edges[e0]:.0f}–{e_edges[e1]:.0f} keV', fontsize=10, labelpad=14)
    sm = f', Gaussian-smoothed (FWHM {fwhm:g}°)' if fwhm else ''
    fig.suptitle(f'Survey {name}, Earth frame minus {what} (counts/pixel){sm}{cut}', y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{fname}{name}.png'), dpi=90, bbox_inches='tight')
    plt.close(fig)


def example(z, fwhm, outdir, cut, e_sl=(2, 4), p_sl=(12, 18), models=None, what='χ-averaged SC map, in Earth frame',
            fname='decoupling_example.png'):
    """Earth map | χ-averaged SC model | residual for one (energy, phi) slice, both modes."""
    fig = plt.figure(figsize=(16, 4.4 * 2))
    for r, name in enumerate(('north', 'south')):
        zd = tuple(z[f'zdir_{name}'])
        earth = z[f'earth_{name}'][e_sl[0]:e_sl[1], p_sl[0]:p_sl[1]].sum(axis=(0, 1)).astype(float)
        sc = z[f'spacecraft_{name}'][e_sl[0]:e_sl[1], p_sl[0]:p_sl[1]].sum(axis=(0, 1)).astype(float)
        mod = models[name][(e_sl[0], e_sl[1], p_sl[0], p_sl[1])] if models else earth_model(sc, zd)
        vmax = np.percentile(P.smooth(earth, fwhm), 99.7)
        panels = [('Earth frame', P.smooth(earth, fwhm), 'viridis', 0, vmax),
                  (what, P.smooth(mod, fwhm), 'viridis', 0, vmax)]
        res = P.smooth(earth - mod, fwhm); lim = np.percentile(np.abs(res), 99.5)
        panels.append(('difference', res, 'RdBu_r', -lim, lim))
        for c, (t, m, cm, lo, hi) in enumerate(panels):
            ax = fig.add_subplot(2, 3, r * 3 + c + 1, projection='mollweide')
            mesh = P.render_map(ax, m, True, vmin=lo, vmax=hi, cmap=cm)
            ax.plot(*P.great_circle_xy(*zd, True), color='k', lw=0.6, ls='--')
            ax.set_title(f'Survey {name}: {t}', fontsize=10)
            fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.06, shrink=0.8, label='counts / pixel')
    e0, e1 = z['e_edges'][e_sl[0]], z['e_edges'][e_sl[1]]
    fig.suptitle(f'{e0:.0f}–{e1:.0f} keV, φ = {np.rad2deg(z["phi_edges"][p_sl[0]]):.0f}–'
                 f'{np.rad2deg(z["phi_edges"][p_sl[1]]):.0f}° (dashed = spacecraft equator){cut}', y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=90, bbox_inches='tight')
    plt.close(fig)


def main(npz, outdir, fwhm=4.0):
    os.makedirs(outdir, exist_ok=True)
    z = np.load(npz)
    md = float(z['min_dist']) if 'min_dist' in z else 0.0
    cut = f' [Distance ≥ {md:g} cm]' if md > 0 else ''
    for name in ('north', 'south'):
        decouple_maps(z[f'earth_{name}'], z[f'spacecraft_{name}'], tuple(z[f'zdir_{name}']),
                      z['e_edges'], z['phi_edges'], outdir, fwhm, name, cut)
    example(z, fwhm, outdir, cut)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 4.0)
