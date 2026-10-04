"""Spacecraft-frame psichi maps for narrow (1 deg) phi bins and a few energy bands.

Bins all events with Distance >= min_dist (cm) at nside 32, for the energy bands
and phi values below, plus the total (all E, all phi) map used as a template.
Draws (a) the raw maps and (b) the maps minus the total scaled to each panel.

Usage: python narrow_phi_sc.py events.fits out_prefix [min_distance_cm] [smooth_fwhm_deg]
"""
import sys
import numpy as np
import healpy as hp
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_dc4 as P

NSIDE = 32
E_BANDS = [(100., 251.), (251., 631.), (631., 1585.)]   # keV
PHI_VALUES = [40, 70, 100, 130, 160]                      # deg; bins are [phi, phi+1)
CHUNK = 2_000_000


def bin_events(evfile, min_dist):
    d = fits.open(evfile, memmap=False)[1].data
    npix = hp.nside2npix(NSIDE)
    h = np.zeros((len(E_BANDS), len(PHI_VALUES), npix))
    tot = np.zeros(npix)
    for s in range(0, len(d), CHUNK):
        sl = slice(s, s + CHUNK)
        ok = np.asarray(d['Distance'][sl]) >= min_dist
        E = np.asarray(d['Energies'][sl]); phi = np.rad2deg(np.asarray(d['Phi'][sl]))
        pix = hp.ang2pix(NSIDE, np.clip(np.asarray(d['Psi local'][sl]), 0, np.pi), np.asarray(d['Chi local'][sl]))
        tot += np.bincount(pix[ok], minlength=npix)
        for i, (e0, e1) in enumerate(E_BANDS):
            for j, p in enumerate(PHI_VALUES):
                sel = ok & (E >= e0) & (E < e1) & (phi >= p) & (phi < p + 1)
                h[i, j] += np.bincount(pix[sel], minlength=npix)
    return h, tot


def draw(h, tot, fwhm, subtract, title, path):
    nr, nc = len(E_BANDS), len(PHI_VALUES)
    fig = plt.figure(figsize=(3.6 * nc, 2.3 * nr + 0.8))
    for i, (e0, e1) in enumerate(E_BANDS):
        for j, p in enumerate(PHI_VALUES):
            ax = fig.add_subplot(nr, nc, i * nc + j + 1, projection='mollweide')
            m = h[i, j]
            if subtract:
                m = m - tot * m.sum() / tot.sum()
            m = P.smooth(m, fwhm)
            if subtract:
                lim = np.percentile(np.abs(m), 99.5)
                mesh = P.render_map(ax, m, False, vmin=-lim, vmax=lim, cmap='RdBu_r')
            else:
                mesh = P.render_map(ax, m, False, vmin=0, vmax=np.percentile(m, 99.7))
            fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8,
                         fraction=0.05).ax.tick_params(labelsize=6)
            ax.set_title((f'φ = {p}–{p + 1}°\n' if i == 0 else '') + f'N={int(h[i, j].sum()):,}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{e0:.0f}–{e1:.0f} keV', fontsize=10, labelpad=14)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=90, bbox_inches='tight')
    plt.close(fig)


def main(evfile, prefix, min_dist=0.0, fwhm=4.0):
    h, tot = bin_events(evfile, min_dist)
    np.savez_compressed(prefix + '.npz', h=h, tot=tot, e_bands=E_BANDS, phi_values=PHI_VALUES, min_dist=min_dist)
    cut = f' [Distance ≥ {min_dist:g} cm]' if min_dist > 0 else ''
    sm = f', Gaussian-smoothed (FWHM {fwhm:g}°)' if fwhm else ''
    draw(h, tot, fwhm, False, f'Spacecraft frame, 1° φ bins, nside {NSIDE}: counts/pixel{sm}{cut}', prefix + '_raw.png')
    draw(h, tot, fwhm, True, f'Spacecraft frame, 1° φ bins: minus scaled total (∑E ∑φ) distribution{sm}{cut}',
         prefix + '_minus_total.png')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.0,
         float(sys.argv[4]) if len(sys.argv) > 4 else 4.0)
