"""Spacecraft-frame psichi matrices only (no Earth-frame step) for binned components.

For each <name>.npz in bindir (from run_components.sh) writes, flat in plotdir (suffix = component name):
  psichi_maps_spacecraft_<name>.png               energy (rows) x phi (columns) matrix
  minus_total_psichi_maps_spacecraft_<name>.png   each slice minus the scaled total (sum over E and phi) map
(the minus_total_ files get their own prefix so that psichi_maps_spacecraft_*.png matches only the plain matrices)

Usage: python plot_sc_components.py bindir plotdir [smooth_fwhm_deg] [name ...]
"""
import sys
import os
import glob
import numpy as np
import plot_dc4 as P
import plot_minus_total as M


def one(npz, outdir, fwhm, name):
    os.makedirs(outdir, exist_ok=True)
    z = np.load(npz)
    md = float(z['min_dist'])
    P.CUT = f' [Distance ≥ {md:g} cm]' if md > 0 else ''
    h = z['spacecraft']
    n = int(h.sum())
    P.plot_maps({'spacecraft': h}, 'spacecraft', z['e_edges'], z['phi_edges'], int(z['nside']), outdir, fwhm, tag=f'_{name}')
    T = h.sum(axis=(0, 1)).astype(float)
    sl = lambda e0, e1, p0, p1: h[e0:e1, p0:p1].sum(axis=(0, 1)).astype(float)
    sm = f', Gaussian-smoothed (FWHM {fwhm:g}°)' if fwhm else ''
    M.matrix(z, lambda *s: sl(*s) - T * sl(*s).sum() / T.sum(), False,
             f'Spacecraft frame: slice minus scaled total (∑E ∑φ) distribution, counts/pixel{sm}{P.CUT}',
             os.path.join(outdir, f'minus_total_psichi_maps_spacecraft_{name}.png'), fwhm)
    return n


if __name__ == '__main__':
    bindir, plotdir = sys.argv[1], sys.argv[2]
    fwhm = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    names = sys.argv[4:] or [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(bindir, '*.npz')))]
    for name in names:
        print(name, one(os.path.join(bindir, name + '.npz'), plotdir, fwhm, name), 'events in range', flush=True)
