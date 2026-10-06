"""Spacecraft-frame psichi distributions projected on SC longitude and on SC latitude (1 deg bins, unsmoothed).

Same energy x phi matrix layout as plot_dc4.plot_maps for the spacecraft frame, but each panel is a 1-D histogram:
  * vs latitude (90 deg - psi, -90..90): events summed over all longitudes,
  * vs longitude (chi, 0..360):          events summed over all latitudes,
in 1 deg bins of counts with no smoothing (note: counts per latitude bin include the cos(lat) solid-angle factor).

Usage: python sc_lonlat.py events.fits[.gz]|hist.npz out_prefix [min_distance_cm]
 -> out_prefix.npz (2-D raw counts, kept so the plots can be redrawn),
    out_prefix_vs_latitude.png and out_prefix_vs_longitude.png
"""
import sys
import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_dc4 as P
import bin_dc4 as B
from sensitivity import stream_rows

LON_EDGES = np.arange(0., 360.01, 1.)
LAT_EDGES = np.arange(-90., 90.01, 1.)


def bin_events(evfile, min_dist):
    e_slice = np.full(len(B.E_EDGES) - 1, -1); p_slice = np.full(len(B.PHI_EDGES) - 1, -1)
    for k, (a, b) in enumerate(P.E_SLICES): e_slice[a:b] = k
    for k, (a, b) in enumerate(P.PHI_SLICES): p_slice[a:b] = k
    H = np.zeros((len(P.E_SLICES), len(P.PHI_SLICES), len(LON_EDGES) - 1, len(LAT_EDGES) - 1))
    raw = open(evfile, 'rb')
    stream = gzip.GzipFile(fileobj=raw) if evfile.endswith('.gz') else raw
    for c in stream_rows(stream):
        ie = np.digitize(c['E'].astype(float), B.E_EDGES) - 1
        ip = np.digitize(c['phi'].astype(float), B.PHI_EDGES) - 1
        ok = (ie >= 0) & (ie < len(e_slice)) & (ip >= 0) & (ip < len(p_slice)) & (c['dist'] >= min_dist)
        lon = np.rad2deg(c['chi'].astype(float)); lat = 90. - np.rad2deg(c['psi'].astype(float))
        ix = np.clip(np.digitize(lon, LON_EDGES) - 1, 0, len(LON_EDGES) - 2)
        iy = np.clip(np.digitize(lat, LAT_EDGES) - 1, 0, len(LAT_EDGES) - 2)
        s = ok & (e_slice[np.clip(ie, 0, None)] >= 0)
        idx = ((e_slice[ie[s]] * len(P.PHI_SLICES) + p_slice[ip[s]]) * (len(LON_EDGES) - 1) + ix[s]) * (len(LAT_EDGES) - 1) + iy[s]
        H += np.bincount(idx, minlength=H.size).reshape(H.shape)
    return H


def draw_1d(H, path, axis, cut):
    """axis='lat': sum over longitude; axis='lon': sum over latitude. One panel per (energy, phi) slice."""
    nr, nc = H.shape[:2]
    edges = LAT_EDGES if axis == 'lat' else LON_EDGES
    x = 0.5 * (edges[:-1] + edges[1:])
    fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.2 * nr + 0.8), squeeze=False)
    for r, (e0, e1) in enumerate(P.E_SLICES):
        for c, (p0, p1) in enumerate(P.PHI_SLICES):
            ax = axs[r, c]
            y = H[r, c].sum(axis=0) if axis == 'lat' else H[r, c].sum(axis=1)   # H[r, c] is (lon, lat)
            ax.step(edges[:-1], y, where='post', lw=0.7, color='C0')
            ax.fill_between(edges[:-1], np.clip(y - np.sqrt(y), 0, None), y + np.sqrt(y), step='post', color='C0', alpha=0.25, lw=0)
            ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(bottom=0)
            ax.set_xticks(range(-90, 91, 30) if axis == 'lat' else range(0, 361, 60)); ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(f'φ = {p0 * 5}–{p1 * 5}°', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{B.E_EDGES[e0]:.0f}–{B.E_EDGES[e1]:.0f} keV\ncounts / 1° bin', fontsize=8)
            if r == nr - 1:
                ax.set_xlabel('SC latitude 90°−ψ [deg]' if axis == 'lat' else 'SC longitude χ [deg]', fontsize=8)
    name = 'latitude 90°−ψ (summed over longitude)' if axis == 'lat' else 'longitude χ (summed over latitude)'
    fig.suptitle(f'Spacecraft frame: events vs {name}, 1° bins, no smoothing (band: ±√N){cut}', y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    src, prefix = sys.argv[1], sys.argv[2]
    md = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    if src.endswith('.npz'):
        z = np.load(src); H, md = z['H'], float(z['min_dist'])
    else:
        H = bin_events(src, md)
        np.savez_compressed(prefix + '.npz', H=H, lon_edges=LON_EDGES, lat_edges=LAT_EDGES, min_dist=md,
                            e_edges=B.E_EDGES, phi_edges=B.PHI_EDGES)
    print('events in slices:', int(H.sum()))
    cut = f' [Distance ≥ {md:g} cm]' if md > 0 else ''
    draw_1d(H, prefix + '_vs_latitude.png', 'lat', cut)
    draw_1d(H, prefix + '_vs_longitude.png', 'lon', cut)
