"""Exploratory plots of the binned DC4 Compton data space (see bin_dc4.py).

For each frame (galactic, spacecraft, earth) this makes
  0. overview: psichi map summed over energy and phi, with the psichi
     "slice" disks of plot 2 marked,
  1. psichi maps for slices of energy (rows) and phi (columns),
  2. energy vs phi for slices of psichi (disks around chosen directions).

Usage: python plot_dc4.py hist.npz outdir [smooth_fwhm_deg]
"""
import sys
import os
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

CUT = ''   # title suffix describing the event selection

FRAMES = {
    # name: (long title, lon label, lat label, longitude increases to the right?)
    'galactic':   ('Galactic', 'l', 'b', False),
    'spacecraft': ('Spacecraft', r'$\chi$', r'$90^\circ-\psi$', False),
    'earth':      ('Earth horizon (alt-az)', 'azimuth (N=0, E=90)', 'altitude', True),
}

# (name, lon [deg], lat [deg]) of the psichi disk centres
SLICE_DIRS = {
    'galactic': [('Galactic centre', 0, 0), ('Crab', 184.56, -5.78),
                 ('Cyg X-1', 71.33, 3.07), ('North Gal. pole', 0, 90)],
    'spacecraft': [('+z (boresight)', 0, 90), ('-z', 0, -90),
                   ('+x', 0, 0), ('+y', 90, 0)],
    'earth': [('zenith', 0, 90), ('nadir', 0, -90),
              ('horizon, North', 0, 0), ('horizon, East', 90, 0)],
}
DISK_RADIUS = 15.0   # deg

# slices of the (merged) energy / phi bins used for plot 1: [first, last) bin
E_SLICES = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
PHI_SLICES = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36)]  # 30 deg each


def render_map(ax, m, lon_right, vmin=None, vmax=None, cmap='viridis', norm=None):
    """Draw a HEALPix map (lon, lat convention of the frame) on a Mollweide axis."""
    nside = hp.npix2nside(len(m))
    x = np.linspace(-np.pi, np.pi, 361)
    y = np.linspace(-np.pi / 2, np.pi / 2, 181)
    xc = 0.5 * (x[1:] + x[:-1]); yc = 0.5 * (y[1:] + y[:-1])
    X, Y = np.meshgrid(xc, yc)
    lon = np.mod(X if lon_right else -X, 2 * np.pi)
    img = m[hp.ang2pix(nside, np.pi / 2 - Y, lon)]
    kw = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(x, y, img, cmap=cmap, shading='flat', rasterized=True, **kw)
    ticks = np.deg2rad([-120, -60, 0, 60, 120])
    ax.set_xticks(ticks)
    lab = [int(round(np.mod(np.rad2deg(t if lon_right else -t), 360))) % 360 for t in ticks]
    ax.set_xticklabels([f'{v}°' for v in lab], fontsize=6)
    ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
    ax.set_yticklabels(['-60°', '-30°', '0°', '30°', '60°'], fontsize=6)
    ax.grid(color='w', alpha=0.25, lw=0.4)
    return mesh


def plot_xy(lon_deg, lat_deg, lon_right):
    lon = np.deg2rad(lon_deg)
    lon = (lon + np.pi) % (2 * np.pi) - np.pi
    return (lon if lon_right else -lon), np.deg2rad(lat_deg)


def smooth(m, fwhm_deg):
    if not fwhm_deg:
        return m.astype(float)
    return hp.smoothing(m.astype(float), fwhm=np.deg2rad(fwhm_deg))


def disk_mask(nside, lon_deg, lat_deg, radius_deg):
    c = hp.ang2vec(np.deg2rad(90 - lat_deg), np.deg2rad(lon_deg))
    return np.isin(np.arange(hp.nside2npix(nside)),
                   hp.query_disc(nside, c, np.deg2rad(radius_deg)))


def overview(hist, nside, outdir, fwhm):
    fig = plt.figure(figsize=(18, 4.6))
    for i, (key, (title, *_rest)) in enumerate(FRAMES.items()):
        right = FRAMES[key][3]
        ax = fig.add_subplot(1, 3, i + 1, projection='mollweide')
        m = smooth(hist[key].sum(axis=(0, 1)), fwhm)
        mesh = render_map(ax, m, right, vmin=0, vmax=m.max())
        for name, lo, la in SLICE_DIRS[key]:
            xs, ys = plot_xy(lo, la, right)
            ax.plot(xs, ys, 'o', mfc='none', mec='r', ms=8)
            ax.annotate(name, (xs, ys), color='w', fontsize=7, ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points')
        ax.set_title(f'{title}: all events{CUT}, ∑E ∑φ', fontsize=10)
        fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8, label='counts / pixel')
    fig.savefig(os.path.join(outdir, 'overview_psichi.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)


def great_circle_xy(zaz, zalt, right, n=1441):
    """Plot coordinates of the great circle 90 deg from the direction (zaz, zalt) [deg]
    (the spacecraft equator when (zaz, zalt) is the z axis), sorted in longitude."""
    zv = vec_lonlat(zaz, zalt)
    a = np.cross(zv, [0, 0, 1.0]); a /= np.linalg.norm(a)
    b = np.cross(zv, a)
    ang = np.linspace(0, 2 * np.pi, n)
    pts = np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b
    lon = np.arctan2(pts[:, 1], pts[:, 0]); lat = np.arcsin(pts[:, 2])
    x = lon if right else -lon
    o = np.argsort(x)
    return x[o], lat[o]


def vec_lonlat(lon_deg, lat_deg):
    lo, la = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def plot_maps(hist, key, e_edges, phi_edges, nside, outdir, fwhm, tag='', equator=None, label=''):
    """equator: optional (az, alt) [deg] of the spacecraft z axis; its equator is overlaid."""
    title, _, _, right = FRAMES[key]
    h = hist[key]
    nr, nc = len(E_SLICES), len(PHI_SLICES)
    fig = plt.figure(figsize=(3.4 * nc, 2.1 * nr + 0.8))
    for r, (e0, e1) in enumerate(E_SLICES):
        for c, (p0, p1) in enumerate(PHI_SLICES):
            ax = fig.add_subplot(nr, nc, r * nc + c + 1, projection='mollweide')
            m = smooth(h[e0:e1, p0:p1].sum(axis=(0, 1)), fwhm)
            vmax = np.percentile(m, 99.7)
            mesh = render_map(ax, m, right, vmin=0, vmax=vmax if vmax > 1e-9 else 1)
            if vmax <= 1e-9:
                ax.text(0, 0, 'no events', ha='center', va='center', color='w', fontsize=9)
            if equator is not None:
                ax.plot(*great_circle_xy(*equator, right), color='r', lw=0.9, ls='--')
            fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8, fraction=0.05).ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(f'φ = {np.rad2deg(phi_edges[p0]):.0f}–{np.rad2deg(phi_edges[p1]):.0f}°', fontsize=10)
            if c == 0:
                ax.set_ylabel(f'{e_edges[e0]:.0f}–{e_edges[e1]:.0f} keV', fontsize=10, labelpad=14)
    sm = f', Gaussian-smoothed (FWHM {fwhm:g}°)' if fwhm else ''
    fig.suptitle(f'{title} frame: ψχ maps per energy (rows) and φ (columns) slice{sm}{CUT}' + (f'\n{label}' if label else ''), y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'psichi_maps_{key}{tag}.png'), dpi=90, bbox_inches='tight')
    plt.close(fig)


def plot_e_phi(hist, key, e_edges, phi_edges, nside, outdir):
    title = FRAMES[key][0]
    h = hist[key]
    dirs = SLICE_DIRS[key]
    fig, axs = plt.subplots(1, len(dirs), figsize=(4.6 * len(dirs), 4.2), squeeze=False)
    for ax, (name, lo, la) in zip(axs[0], dirs):
        mask = disk_mask(nside, lo, la, DISK_RADIUS)
        img = h[:, :, mask].sum(axis=2).astype(float)
        mesh = ax.pcolormesh(np.rad2deg(phi_edges), e_edges, img, cmap='viridis',
                             norm=LogNorm(vmin=max(img[img > 0].min(), 1), vmax=img.max()),
                             rasterized=True)
        ax.set_yscale('log')
        ax.set_xlabel('Compton scattering angle φ [deg]')
        ax.set_title(f'{name}\n({lo:g}°, {la:g}°), r<{DISK_RADIUS:g}°, N={int(img.sum()):,}', fontsize=9)
        fig.colorbar(mesh, ax=ax, label='counts')
    axs[0, 0].set_ylabel('Measured energy [keV]')
    fig.suptitle(f'{title} frame: energy vs φ for ψχ slices (disks){CUT}', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'energy_vs_phi_{key}.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_e_phi_modes(modes, nside, e_edges, phi_edges, outdir):
    """modes: list of (label, Earth-frame histogram) -> one row of disks per mode."""
    dirs = SLICE_DIRS['earth']
    fig, axs = plt.subplots(len(modes), len(dirs), figsize=(4.6 * len(dirs), 4.0 * len(modes)), squeeze=False)
    for r, (label, h) in enumerate(modes):
        for ax, (name, lo, la) in zip(axs[r], dirs):
            img = h[:, :, disk_mask(nside, lo, la, DISK_RADIUS)].sum(axis=2).astype(float)
            mesh = ax.pcolormesh(np.rad2deg(phi_edges), e_edges, img, cmap='viridis',
                                 norm=LogNorm(vmin=max(img[img > 0].min(), 1), vmax=img.max()), rasterized=True)
            ax.set_yscale('log')
            ax.set_xlabel('Compton scattering angle φ [deg]')
            ax.set_title(f'{label}: {name}\n({lo:g}°, {la:g}°), r<{DISK_RADIUS:g}°, N={int(img.sum()):,}', fontsize=9)
            fig.colorbar(mesh, ax=ax, label='counts')
        axs[r, 0].set_ylabel('Measured energy [keV]')
    fig.suptitle(f'Earth horizon frame: energy vs φ for ψχ slices (disks), by survey mode{CUT}', y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'energy_vs_phi_earth_by_mode.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)


def main(npz, outdir, fwhm=4.0):
    global CUT
    os.makedirs(outdir, exist_ok=True)
    z = np.load(npz)
    md = float(z['min_dist']) if 'min_dist' in z else 0.0
    CUT = f' [Distance ≥ {md:g} cm]' if md > 0 else ''
    hist = {k: z[k] for k in FRAMES}
    nside = int(z['nside'])
    overview(hist, nside, outdir, fwhm)
    for key in FRAMES:
        plot_maps(hist, key, z['e_edges'], z['phi_edges'], nside, outdir, fwhm)
        plot_e_phi(hist, key, z['e_edges'], z['phi_edges'], nside, outdir)
    if 'earth_north' in z:
        modes = []
        for name in ('north', 'south'):
            zd = tuple(z[f'zdir_{name}'])
            lab = (f'Survey {name} (z axis at az={zd[0]:.0f}°, alt={zd[1]:.0f}°; '
                   'red dashed = spacecraft equator)')
            h = z[f'earth_{name}']
            plot_maps({'earth': h}, 'earth', z['e_edges'], z['phi_edges'], nside, outdir, fwhm,
                      tag=f'_survey_{name}', equator=zd, label=lab)
            modes.append((f'Survey {name}', h))
        plot_e_phi_modes(modes, nside, z['e_edges'], z['phi_edges'], outdir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 4.0)
