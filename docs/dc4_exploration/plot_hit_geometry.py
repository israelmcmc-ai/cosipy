"""Geometry of the first->second hit displacement for events with distance > dmin (outputs of hit_positions.py with dmin).

Panels: azimuth of the lateral displacement (atan2(dy, dx)) split by lateral distance band; azimuth split by the relation of the two
detectors; (dx, dy) map; vertical separation dz; relation fractions; lateral separation; and a simple geometric Monte Carlo of the
2 x 2 detector layout: hit 1 uniform over the active area of one layer, hit 2 uniform over the same layer, kept if the lateral separation
is > dmin, weighted by exp(-rho / lambda) / rho^2 (probability for the scattered photon to interact at lateral distance rho in a thin sheet).
The Monte Carlo is scaled to the data and overlaid on the azimuth panel (for lambda values given).
Usage: python plot_hit_geometry.py hits_dir out.png [dmin_cm] [lambda_cm ...]
"""
import sys, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

XB = [(-8.39, -1.06), (1.06, 8.39)]          # detector extents in x and y [cm]
YB = [(-6.71, 0.62), (5.86, 13.20)]


def mc_azimuth(dmin, lam, n=4_000_000, seed=1):
    rng = np.random.default_rng(seed)
    def pt():
        ix = rng.integers(0, 2, n); iy = rng.integers(0, 2, n)
        x = np.where(ix == 0, rng.uniform(*XB[0], n), rng.uniform(*XB[1], n))
        y = np.where(iy == 0, rng.uniform(*YB[0], n), rng.uniform(*YB[1], n))
        return x, y, ix + 2 * iy
    x1, y1, d1 = pt(); x2, y2, d2 = pt()
    dx, dy = x2 - x1, y2 - y1; rho = np.hypot(dx, dy)
    w = np.exp(-rho / lam) / rho ** 2 * (rho > dmin)
    az = np.rad2deg(np.arctan2(dy, dx))
    rel = np.where(d1 == d2, 0, np.where(d1 // 2 == d2 // 2, 1, np.where(d1 % 2 == d2 % 2, 2, 3)))
    h = np.histogram(az, bins=np.arange(-180, 180.001, 0.5), weights=w)[0]
    fr = np.array([w[rel == r].sum() for r in range(4)]); return h / h.sum(), fr / fr.sum()


def main(hits_dir, out, dmin=8.0, lams=(2.0, 4.0)):
    Z = [np.load(f) for f in sorted(glob.glob(os.path.join(hits_dir, '*.npz')))]
    S = lambda k: sum(z[k] for z in Z)
    az_edges = Z[0]['az_edges']; az = 0.5 * (az_edges[1:] + az_edges[:-1])
    ne, nea = int(S('n_events')), int(S('n_events_all'))
    fig, axs = plt.subplots(2, 3, figsize=(18, 9.5))
    # (0,0) azimuth by lateral band
    for k, lab in enumerate(['lateral < 1 cm', '1–3 cm', '3–6 cm', '> 6 cm']):
        axs[0, 0].step(az, S('az_lat')[k], where='mid', lw=0.8, label=lab)
    axs[0, 0].legend(fontsize=8); axs[0, 0].set_xlabel('azimuth of the first→second hit lateral displacement [deg]'); axs[0, 0].set_ylabel('events / 0.5°')
    axs[0, 0].set_title(f'D > {dmin:g} cm: {ne / 1e3:.0f}k events ({100 * ne / nea:.1f}% of {nea / 1e6:.1f}M)')
    # (0,1) azimuth by detector relation + MC
    tot = S('az_lat').sum(0)
    rel_names = ['same detector', 'x-neighbour', 'y-neighbour', 'diagonal neighbour']
    for r in range(4):
        axs[0, 1].step(az, S('az_rel')[r], where='mid', lw=0.8, label=rel_names[r])
    axs[0, 1].legend(fontsize=8); axs[0, 1].set_xlabel('azimuth [deg]'); axs[0, 1].set_title('azimuth by relation of the two detectors')
    # (0,2) dx dy
    im = axs[0, 2].imshow(S('dxdy').T, origin='lower', extent=[-20, 20, -20, 20], cmap='viridis'); axs[0, 2].set_xlabel('Δx [cm]'); axs[0, 2].set_ylabel('Δy [cm]')
    axs[0, 2].set_title('first→second displacement (Δx, Δy)'); fig.colorbar(im, ax=axs[0, 2])
    # (1,0) dz and (1,1) lateral separation
    zc = 0.5 * (np.linspace(-10, 10, 401)[1:] + np.linspace(-10, 10, 401)[:-1])
    axs[1, 0].step(zc, S('dz'), where='mid'); axs[1, 0].set_xlabel('Δz = z2 − z1 [cm]'); axs[1, 0].set_title('vertical separation')
    dc = 0.5 * (np.linspace(-20, 20, 201)[1:] + np.linspace(-20, 20, 201)[:-1]); X, Y = np.meshgrid(dc, dc, indexing='ij')
    h, e = np.histogram(np.hypot(X, Y).ravel(), bins=np.arange(0, 25, 0.5), weights=S('dxdy').ravel())
    axs[1, 1].step(e[:-1], h, where='post'); axs[1, 1].set_xlabel('lateral separation [cm]'); axs[1, 1].set_title('lateral separation')
    # (1,2) relation fractions: data vs MC
    rc = S('rel_counts'); frac = rc / rc.sum()
    width = 0.8 / (1 + len(lams)); xs = np.arange(4)
    axs[1, 2].bar(xs, frac, width, label='data')
    for i, lam in enumerate(lams):
        _, fr = mc_azimuth(dmin, lam)
        axs[1, 2].bar(xs + (i + 1) * width, fr, width, label=f'geometry MC, λ = {lam:g} cm')
    axs[1, 2].set_xticks(xs + 0.4 - width / 2); axs[1, 2].set_xticklabels(rel_names, fontsize=8); axs[1, 2].set_ylabel('fraction of events'); axs[1, 2].legend(fontsize=8)
    axs[1, 2].set_title('detector relation of the hit pair')
    # MC azimuth overlay on the combined azimuth (panel 0,1 uses relations; add a dedicated axis inset)
    ins = axs[0, 1].inset_axes([0.0, 1.02, 1.0, 0.0001]); ins.axis('off')
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.step(az, tot / tot.sum(), where='mid', color='k', lw=0.8, label='data')
    for lam in lams:
        m, _ = mc_azimuth(dmin, lam); ax2.step(az, m, where='mid', lw=0.9, label=f'geometry MC, λ = {lam:g} cm')
    ax2.set_xlabel('azimuth of the lateral displacement [deg]'); ax2.set_ylabel('fraction / 0.5°'); ax2.legend(); ax2.set_title(f'azimuth, D > {dmin:g} cm: data vs a pure-geometry Monte Carlo')
    fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)
    fig2.tight_layout(); fig2.savefig(out.replace('.png', '_azimuth_vs_geometry_mc.png'), dpi=100); plt.close(fig2)
    print('events', ne, 'relation fractions (same, x, y, diag):', np.round(frac, 3))
    # azimuth summary: fraction within +-15 deg of the x axis (0/180) and of the y axis (+-90) and of the diagonals
    ax_x = (np.abs(az) < 15) | (np.abs(az) > 165); ax_y = np.abs(np.abs(az) - 90) < 15
    print('azimuth fractions: x axis %.3f, y axis %.3f, diagonals %.3f (each 15deg-wide window pair; uniform = 0.167, 0.167, 0.667)' %
          (tot[ax_x].sum() / tot.sum(), tot[ax_y].sum() / tot.sum(), 1 - tot[ax_x].sum() / tot.sum() - tot[ax_y].sum() / tot.sum()))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 8.0,
         tuple(float(v) for v in sys.argv[4:]) or (2.0, 4.0))
