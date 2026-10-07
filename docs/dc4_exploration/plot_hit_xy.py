"""x-y hit-position density for each of the 4 detector layers (week 1, all background components combined).

Uses the 2-D histograms of hit_positions.py. Hits sit on the strip lattice (0.1164 cm pitch, 64 strips per detector, 2 detectors per axis),
so the density is shown per strip pair: strips are the narrow peaks of the x and y marginals, each falls in one 0.05 cm bin, and the
(x strip, y strip) cell content is read directly from the 2-D histogram (the ~10% of hits between strips are not shown).
Usage: python plot_hit_xy.py hits_dir outprefix   -> outprefix_all_hits.png, outprefix_first_hit.png, outprefix_second_hit.png
"""
import sys, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_hit_positions import per_strip

PITCH = 0.1164


def blocks_of(pos):
    j = np.flatnonzero(np.diff(pos) > 0.5) + 1
    return np.split(np.arange(len(pos)), j)


def main(hits_dir, prefix):
    Z = [np.load(f) for f in sorted(glob.glob(os.path.join(hits_dir, '*.npz')))]
    ex, ey = Z[0]['edges_x'], Z[0]['edges_y']
    px = per_strip(ex, sum(z['all_x'] for z in Z))[3]; py = per_strip(ey, sum(z['all_y'] for z in Z))[3]
    cx = 0.5 * (ex[1:] + ex[:-1])[px]; cy = 0.5 * (ey[1:] + ey[:-1])[py]
    bx, by = blocks_of(cx), blocks_of(cy)
    zl = Z[0]['layer_z']
    for kind, title in (('all', 'all hits'), ('first', 'first hit'), ('second', 'second hit')):
        fig, axs = plt.subplots(2, 2, figsize=(12, 11.5))
        tot_kind = sum(float(sum(z[f'xy_{kind}_L{l}'].sum() for l in range(4))) for z in Z)
        for l, ax in zip((3, 2, 1, 0), axs.flat):                  # top layer first
            H = sum(z[f'xy_{kind}_L{l}'] for z in Z)               # [x bin, y bin]
            D = H[np.ix_(px, py)]                                  # [x strip, y strip]
            vmax = np.percentile(D[D > 0], 99.5)
            for ib in bx:
                for jb in by:
                    xe = np.linspace(cx[ib][0] - PITCH / 2, cx[ib][-1] + PITCH / 2, len(ib) + 1)
                    ye = np.linspace(cy[jb][0] - PITCH / 2, cy[jb][-1] + PITCH / 2, len(jb) + 1)
                    im = ax.pcolormesh(xe, ye, D[np.ix_(ib, jb)].T, cmap='viridis', vmin=0, vmax=vmax, rasterized=True)
            ax.set_aspect('equal'); ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]')
            ax.set_title(f'layer {l + 1} (z = {zl[l]:.1f}–{zl[l + 1]:.1f} cm): {H.sum() / 1e6:.2f}M {title}, {100 * H.sum() / tot_kind:.0f}% of the total', fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.8, label=f'{title} per strip pair')
        fig.suptitle(f'x–y position density of the {title} in each detector layer (week 1, all background components)', y=0.995)
        fig.tight_layout()
        fig.savefig(f'{prefix}_{kind}_hits.png' if kind == 'all' else f'{prefix}_{kind}_hit.png', dpi=100)
        plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
