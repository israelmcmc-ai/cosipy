"""Plot the x, y, z hit-position histograms (week 1, all background components combined) from hit_positions.py outputs.

Hit x and y positions sit on a strip lattice (pitch 0.1164 cm, 64 strips per detector), so a regular histogram beats against the lattice.
x and y are therefore shown as counts per strip (each strip falls in one 0.05 cm histogram bin because the bins are narrower than the pitch;
the small fraction of hits lying between strips is not shown, and is printed). z is shown in 0.02 cm bins. Rows: all hits / first hit / second hit; columns: x, y, z [cm] (coordinates of the .tra files).
Usage: python plot_hit_positions.py hits_dir out.png
"""
import sys, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def per_strip(edges, h, frac=0.25):
    """Counts of each strip: the strips are the narrow peaks (bins above frac * the max of their block; each strip falls in one 0.05 cm
    bin since the bins are narrower than the pitch). Returns strip positions, counts, the fraction of the hits that lie
    between strips (shared / interpolated positions) and the bin indices of the strips."""
    c = 0.5 * (edges[1:] + edges[:-1])
    nz = np.flatnonzero(h > 0)
    blocks = np.split(nz, np.flatnonzero(np.diff(nz) > 6) + 1)           # blocks separated by gaps > 6 empty bins
    peaks = np.concatenate([b[h[b] > frac * h[b].max()] for b in blocks])
    return c[peaks], h[peaks], 1 - h[peaks].sum() / h.sum(), peaks


def split_blocks(pos, cnt, gap=0.5):
    """Insert NaN where the strip positions jump (> gap cm) so that the line is not drawn across the gap between detectors."""
    j = np.flatnonzero(np.diff(pos) > gap) + 1
    return np.insert(pos, j, np.nan), np.insert(cnt, j, np.nan)


if __name__ == '__main__':
    hits_dir, out = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(hits_dir, '*.npz')))
    Z = [np.load(f) for f in files]
    names = [os.path.basename(f)[:-4] for f in files]
    ne = sum(int(z['n_events']) for z in Z); nh = sum(int(z['n_hits']) for z in Z)
    kinds = (('all', 'all hits'), ('first', 'first hit'), ('second', 'second hit'))
    fig, axs = plt.subplots(3, 3, figsize=(16, 10))
    for r, (k, title) in enumerate(kinds):
        for c, a in enumerate('xyz'):
            e = Z[0][f'edges_{a}']
            h = sum(z[f'{k}_{a}'] for z in Z)
            ax = axs[r, c]
            if a in 'xy':
                pos, cnt, between, _ = per_strip(e, h)
                ax.plot(*split_blocks(pos, cnt), lw=0.9, marker='.', ms=2)
                ax.set_ylabel(f'{title}: counts / strip')
                if r == 0:
                    print(a, 'strips:', len(pos), '; hits between strips: %.1f%%' % (100 * between))
            else:
                ax.step(e[:-1], h, where='post', lw=0.8)
                ax.set_ylabel(f'{title}: counts / {e[1] - e[0]:g} cm')
            ax.set_xlim(e[0], e[-1]); ax.set_ylim(bottom=0)
            ax.set_xlabel(f'{a} [cm]')
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.suptitle(f'Hit positions, week 1, all background components combined: {ne / 1e6:.1f}M events, {nh / 1e6:.1f}M hits\n(' + ', '.join(names) + ')', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(len(files), 'components,', ne, 'events,', nh, 'hits')
