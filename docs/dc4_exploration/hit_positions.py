"""Histogram the x, y, z hit positions in week 1 of a DC4 .tra component (read from stdin).

stdin: one line per hit, 'TI idx x y z E' (written by run_hit_positions.sh via awk from the .tra file: only events with TI inside
week 1, hits from 'CH' lines; idx 0 = first hit, 1 = second hit of the Compton sequence).
Writes an .npz with, for all hits / first hits / second hits, the x, y, z histograms (cm) and the number of events, plus the 2-D (x, y)
histograms 'xy_<kind>_L<layer>' for each of the 4 detector layers (layer 0 = lowest z); 0.05 cm bins, array index [x, y].

Usage: <stream> | python hit_positions.py out.npz
"""
import sys
import numpy as np

EDGES = {'x': np.arange(-9., 9.0001, 0.05), 'y': np.arange(-8., 14.5001, 0.05), 'z': np.arange(12., 22.0001, 0.02)}
KINDS = ('all', 'first', 'second')
LAYER_Z = [12.0, 14.2, 16.9, 19.3, 22.0]   # layer boundaries in z [cm] (the gaps between the detector layers)
CHUNK = 64 * 2**20


def main(out):
    H = {(k, a): np.zeros(len(EDGES[a]) - 1) for k in KINDS for a in 'xyz'}
    nx, ny = len(EDGES['x']) - 1, len(EDGES['y']) - 1
    H2 = {(k, l): np.zeros(nx * ny) for k in KINDS for l in range(4)}
    n_events = n_hits = 0
    rest = b''
    inp = sys.stdin.buffer
    while True:
        buf = inp.read(CHUNK)
        if not buf and not rest:
            break
        data = rest + buf
        if buf:
            cut = data.rfind(b'\n') + 1
            data, rest = data[:cut], data[cut:]
        else:
            rest = b''
        if not data:
            continue
        a = np.fromstring(data, dtype=float, sep=' ').reshape(-1, 6)
        n_hits += len(a); n_events += int((a[:, 1] == 0).sum())
        for k, sel in (('all', slice(None)), ('first', a[:, 1] == 0), ('second', a[:, 1] == 1)):
            for i, ax in ((2, 'x'), (3, 'y'), (4, 'z')):
                H[(k, ax)] += np.histogram(a[sel, i], bins=EDGES[ax])[0]
            b = a[sel]
            ix = np.floor((b[:, 2] - EDGES['x'][0]) / 0.05).astype(int); iy = np.floor((b[:, 3] - EDGES['y'][0]) / 0.05).astype(int)
            il = np.digitize(b[:, 4], LAYER_Z) - 1
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (il >= 0) & (il < 4)
            for l in range(4):
                s2 = ok & (il == l)
                H2[(k, l)] += np.bincount(ix[s2] * ny + iy[s2], minlength=nx * ny)
        if not buf:
            break
    np.savez_compressed(out, n_events=n_events, n_hits=n_hits, **{f'{k}_{a}': H[(k, a)] for k, a in H},
                        **{f'xy_{k}_L{l}': H2[(k, l)].reshape(nx, ny) for k, l in H2}, layer_z=np.array(LAYER_Z),
                        **{f'edges_{a}': EDGES[a] for a in 'xyz'})
    print(out, 'events', n_events, 'hits', n_hits)


if __name__ == '__main__':
    main(sys.argv[1])
