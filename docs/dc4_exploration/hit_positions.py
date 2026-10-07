"""Histogram the x, y, z hit positions in week 1 of a DC4 .tra component (read from stdin).

stdin: one line per hit, 'TI idx x y z E' (written by run_hit_positions.sh via awk from the .tra file: only events with TI inside
week 1, hits from 'CH' lines; idx 0 = first hit, 1 = second hit of the Compton sequence).
Writes an .npz with, for all hits / first hits / second hits, the x, y, z histograms (cm) and the number of events, plus the 2-D (x, y)
histograms 'xy_<kind>_L<layer>' for each of the 4 detector layers (layer 0 = lowest z); 0.05 cm bins, array index [x, y].

With a second argument dmin (cm) only events whose first-to-second-hit distance is > dmin are kept; the file then also holds
geometry diagnostics of the first->second hit displacement of the kept events: 'az_lat<k>' (azimuth atan2(dy, dx), 0.5 deg bins from -180
to 180, for lateral distance bands k = 0..3 of [0,1), [1,3), [3,6), [6,inf) cm), 'dxdy' (0.2 cm bins, +-20 cm), 'dz' (0.05 cm bins, +-10 cm)
and 'dist' (0.1 cm bins, 0-30 cm). 'rel_counts' and 'az_rel' split the kept events by the relation of the detectors of hit 1 and hit 2
(detector id = (x > 0) + 2 * (y > 3.2); relations 0 same detector, 1 x-neighbour, 2 y-neighbour, 3 diagonal); 'layer_pair' is the 4 x 4 matrix of
(layer of hit 1, layer of hit 2).

Usage: <stream> | python hit_positions.py out.npz [dmin_cm]
"""
import sys
import numpy as np

EDGES = {'x': np.arange(-9., 9.0001, 0.05), 'y': np.arange(-8., 14.5001, 0.05), 'z': np.arange(12., 22.0001, 0.02)}
KINDS = ('all', 'first', 'second')
LAYER_Z = [12.0, 14.2, 16.9, 19.3, 22.0]   # layer boundaries in z [cm] (the gaps between the detector layers)
CHUNK = 64 * 2**20


def main(out, dmin=0.0):
    H = {(k, a): np.zeros(len(EDGES[a]) - 1) for k in KINDS for a in 'xyz'}
    nx, ny = len(EDGES['x']) - 1, len(EDGES['y']) - 1
    H2 = {(k, l): np.zeros(nx * ny) for k in KINDS for l in range(4)}
    az_edges = np.arange(-180., 180.001, 0.5); lat_bands = [0., 1., 3., 6., np.inf]
    AZ = np.zeros((4, len(az_edges) - 1)); DXDY = np.zeros((200, 200)); DZ = np.zeros(400); DIST = np.zeros(300)
    REL = np.zeros(4); AZREL = np.zeros((4, len(az_edges) - 1)); LPAIR = np.zeros((4, 4))
    n_events = n_hits = n_events_all = 0
    carry = None
    inp = sys.stdin.buffer
    rest = b''
    while True:
        buf = inp.read(CHUNK)
        data = rest + buf
        last = not buf
        if buf:
            cut = data.rfind(b'\n') + 1
            data, rest = data[:cut], data[cut:]
        else:
            rest = b''
        a = np.fromstring(data, dtype=float, sep=' ').reshape(-1, 6) if data else np.zeros((0, 6))
        if carry is not None and len(carry):
            a = np.vstack([carry, a]); carry = None
        if not last and len(a):                      # keep the possibly incomplete last event for the next chunk
            st = np.flatnonzero(a[:, 1] == 0)
            if len(st) == 0:
                carry = a; continue
            carry = a[st[-1]:]; a = a[:st[-1]]
        if len(a):
            st = np.flatnonzero(a[:, 1] == 0)
            ev = np.cumsum(a[:, 1] == 0) - 1
            p0 = a[st, 2:5]
            has2 = np.zeros(len(st), bool); has2[:] = (st + 1 < len(a))
            has2[has2] &= a[st[has2] + 1, 1] == 1
            p1 = np.zeros_like(p0); p1[has2] = a[st[has2] + 1, 2:5]
            d = np.linalg.norm(p1 - p0, axis=1)
            keep = has2 & (d > dmin)
            n_events_all += len(st); n_events += int(keep.sum())
            a = a[keep[ev]]
            n_hits += len(a)
            dv = (p1 - p0)[keep]; lat = np.hypot(dv[:, 0], dv[:, 1]); az = np.rad2deg(np.arctan2(dv[:, 1], dv[:, 0]))
            for k in range(4):
                m = (lat >= lat_bands[k]) & (lat < lat_bands[k + 1])
                AZ[k] += np.histogram(az[m], bins=az_edges)[0]
            det = (p0[:, 0] > 0).astype(int) + 2 * (p0[:, 1] > 3.2), (p1[:, 0] > 0).astype(int) + 2 * (p1[:, 1] > 3.2)
            rel = np.where(det[0] == det[1], 0, np.where((det[0] // 2) == (det[1] // 2), 1, np.where((det[0] % 2) == (det[1] % 2), 2, 3)))[keep]
            for r in range(4):
                REL[r] += (rel == r).sum()
                AZREL[r] += np.histogram(az[rel == r], bins=az_edges)[0]
            l0 = np.clip(np.digitize(p0[keep, 2], LAYER_Z) - 1, 0, 3); l1 = np.clip(np.digitize(p1[keep, 2], LAYER_Z) - 1, 0, 3)
            LPAIR += np.bincount(l0 * 4 + l1, minlength=16).reshape(4, 4)
            DXDY += np.histogram2d(dv[:, 0], dv[:, 1], bins=[np.linspace(-20, 20, 201)] * 2)[0]
            DZ += np.histogram(dv[:, 2], bins=np.linspace(-10, 10, 401))[0]
            DIST += np.histogram(d[keep], bins=np.linspace(0, 30, 301))[0]
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
        if last:
            break
    np.savez_compressed(out, n_events=n_events, n_hits=n_hits, n_events_all=n_events_all, dmin=dmin,
                        **{f'{k}_{a}': H[(k, a)] for k, a in H},
                        **{f'xy_{k}_L{l}': H2[(k, l)].reshape(nx, ny) for k, l in H2}, layer_z=np.array(LAYER_Z),
                        az_edges=az_edges, az_lat=AZ, rel_counts=REL, az_rel=AZREL, layer_pair=LPAIR, dxdy=DXDY, dz=DZ, dist=DIST,
                        **{f'edges_{a}': EDGES[a] for a in 'xyz'})
    print(out, 'events', n_events, 'of', n_events_all, 'hits', n_hits)


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
