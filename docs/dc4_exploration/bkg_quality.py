"""Background event-level histograms for the quality-cut study (reads tra_events.awk output on stdin).

For each (measured-energy bin, LA bin, number of hits, depth category of hit 1, depth category of hit 2) it stores the number of events
'N' and the sum of sin(phi) 'SINPHI' (phi = Compton angle). The sum of sin(phi) gives the expected number of events inside a |ARM| < w window
for a source direction uniformly distributed over the sphere: B(w) = sin(w) * SINPHI (see sensitivity_quality.py).
Bins: E tra_quality.E_EDGES (6); LA 0..30 cm in 0.25 cm (120, the last holds all larger values); SQ 2,3,4,5,6,7+ (6); categories 12 x 12.
Usage: <stream> | python bkg_quality.py out.npz
"""
import sys
import numpy as np
from tra_quality import derive, E_EDGES

NL, NS, NC = 120, 6, 12
CHUNK = 64 * 2**20
N = np.zeros((6, NL, NS, NC, NC)); SIN = np.zeros_like(N)
n_in = 0
rest = b''
inp = sys.stdin.buffer
while True:
    buf = inp.read(CHUNK)
    data = rest + buf
    if buf:
        cut = data.rfind(b'\n') + 1
        data, rest = data[:cut], data[cut:]
    if data:
        a = np.fromstring(data, dtype=float, sep=' ').reshape(-1, 15)
        n_in += len(a)
        d = derive(a)
        iE = np.digitize(d['E'], E_EDGES) - 1
        iL = np.clip((d['LA'] / 0.25).astype(int), 0, NL - 1)
        iS = np.clip(d['sq'].astype(int), 2, 7) - 2
        flat = (((iE * NL + iL) * NS + iS) * NC + d['cat1']) * NC + d['cat2']
        N += np.bincount(flat, minlength=N.size).reshape(N.shape)
        SIN += np.bincount(flat, weights=np.sin(d['phi'].astype(float)), minlength=N.size).reshape(N.shape)
    if not buf:
        break
np.savez_compressed(sys.argv[1], N=N, SINPHI=SIN, n_in=n_in, e_edges=E_EDGES)
print(sys.argv[1], 'compton events read', n_in, 'kept', int(N.sum()))
