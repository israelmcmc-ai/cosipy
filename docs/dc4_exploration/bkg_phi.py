"""Background event counts binned in (measured-energy bin, scatter angle phi [2.5 deg, 72 bins], Distance bin [10], number of hits [6], first-hit layer [4])
for the scatter-angle study (reads tra_events.awk output on stdin). Distance = |hit2 - hit1| (the FITS Distance).
Usage: <stream> | python bkg_phi.py out.npz"""
import sys
import numpy as np
from tra_quality import derive, E_EDGES

NP, DEDGE = 72, np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 20])
N = np.zeros((6, NP, 10, 6, 4)); n_in = 0; rest = b''; inp = sys.stdin.buffer
while True:
    buf = inp.read(64 * 2**20); data = rest + buf
    if buf:
        cut = data.rfind(b'\n') + 1; data, rest = data[:cut], data[cut:]
    if data:
        a = np.fromstring(data, dtype=float, sep=' ').reshape(-1, 15); n_in += len(a); d = derive(a)
        iE = np.digitize(d['E'], E_EDGES) - 1; iP = np.clip((np.rad2deg(d['phi'].astype(float)) / 2.5).astype(int), 0, NP - 1)
        iD = np.digitize(d['d01'], DEDGE); iS = np.clip(d['sq'].astype(int), 2, 7) - 2; iL = d['cat1'] // 3
        flat = (((iE * NP + iP) * 10 + iD) * 6 + iS) * 4 + iL
        N += np.bincount(flat, minlength=N.size).reshape(N.shape)
    if not buf: break
np.savez_compressed(sys.argv[1], N=N, n_in=n_in)
print(sys.argv[1], 'compton events read', n_in, 'kept', int(N.sum()))
