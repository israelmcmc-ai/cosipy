"""Background counts for the psichi-consistency study: (measured-energy bin, phi [2.5 deg, 72], psichi polar angle psi from SC +z [2.5 deg, 72],
Distance bin [10], first-hit layer [4], SQ > 5 flag [2], hit-2-below-hit-1 flag [2]). psichi direction p = -(scattered direction) (the source lies at angle phi from p),
psi = arccos(-(z2 - z1)/|hit2 - hit1|). Reads tra_events.awk output on stdin. Usage: <stream> | python bkg_psichi.py out.npz"""
import sys
import numpy as np
from tra_quality import derive, E_EDGES

NP, DEDGE = 72, np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 20])
N = np.zeros((6, NP, NP, 10, 4, 2, 2)); n_in = 0; rest = b''; inp = sys.stdin.buffer
while True:
    buf = inp.read(64 * 2**20); data = rest + buf
    if buf:
        cut = data.rfind(b'\n') + 1; data, rest = data[:cut], data[cut:]
    if data:
        a = np.fromstring(data, dtype=float, sep=' ').reshape(-1, 15); n_in += len(a); d = derive(a)
        iE = np.digitize(d['E'], E_EDGES) - 1; iP = np.clip((np.rad2deg(d['phi'].astype(float)) / 2.5).astype(int), 0, NP - 1)
        psi = np.rad2deg(np.arccos(np.clip(-(d['z2'].astype(float) - d['z1']) / d['d01'], -1, 1))); iQ = np.clip((psi / 2.5).astype(int), 0, NP - 1)
        iD = np.digitize(d['d01'], DEDGE); iL = d['cat1'] // 3; iS = (d['sq'] > 5).astype(int); dr = (d['z2'] < d['z1']).astype(int)
        flat = (((((iE * NP + iP) * NP + iQ) * 10 + iD) * 4 + iL) * 2 + iS) * 2 + dr
        N += np.bincount(flat, minlength=N.size).reshape(N.shape)
    if not buf: break
np.savez_compressed(sys.argv[1], N=N, n_in=n_in)
print(sys.argv[1], 'compton events read', n_in, 'kept', int(N.sum()))
