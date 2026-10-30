"""Signal events (isotropic far-field simulations with true direction) -> per-event arrays with the true-direction ARM.

Reads the output of tra_events.awk on stdin, writes an .npz with the arrays of tra_quality.derive().
Usage: <stream> | python resp_signal.py out.npz
"""
import sys
import numpy as np
from tra_quality import derive

CHUNK = 64 * 2**20
parts = []
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
        parts.append(derive(a))
    if not buf:
        break
out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
np.savez_compressed(sys.argv[1], **out)
print(sys.argv[1], len(out['E']), 'events')
