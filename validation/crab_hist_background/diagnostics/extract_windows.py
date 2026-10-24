"""Stream a gzipped DC3/DC4 FITS event file from Wasabi and keep the rows in [tmin, tmax) (unix s)."""
import sys, gzip, numpy as np, boto3
key, outdir = sys.argv[1], sys.argv[2]
wins = [(float(a), float(a) + 3 * 3600) for a in sys.argv[3:]]
import inspect
from cosipy.util.data_fetching import fetch_wasabi_file
_wasabi = {k: v.default for k, v in inspect.signature(fetch_wasabi_file).parameters.items()}
s3 = boto3.client('s3', endpoint_url=_wasabi['endpoint'],
                  aws_access_key_id=_wasabi['access_key'], aws_secret_access_key=_wasabi['secret_key'])
import time as _t
class Resumable:
    def __init__(self): self.pos=0; self._open()
    def _open(self):
        self.body = s3.get_object(Bucket=_wasabi['bucket'], Key=key, Range=f'bytes={self.pos}-')['Body']
    def read(self, n=-1):
        for attempt in range(20):
            try:
                b = self.body.read(n); self.pos += len(b); return b
            except Exception as e:
                print("retry", attempt, self.pos, e, flush=True); _t.sleep(2*(attempt+1)); self._open()
        raise RuntimeError("too many retries")
f = gzip.GzipFile(fileobj=Resumable())
def read_exact(n):
    buf = bytearray()
    while len(buf) < n:
        b = f.read(n - len(buf))
        if not b: break
        buf += b
    return bytes(buf)
def header():
    cards = ''
    while True:
        blk = read_exact(2880).decode('latin1'); cards += blk
        if any(blk[i:i+80].rstrip() == 'END' for i in range(0, 2880, 80)): return cards
header(); h = header()
cards = {h[i:i+8].strip(): h[i+10:i+80].split('/')[0].strip().strip("'").strip() for i in range(0, len(h), 80)}
ncol = int(cards['TFIELDS']); names = [cards[f'TTYPE{i}'] for i in range(1, ncol+1)]; forms = [cards[f'TFORM{i}'] for i in range(1, ncol+1)]
print(names, forms, flush=True)
dt = np.dtype([(n, '>f8', (int(fm[:-1]),) if fm[:-1] not in ('', '1') else ()) for n, fm in zip(names, forms)])
assert dt.itemsize == int(cards['NAXIS1']), (dt.itemsize, cards['NAXIS1'])
n = int(cards['NAXIS2']); done = 0; keep = [[] for _ in wins]
while done < n:
    k = min(1_000_000, n - done); raw = read_exact(k * dt.itemsize); got = len(raw) // dt.itemsize
    if got == 0: break
    c = np.frombuffer(raw[:got*dt.itemsize], dtype=dt); t = c['TimeTags']
    for i, (a, b) in enumerate(wins):
        keep[i].append(c[(t >= a) & (t < b)].copy())
    done += got
    if done % 20_000_000 < 1_000_000: print(done, n, flush=True)
for (a, b), k in zip(wins, keep):
    rows = np.concatenate(k)
    np.savez(f'{outdir}/bkg_{int(a)}.npz', **{nm: rows[nm] for nm in ['TimeTags', 'Energies', 'Phi', 'Psi local', 'Chi local', 'Distance']})
    print("kept", int(a), len(rows), "of", done, flush=True)
