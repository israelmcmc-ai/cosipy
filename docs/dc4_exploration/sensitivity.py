"""Signal / sqrt(background) vs Distance and ARM cuts (percentage changes only).

Signal = Crab dataset, background = DC4 total background; both are streamed from
gzipped FITS tables (no decompressed copy on disk) and reduced to a histogram over
(energy bin, Distance, ARM) that is saved to an .npz file.

ARM = (angle between the Crab and the scattered-photon direction) - phi, in deg,
using the Galactic Chi/Psi columns.
Background events are kept only at times when the Crab is above the Earth limb
(zenith angle < ZENITH_MAX), matching the Crab simulation, which blocks photons
beyond ~113 deg.

Usage: python sensitivity.py events.fits[.gz] orientation.fits out.npz [max_rows]
       ('-' as events file reads the gzipped FITS from stdin)
"""
import sys
import gzip
import numpy as np
from astropy.io import fits

CRAB_LB = np.deg2rad([184.5575, -5.7843])           # Galactic (l, b) of the Crab
E_EDGES = np.geomspace(200., 5000., 7)              # keV, 6 log bins
D_EDGES = np.r_[np.arange(0, 10.01, 0.25), 12, 14, 16, 20, 30]   # cm
A_EDGES = np.r_[-180., np.arange(-30, 30.01, 0.25), 180.]        # deg
ZENITH_MAX = 113.0
ORI_DT = 15.0
CHUNK_ROWS = 1_000_000
DT = np.dtype([('E', '>f8'), ('t', '>f8'), ('X', '>f8', 2), ('Y', '>f8', 2), ('Z', '>f8', 2),
               ('phi', '>f8'), ('chi', '>f8'), ('psi', '>f8'), ('dist', '>f8'),
               ('l', '>f8'), ('b', '>f8')])


def vec(lon, lat):
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


CRAB = vec(*CRAB_LB)


def read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        b = f.read(n - len(buf))
        if not b:
            break
        buf += b
    return bytes(buf)


def stream_rows(f, max_rows=None):
    """Yield structured-array chunks from a (decompressed) FITS binary-table stream."""
    def header():
        cards = b''
        while True:
            blk = read_exact(f, 2880)
            cards += blk
            if any(blk[i:i + 80].startswith(b'END ') or blk[i:i + 80].rstrip() == b'END' for i in range(0, 2880, 80)):
                return cards.decode('latin1')
    header()                                    # primary HDU
    h = header()                                # table HDU
    get = lambda k: int(next(h[i + 10:i + 80].split('/')[0] for i in range(0, len(h), 80) if h[i:i + 8].strip() == k))
    assert get('NAXIS1') == DT.itemsize, get('NAXIS1')
    n = get('NAXIS2')
    if max_rows:
        n = min(n, max_rows)
    done = 0
    while done < n:
        k = min(CHUNK_ROWS, n - done)
        raw = read_exact(f, k * DT.itemsize)
        got = len(raw) // DT.itemsize
        if got == 0:
            break
        yield np.frombuffer(raw[:got * DT.itemsize], dtype=DT)
        done += got


def arm_deg(c):
    v = vec(np.deg2rad(c['l']), np.deg2rad(c['b']))
    geo = np.rad2deg(np.arccos(np.clip(v @ CRAB, -1, 1)))
    return geo - np.rad2deg(c['phi'])


def crab_zenith(t, ori_t, ori_zen):
    f = (t - ori_t[0]) / ORI_DT
    i = np.clip(np.floor(f).astype(int), 0, len(ori_t) - 2)
    w = (f - i)[:, None]
    z = (1 - w) * ori_zen[i] + w * ori_zen[i + 1]
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(z @ CRAB, -1, 1)))


def main(evfile, orifile, out, max_rows=None):
    ori = fits.open(orifile)[1].data
    ori_t = np.asarray(ori['TimeStamp'])
    ori_zen = vec(*np.deg2rad(np.asarray(ori['EarthZenith']).T))
    H = np.zeros((len(E_EDGES) - 1, len(D_EDGES) - 1, len(A_EDGES) - 1))
    Hall = H.copy()                                   # without the Crab-visibility selection
    zen_hist = np.zeros(361)                          # Crab zenith angle of all events (diagnostic)
    f = sys.stdin.buffer if evfile == '-' else open(evfile, 'rb')
    f = gzip.GzipFile(fileobj=f) if (evfile == '-' or evfile.endswith('.gz')) else f
    nread = 0
    for c in stream_rows(f, max_rows):
        zen = crab_zenith(c['t'], ori_t, ori_zen)
        zen_hist += np.bincount(np.clip(zen.astype(int), 0, 360), minlength=361)
        arm = arm_deg(c)
        idx = (np.digitize(c['E'], E_EDGES) - 1, np.digitize(c['dist'], D_EDGES) - 1, np.digitize(arm, A_EDGES) - 1)
        ok = (idx[0] >= 0) & (idx[0] < H.shape[0]) & (idx[1] >= 0) & (idx[1] < H.shape[1])
        flat = (idx[0] * H.shape[1] + idx[1]) * H.shape[2] + idx[2]
        for tgt, sel in ((Hall, ok), (H, ok & (zen < ZENITH_MAX))):
            tgt += np.bincount(flat[sel], minlength=H.size).reshape(H.shape)
        nread += len(c)
        print(nread, flush=True)
    np.savez_compressed(out, H=H, Hall=Hall, e_edges=E_EDGES, d_edges=D_EDGES, a_edges=A_EDGES,
                        zen_hist=zen_hist, nread=nread)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None)
