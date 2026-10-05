"""Spacecraft-frame psichi maps split by survey mode and roll angle (10 deg bins).

Roll angle = angle, counterclockwise about the spacecraft +z axis, from +x to the projection of
the Earth zenith onto the xy plane, i.e. the spacecraft azimuth chi of the Earth zenith
(zenith = 'EarthZenith' of the orientation file interpolated to the event time; x, y from the
event's own pointings). Survey mode as in bin_dc4.py (slews excluded).

Streams an event file and saves, for each (mode, roll bin, energy slice, phi slice), the SC-frame
psichi HEALPix map; slices are the merged bins plot_dc4.E_SLICES x plot_dc4.PHI_SLICES.

Usage: python roll_modes.py events.fits[.gz]|- orientation.fits out.npz [nside] [min_distance_cm]
"""
import sys
import gzip
import numpy as np
import healpy as hp
from astropy.io import fits
import bin_dc4 as B
import plot_dc4 as P
from sensitivity import stream_rows

ROLL_EDGES = np.arange(-180., 180.01, 10.)


def main(evfile, orifile, out, nside=32, min_dist=0.0):
    npix = hp.nside2npix(nside)
    ori = fits.open(orifile)[1].data
    ori_t = np.asarray(ori['TimeStamp'])
    ori_zen = B.vec(*np.deg2rad(np.asarray(ori['EarthZenith']).T))
    ori_z = B.vec(*np.deg2rad(np.asarray(ori['ZPointings']).T))
    ori_mode, zdir = B.survey_modes(ori_t, ori_zen, ori_z)

    e_slice = np.full(len(B.E_EDGES) - 1, -1); p_slice = np.full(len(B.PHI_EDGES) - 1, -1)
    for k, (a, b) in enumerate(P.E_SLICES): e_slice[a:b] = k
    for k, (a, b) in enumerate(P.PHI_SLICES): p_slice[a:b] = k
    nS = len(P.E_SLICES) * len(P.PHI_SLICES)
    H = np.zeros((2, len(ROLL_EDGES) - 1, nS, npix), dtype=np.int64)
    roll_all = np.zeros((3, len(ROLL_EDGES) - 1), dtype=np.int64)      # rows: north, south, slew (all selected events)
    raw = sys.stdin.buffer if evfile == '-' else open(evfile, 'rb')
    stream = gzip.GzipFile(fileobj=raw) if (evfile == '-' or evfile.endswith('.gz')) else raw
    n = 0
    for c in stream_rows(stream):
        t = c['t'].astype(float); n += len(c)
        ie = np.digitize(c['E'].astype(float), B.E_EDGES) - 1
        ip = np.digitize(c['phi'].astype(float), B.PHI_EDGES) - 1
        ok = (ie >= 0) & (ie < len(e_slice)) & (ip >= 0) & (ip < len(p_slice)) & (c['dist'] >= min_dist)
        zen = B.zenith_at(t, ori_t, ori_zen)
        X = B.vec(c['X'][:, 0], c['X'][:, 1]); Y = B.vec(c['Y'][:, 0], c['Y'][:, 1])
        roll = np.rad2deg(np.arctan2((zen * Y).sum(1), (zen * X).sum(1)))
        ir = np.clip(np.digitize(roll, ROLL_EDGES) - 1, 0, len(ROLL_EDGES) - 2)
        oi = np.clip(np.floor((t - ori_t[0]) / B.ORI_DT).astype(int), 0, len(ori_t) - 1)
        mode = ori_mode[oi]
        pix = hp.ang2pix(nside, np.clip(c['psi'].astype(float), 0, np.pi), c['chi'].astype(float))
        for m, row in ((1, 0), (-1, 1), (0, 2)):
            sel = ok & (mode == m)
            roll_all[row] += np.bincount(ir[sel], minlength=len(ROLL_EDGES) - 1)
            if m == 0:
                continue
            s = sel & (e_slice[np.clip(ie, 0, None)] >= 0)
            flat = ((ir[s] * nS + e_slice[ie[s]] * len(P.PHI_SLICES) + p_slice[ip[s]]) * npix + pix[s])
            H[row] += np.bincount(flat, minlength=H[row].size).reshape(H[row].shape)
        print(n, flush=True)
    np.savez_compressed(out, H=H, roll_all=roll_all, roll_edges=ROLL_EDGES, nside=nside, min_dist=min_dist,
                        e_edges=B.E_EDGES, phi_edges=B.PHI_EDGES, zdir_north=zdir['north'], zdir_south=zdir['south'])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 32,
         float(sys.argv[5]) if len(sys.argv) > 5 else 0.0)
