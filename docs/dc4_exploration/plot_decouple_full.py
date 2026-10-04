"""Earth-minus-spacecraft decoupling using the full (chi-dependent) spacecraft map.

For each survey mode, the spacecraft-frame psichi map S of that mode's events is
treated as a spacecraft-fixed template. Its expected appearance in the Earth
(alt-az) frame is the exposure-weighted sum, over the 15 s orientation samples k
of the mode, of S rotated by the attitude of sample k:

    model(p) = sum_k n_k S(R_k^-1 p) / sum_k n_k

with n_k the number of selected events in sample k. Subtracting it from the
Earth-frame map leaves structure that is not explained by a fixed spacecraft
pattern. (S itself still contains the Earth signal smeared by the roll, so this
is approximate.)

Usage: python plot_decouple_full.py hist.npz orientation.fits outdir [smooth_fwhm_deg]
"""
import sys
import os
import numpy as np
import healpy as hp
from astropy.io import fits
import plot_dc4 as P
import plot_decouple as D
from bin_dc4 import vec, POLE, survey_modes


def rotation_models(z, ori, name, nside, templates=None):
    """Earth-frame model maps for every (E, phi) slice of P.E_SLICES x P.PHI_SLICES."""
    ori_t = np.asarray(ori['TimeStamp'])
    zen = vec(*np.deg2rad(np.asarray(ori['EarthZenith']).T))
    X = vec(*np.deg2rad(np.asarray(ori['XPointings']).T))
    Z = vec(*np.deg2rad(np.asarray(ori['ZPointings']).T))
    Y = np.cross(Z, X)
    mode, _ = survey_modes(ori_t, zen, Z)
    w_all = np.asarray(z['ori_counts'], dtype=float)
    ks = np.where((mode == (1 if name == 'north' else -1)) & (w_all > 0))[0]
    north = POLE[None, :] - (zen @ POLE)[:, None] * zen
    north /= np.linalg.norm(north, axis=1, keepdims=True)
    east = np.cross(north, zen)
    basis = np.stack([north, east, zen], axis=1)               # (K, 3 horizon axes, 3 gal comps)
    H = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T   # (npix, 3) = (N, E, U)

    slices = [(e0, e1, p0, p1) for (e0, e1) in P.E_SLICES for (p0, p1) in P.PHI_SLICES]
    # spacecraft-frame template per slice: the mode's own SC map, or a user-supplied {slice: map}
    S = templates or {s: z[f'spacecraft_{name}'][s[0]:s[1], s[2]:s[3]].sum(axis=(0, 1)).astype(np.float32) for s in slices}
    model = {s: np.zeros(len(H)) for s in slices}
    for c in range(0, len(ks), 400):
        k = ks[c:c + 400]
        # SC axes expressed in the horizon basis -> SC-frame components of each Earth pixel direction
        comp = []
        for A in (X[k], Y[k], Z[k]):
            Ah = np.einsum('kha,ka->kh', basis[k], A)           # (K, 3) horizon components
            comp.append(Ah @ H.T)                               # (K, npix)
        pix = hp.vec2pix(nside, *comp)
        w = w_all[k][:, None]
        for s in slices:
            model[s] += (w * S[s][pix]).sum(axis=0)
    wsum = w_all[ks].sum()
    return {s: m / wsum for s, m in model.items()}


def main(npz, orifile, outdir, fwhm=4.0):
    os.makedirs(outdir, exist_ok=True)
    z = np.load(npz)
    nside = int(z['nside'])
    ori = fits.open(orifile)[1].data
    md = float(z['min_dist']) if 'min_dist' in z else 0.0
    cut = f' [Distance ≥ {md:g} cm]' if md > 0 else ''
    models = {}
    for name in ('north', 'south'):
        models[name] = rotation_models(z, ori, name, nside)
        print(name, 'model done', flush=True)
        D.decouple_maps(z[f'earth_{name}'], z[f'spacecraft_{name}'], tuple(z[f'zdir_{name}']),
                        z['e_edges'], z['phi_edges'], outdir, fwhm, name, cut,
                        model_fn=lambda sc, s, m=models[name]: m[s],
                        what='rotated full spacecraft map', fname='psichi_maps_earth_minus_fullsc_survey_')
    D.example(z, fwhm, outdir, cut, models=models, what='full SC map, rotated to Earth frame',
              fname='decoupling_full_example.png')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 4.0)
