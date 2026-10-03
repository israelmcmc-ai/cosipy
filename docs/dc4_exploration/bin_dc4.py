"""Bin a DC4 mock-dataset FITS file in (energy, phi, psichi) for three frames.

Frames for the psichi direction (direction of the scattered gamma ray):
  galactic   : (l, b)             -- the 'Chi galactic'/'Psi galactic' columns
  spacecraft : (chi, 90deg-psi)   -- the 'Chi local'/'Psi local' columns
  earth      : (azimuth, altitude) of the local horizon at the satellite;
               zenith = 'EarthZenith' of the orientation file (interpolated to
               the event time), North = projection of the celestial pole
               (the Earth's spin axis) onto the horizon plane, azimuth
               increases from North towards East.

The result is a dense histogram per frame with axes (energy, phi, HEALPix
pixel), saved to an .npz file.

Usage: python bin_dc4.py events.fits.gz orientation.fits out.npz [nside]
"""
import sys
import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

E_EDGES = np.geomspace(100., 10000., 11)        # keV, 10 log bins
PHI_EDGES = np.deg2rad(np.arange(0., 180.001, 5.))  # 36 bins
CHUNK = 2_000_000
ORI_DT = 15.0


def vec(lon, lat):
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def unvec(v):
    return np.arctan2(v[:, 1], v[:, 0]), np.arcsin(np.clip(v[:, 2], -1, 1))


# Celestial (equatorial) north pole in Galactic coordinates
_pole = SkyCoord(ra=0 * u.deg, dec=90 * u.deg, frame='icrs').galactic
POLE = vec(_pole.l.rad, _pole.b.rad)


def zenith_at(t, ori_t, ori_zen):
    """Linearly interpolate the Earth-zenith unit vectors to times t."""
    f = (t - ori_t[0]) / ORI_DT
    i = np.clip(np.floor(f).astype(int), 0, len(ori_t) - 2)
    w = (f - i)[:, None]
    z = (1 - w) * ori_zen[i] + w * ori_zen[i + 1]
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def main(evfile, orifile, out, nside=32):
    npix = hp.nside2npix(nside)
    ori = fits.open(orifile)[1].data
    ori_t = np.asarray(ori['TimeStamp'])
    ori_zen = vec(*np.deg2rad(np.asarray(ori['EarthZenith']).T))
    assert np.allclose(np.diff(ori_t), ORI_DT)

    d = fits.open(evfile, memmap=False)[1].data
    n = len(d)
    shape = (len(E_EDGES) - 1, len(PHI_EDGES) - 1, npix)
    hists = {k: np.zeros(shape, dtype=np.int64)
             for k in ('galactic', 'spacecraft', 'earth')}
    lo = lambda a: np.asarray(a)

    for s in range(0, n, CHUNK):
        sl = slice(s, min(s + CHUNK, n))
        E = lo(d['Energies'][sl]); phi = lo(d['Phi'][sl]); t = lo(d['TimeTags'][sl])
        ie = np.digitize(E, E_EDGES) - 1
        ip = np.digitize(phi, PHI_EDGES) - 1
        ok = (ie >= 0) & (ie < shape[0]) & (ip >= 0) & (ip < shape[1])

        chi = lo(d['Chi local'][sl]); psi = lo(d['Psi local'][sl])
        l = np.deg2rad(lo(d['Chi galactic'][sl])); b = np.deg2rad(lo(d['Psi galactic'][sl]))

        # spacecraft: lon = chi, lat = 90deg - psi
        pix_sc = hp.ang2pix(nside, np.clip(psi, 0, np.pi), chi)  # theta = psi (colatitude), phi = chi
        # galactic
        pix_gal = hp.ang2pix(nside, np.pi / 2 - b, l)

        # earth horizon frame
        v = vec(l, b)
        zen = zenith_at(t, ori_t, ori_zen)
        north = POLE[None, :] - (zen @ POLE)[:, None] * zen
        north /= np.linalg.norm(north, axis=1, keepdims=True)
        east = np.cross(north, zen)
        alt = np.arcsin(np.clip((v * zen).sum(1), -1, 1))
        az = np.mod(np.arctan2((v * east).sum(1), (v * north).sum(1)), 2 * np.pi)
        pix_ea = hp.ang2pix(nside, np.pi / 2 - alt, az)

        for name, pix in (('galactic', pix_gal), ('spacecraft', pix_sc), ('earth', pix_ea)):
            flat = (ie[ok] * shape[1] + ip[ok]) * npix + pix[ok]
            hists[name] += np.bincount(flat, minlength=np.prod(shape)).reshape(shape)
        print(f'{sl.stop}/{n}', flush=True)

    np.savez_compressed(out, e_edges=E_EDGES, phi_edges=PHI_EDGES, nside=nside, **hists)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 32)
