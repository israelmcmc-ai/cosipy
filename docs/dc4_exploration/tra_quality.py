"""Event-level quantities from the output of tra_events.awk, shared by the signal and background scripts.

Columns of the awk output: TI SQ Eg Ee LA x1 y1 z1 x2 y2 z2 Etrue dx dy dz  (see tra_events.awk).
 * Emeas = Eg + Ee [keV]; phi = Compton angle from the energies, cos(phi) = 1 - 510.999 (1/Eg - 1/Emeas) (events with |cos| > 1 are dropped).
 * LA = lever arm = the MINIMUM consecutive-hop distance of the Compton sequence (equal to the first-second distance only for 2-hit events);
   this is what the FITS 'Distance' column holds.
 * cat1 / cat2: depth category of the first / second hit: 3 * layer + type; type 0 = bulk, 1 = exactly on the middle plane of the layer,
   2 = within 0.10 cm of the top or bottom surface of the layer (layer 0 = lowest z).
 * arm [deg] (simulations only): angle(true propagation direction, scattered direction g = (p2 - p1)/|p2 - p1|) - phi.
"""
import numpy as np

E_EDGES = np.geomspace(200., 5000., 7)           # keV, 6 log bins
LAYER_Z = np.array([12.0, 14.2, 16.9, 19.3, 22.0])
LAYER_LO = np.array([12.257, 14.823, 17.389, 19.955])
LAYER_HI = np.array([13.756, 16.322, 18.888, 21.454])
LAYER_MID = np.array([13.0067, 15.5727, 18.1387, 20.7047])
SURFACE = 0.10     # cm
ME = 510.999       # keV


def depth_category(z):
    layer = np.clip(np.digitize(z, LAYER_Z) - 1, 0, 3)
    mid = np.abs(z - LAYER_MID[layer]) < 0.0015
    surf = ((z - LAYER_LO[layer]) < SURFACE) | ((LAYER_HI[layer] - z) < SURFACE)
    return (3 * layer + np.where(mid, 1, np.where(surf, 2, 0))).astype(np.int8)


def derive(a):
    """a: (N, 15) array -> dict of float32/int arrays for the events with a defined Compton angle in 200-5000 keV."""
    t, sq, eg, ee, la, x1, y1, z1, x2, y2, z2, et, dx, dy, dz = a.T
    em = eg + ee
    with np.errstate(divide='ignore', invalid='ignore'):
        cosphi = 1.0 - ME * (1.0 / eg - 1.0 / em)
    ok = np.isfinite(cosphi) & (np.abs(cosphi) <= 1) & (em >= E_EDGES[0]) & (em < E_EDGES[-1]) & (eg > 0) & (ee > 0)
    a = a[ok]; t, sq, eg, ee, la, x1, y1, z1, x2, y2, z2, et, dx, dy, dz = a.T
    em = eg + ee; phi = np.arccos(1.0 - ME * (1.0 / eg - 1.0 / em))
    g = np.stack([x2 - x1, y2 - y1, z2 - z1], 1); d01 = np.linalg.norm(g, axis=1); g /= d01[:, None]
    out = dict(t=t, sq=sq.astype(np.int8), E=em.astype(np.float32), phi=phi.astype(np.float32), LA=la.astype(np.float32),
               d01=d01.astype(np.float32), z1=z1.astype(np.float32), z2=z2.astype(np.float32),
               cat1=depth_category(z1), cat2=depth_category(z2))
    if np.any(et > 0):
        cosgeo = np.clip(dx * g[:, 0] + dy * g[:, 1] + dz * g[:, 2], -1, 1)
        out['arm'] = np.rad2deg(np.arccos(cosgeo) - phi).astype(np.float32)
        out['Etrue'] = et.astype(np.float32)
        out['theta_src'] = np.rad2deg(np.arccos(np.clip(-dz, -1, 1))).astype(np.float32)   # polar angle of the source direction from +z
    return out
