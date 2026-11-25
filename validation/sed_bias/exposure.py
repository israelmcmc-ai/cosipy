import sys, numpy as np, astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from cosipy.spacecraftfile import SpacecraftHistory
S = sys.argv[1]
sc = SpacecraftHistory.open(f'{S}/data/DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits')
res = {}
srcs = {'crab': SkyCoord(l=184.56*u.deg, b=-5.78*u.deg, frame='galactic'),
        'grb': SkyCoord(l=21.293418951586656*u.deg, b=-42.384846594045484*u.deg, frame='galactic')}
windows = {'crab3h': ('crab', Time('2028-03-23 10:36:00'), Time('2028-03-23 13:36:00')),
           'crab3m': ('crab', None, None),
           'grb': ('grb', Time('2028-03-24 10:36:42'), Time('2028-03-24 10:36:51.921'))}
for name, (src, t0, t1) in windows.items():
    s = sc if t0 is None else sc.select_interval(t0, t1)
    for nside in (8, 32):
        for occ in (False, True):
            m = s.get_dwell_map(srcs[src], nside=nside, earth_occ=occ)
            res[f'{name}_n{nside}_occ{int(occ)}'] = (m.data * m.unit).to_value(u.s)
            print(name, nside, occ, m.data.sum(), m.unit, flush=True)
np.savez(f'{S}/out/exposure.npz', **res)
