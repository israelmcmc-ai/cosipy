import sys, warnings
import numpy as np
import astropy.units as u
from astropy.time import Time
warnings.filterwarnings('ignore', category=RuntimeWarning)
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.background_estimation.free_norm_hist_background import HistBackgroundTemplate, FreeNormHistBackgroundDensity as F, _unitless
from cosipy.event_selection.energy_selection import EnergySelector
d = '/home/user/dc4data/'
t = HistBackgroundTemplate.open(d + sys.argv[1]).smooth(psichi_fwhm=4 * u.deg, phi_fwhm=20 * u.deg, time_counts=500)
sc = SpacecraftHistory.open(d + 'DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits')
sc = sc.select_interval(Time('2028-03-23 10:36:00'), Time('2028-03-23 13:36:00'))
print('Em edges', _unitless(t.rate_counts.axes['Em'], u.keV).edges)
B = F._predicted_counts(t, sc, EnergySelector([200, 5000] * u.keV))
lt = sc.livetime.to_value(u.s); ob = sc.obstime.utc.unix; mid = (ob[:-1] + ob[1:]) / 2
act = lt > 0
e = np.geomspace(200, 5000, 200001); em = np.sqrt(e[1:] * e[:-1]); de = np.diff(e)
tot = 0.
for tm, l in zip(mid[act], lt[act]):
    pass
tb = F._time_bins(t, mid[act])
ub, inv = np.unique(tb, return_inverse=True)
lt_b = np.bincount(inv, weights=lt[act])
for b, l in zip(ub, lt_b):
    r = F._rate(t, np.full(em.size, t.livetime.axis.centers[b]), em)
    tot += l * np.sum(r * de)
print('B (bins) =', B, ' integral of interpolated rate =', tot, ' ratio', tot / B)
