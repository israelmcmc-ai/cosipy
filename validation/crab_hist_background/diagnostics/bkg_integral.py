import sys, time, warnings
import numpy as np
import astropy.units as u
from astropy.time import Time
warnings.filterwarnings('ignore', category=RuntimeWarning)
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.background_estimation.analytic_background import IsotropicPowerLawBackground
from cosipy.background_estimation.free_norm_hist_background import HistBackgroundTemplate, FreeNormHistBackgroundDensity

d = '/home/user/dc4data/'
sc = SpacecraftHistory.open(d + 'DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits')
sc = sc.select_interval(Time('2028-03-23 10:36:00'), Time('2028-03-23 13:36:00'))
erange = (200., 5000.)
template = HistBackgroundTemplate.open(d + sys.argv[1])
if len(sys.argv) > 2 and sys.argv[2] == 'smooth':
    template = template.smooth(psichi_fwhm=4 * u.deg, phi_fwhm=20 * u.deg, time_counts=500)
rate, index = 10 * u.Hz, -2.
rng = np.random.default_rng(0)
vals = []
for rep in range(5):
    ev = IsotropicPowerLawBackground.random_events(sc, rate, index, erange, rng)
    q = IsotropicPowerLawBackground(ev, sc, rate, index, erange)
    Q = q.expected_counts()
    b = FreeNormHistBackgroundDensity(ev, sc, template, energy_selection=EnergySelector(erange * u.keV))
    w = np.asarray(b.expectation_density()) / np.asarray(q.expectation_density()) * Q / b.expected_counts()
    vals.append(w)
    print(rep, w.size, w.mean(), w.std() / np.sqrt(w.size), flush=True)
w = np.concatenate(vals)
print('c_b =', w.mean(), '+-', w.std() / np.sqrt(w.size))
