import sys, glob, warnings
import numpy as np
import astropy.units as u
from astropy.time import Time
warnings.filterwarnings('ignore')
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromArrays
from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.background_estimation import HistBackgroundTemplate, FreeNormHistBackgroundDensity

d = '/home/user/dc4data/'
sel = EnergySelector([200, 5000] * u.keV)
sc_full = SpacecraftHistory.open(d + 'DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits')
windows = []
for f in sorted(glob.glob(d + 'sref/w*.npz')):
    z = np.load(f)
    t0 = Time(str(z['start']), scale='utc')
    sc = sc_full.select_interval(t0, t0 + 3 * u.h)
    t = Time(z['unix'], format='unix', scale='utc')
    data = TimeTagEmCDSEventDataInSCFrameFromArrays(t.jd1, t.jd2, z['energy'], z['lon'], z['lat'], z['phi'])
    windows.append((str(z['start']), sc, data, z['s'], float(z['S'])))

raw = {name: HistBackgroundTemplate.open(d + f) for name, f in
       (('bkg-only', 'bkgonly_template_dist1cm.h5'), ('all-events', 'dc4_mock_bkg_template_dist1cm.h5'))}

variants = {
    'default (4 deg, 20 deg, 500)': dict(psichi_fwhm=4 * u.deg, phi_fwhm=20 * u.deg, time_counts=500),
    'psichi 2 deg': dict(psichi_fwhm=2 * u.deg, phi_fwhm=20 * u.deg, time_counts=500),
    'psichi 8 deg': dict(psichi_fwhm=8 * u.deg, phi_fwhm=20 * u.deg, time_counts=500),
    'no phi smoothing': dict(psichi_fwhm=4 * u.deg, time_counts=500),
    'time_counts 100': dict(psichi_fwhm=4 * u.deg, phi_fwhm=20 * u.deg, time_counts=100),
    'no angular smoothing': dict(time_counts=500),
}
which = sys.argv[1:] or list(variants)
for tname, template in raw.items():
    for vname in which:
        t = template.smooth(**variants[vname])
        ratios, errs = [], []
        for start, sc, data, s, S in windows:
            bkg = FreeNormHistBackgroundDensity(data, sc, t, energy_selection=sel)
            b = np.asarray(bkg.expectation_density()) * data.nevents / bkg.expected_counts()
            w = s / b
            ratios.append(w.sum() / S); errs.append(np.sqrt(np.sum(w ** 2)) / S)
        ratios, errs = np.array(ratios), np.array(errs)
        print(f'{tname:10s} {vname:30s} mean {ratios.mean():.4f} +- {np.sqrt(np.sum(errs ** 2)) / len(errs):.4f}  '
              f'[{", ".join(f"{r:.3f}" for r in ratios)}]', flush=True)
