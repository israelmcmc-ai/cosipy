import glob, warnings
import numpy as np
import astropy.units as u
from astropy.time import Time
warnings.filterwarnings('ignore')
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromArrays
from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.background_estimation import HistBackgroundTemplate, FreeNormHistBackgroundDensity, rocking_angle

d = '/home/user/dc4data/'
sel = EnergySelector([200, 5000] * u.keV)
sc_full = SpacecraftHistory.open(d + 'DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits')
crab = np.load(d + 'crabfull/crab_all.npz')
crab = {k: crab[k] for k in crab.files}
template = HistBackgroundTemplate.open(d + 'bkgonly_template_dist1cm.h5').smooth(psichi_fwhm=4 * u.deg, phi_fwhm=20 * u.deg, time_counts=500)

W, C = [], []
for f in sorted(glob.glob(d + 'sref/w*.npz')):
    z = np.load(f)
    t0 = Time(str(z['start']), scale='utc')
    sc = sc_full.select_interval(t0, t0 + 3 * u.h)
    t = Time(z['unix'], format='unix', scale='utc')
    data = TimeTagEmCDSEventDataInSCFrameFromArrays(t.jd1, t.jd2, z['energy'], z['lon'], z['lat'], z['phi'])
    bkg = FreeNormHistBackgroundDensity(data, sc, template, energy_selection=sel)
    b = np.asarray(bkg.expectation_density()) * data.nevents / bkg.expected_counts()
    rock = rocking_angle(sc).to_value(u.deg)
    tb = sc.obstime.utc.unix
    def rk(unix):
        return rock[np.clip(np.searchsorted(tb, unix, side='right') - 1, 0, rock.size - 1)]
    W.append(dict(w=z['s'] / b, E=z['energy'], phi=np.rad2deg(z['phi']), elev=np.rad2deg(z['lat']), rock=rk(z['unix']),
                  tfrac=(z['unix'] - sc.tstart.utc.unix) / 10800))
    m = ((crab['TimeTags'] >= sc.tstart.utc.unix) & (crab['TimeTags'] < sc.tstop.utc.unix) & (crab['Distance'] >= 1)
         & (crab['Energies'] >= 200) & (crab['Energies'] < 5000))
    C.append(dict(E=crab['Energies'][m], phi=np.rad2deg(crab['Phi'][m]), elev=90 - np.rad2deg(crab['Psi local'][m]),
                  rock=rk(crab['TimeTags'][m]), tfrac=(crab['TimeTags'][m] - sc.tstart.utc.unix) / 10800, S=float(z['S'])))

w = {k: np.concatenate([x[k] for x in W]) for k in W[0]}
c = {k: np.concatenate([x[k] for x in C]) for k in ('E', 'phi', 'elev', 'rock', 'tfrac')}
print(f'all: sum s/b {w["w"].sum():.0f}, N crab {c["E"].size}, sum S {sum(x["S"] for x in C):.0f}')
for var, edges in (('E', [200, 300, 450, 700, 1000, 1500, 2500, 5000]), ('phi', [0, 20, 40, 60, 90, 120, 180]),
                   ('elev', [-90, -30, 0, 15, 30, 45, 60, 90]), ('rock', [-90, 0, 90]), ('tfrac', [0, .25, .5, .75, 1.0001])):
    print(var)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sw = w['w'][(w[var] >= lo) & (w[var] < hi)].sum()
        n = np.sum((c[var] >= lo) & (c[var] < hi))
        print(f'   [{lo}, {hi}): sum s/b {sw:7.0f}   N crab {n:6d}   diff {sw - n:+7.0f} +- {np.sqrt(n):.0f}')
