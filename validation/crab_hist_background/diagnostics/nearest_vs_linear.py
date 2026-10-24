import sys, warnings
import numpy as np
import astropy.units as u
from astropy.time import Time
warnings.filterwarnings('ignore', category=RuntimeWarning)
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromArrays
from cosipy.event_selection.energy_selection import EnergySelector
import cosipy.background_estimation.free_norm_hist_background as m
from cosipy.background_estimation.free_norm_hist_background import HistBackgroundTemplate, FreeNormHistBackgroundDensity as F, _unitless

d = '/home/user/dc4data/'
sc = SpacecraftHistory.open(d + 'DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits')
sc = sc.select_interval(Time('2028-03-23 10:36:00'), Time('2028-03-23 13:36:00'))
z = np.load(d + 'truth/bkg_crabwin.npz')
msk = ((z['TimeTags'] >= sc.tstart.unix) & (z['TimeTags'] < sc.tstop.unix) & (z['Distance'] >= 1)
       & (z['Energies'] >= 200) & (z['Energies'] < 5000))
t = Time(z['TimeTags'][msk], format='unix')
data = TimeTagEmCDSEventDataInSCFrameFromArrays(t.jd1, t.jd2, z['Energies'][msk], z['Chi local'][msk],
                                                np.pi / 2 - z['Psi local'][msk], z['Phi'][msk])
sel = EnergySelector([200, 5000] * u.keV)
print('events', data.nevents)

for name in sys.argv[1:]:
    tpl = HistBackgroundTemplate.open(d + name).smooth(psichi_fwhm=4 * u.deg, phi_fwhm=20 * u.deg, time_counts=500)
    lin = F(data, sc, tpl, energy_selection=sel)
    b = np.asarray(lin.expectation_density()); B = lin.expected_counts()
    ll_lin = np.sum(np.log(b / B))

    # Nearest bin everywhere
    time, energy, phi, psichi = m._event_arrays(data)
    intervals = m._event_intervals(time, sc)
    rocking = m.rocking_angle(sc).to_value(u.deg)[intervals]
    f_live = (sc.livetime / sc.intervals_duration).to_value(u.dimensionless_unscaled)[intervals]
    k = F._rocking_bins(tpl, rocking)
    rate = F._rate_per_keV(tpl)
    em_r = _unitless(tpl.rate_counts.axes['Em'], u.keV)
    r = rate[F._time_bins(tpl, time), em_r.find_bin(energy)]
    ea = _unitless(tpl.phi_counts.axes['Em'], u.keV); pa = _unitless(tpl.phi_counts.axes['Phi'], u.rad)
    pc = tpl.phi_counts.contents
    pphi = pc / (pc.sum(-1, keepdims=True) * pa.widths)
    pphi = np.nan_to_num(pphi)[k, ea.find_bin(energy), pa.find_bin(phi)]
    ps_ax = tpl.psichi_counts.axes
    ea2 = _unitless(ps_ax['Em'], u.keV); pa2 = _unitless(ps_ax['Phi'], u.rad); hax = ps_ax['PsiChi']
    sc_ = tpl.psichi_counts.contents
    ppsi = np.nan_to_num(sc_ / (sc_.sum(-1, keepdims=True) * hax.pixarea().to_value(u.sr)))
    ppsi = ppsi[k, ea2.find_bin(energy), pa2.find_bin(phi), hax.find_bin(psichi)]
    bn = r * f_live * pphi * ppsi
    # Expected counts with bin overlaps (exact for nearest)
    lt = sc.livetime.to_value(u.s); ob = sc.obstime.utc.unix; mid = (ob[:-1] + ob[1:]) / 2; act = lt > 0
    edges = em_r.edges
    overlap = np.clip(np.minimum(edges[1:], 5000) - np.maximum(edges[:-1], 200), 0, None)
    Bn = np.sum(lt[act] * (rate[F._time_bins(tpl, mid[act])] @ overlap))
    ll_near = np.sum(np.log(bn / Bn))
    print(f'{name}: B linear {B:.1f} nearest {Bn:.1f}; zero density: linear {np.sum(b <= 0)} nearest {np.sum(bn <= 0)}; '
          f'logL(nearest) - logL(linear) = {ll_near - ll_lin:+.1f}', flush=True)
    rl = F._rate(tpl, time, energy)
    pl = F._phi_density(tpl, rocking, energy, phi)
    sl = F._psichi_density(tpl, rocking, energy, phi, psichi)
    for lab, a, bb in (('Em rate', rl, r), ('phi', pl, pphi), ('PsiChi', sl, ppsi)):
        norm = (Bn / B) if lab == 'Em rate' else 1.
        print(f'  {lab}: nearest - linear = {np.sum(np.log(bb / a)) - data.nevents * np.log(norm):+.1f}', flush=True)
