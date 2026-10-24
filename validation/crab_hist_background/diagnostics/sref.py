# %% [markdown]
# # Crab fits with the histogram background on the DC4 mock data
#
# Unbinned fits of the Crab in 3 h windows spread over the 3 months of DC4, with the
# `hist_simple` response folded by `UnbinnedThreeMLPointSourceResponseTrapz` and the
# data-driven `FreeNormHistBackgroundDensity`. The events are the simulated DC4 Crab (nebula +
# pulsar) and the simulated DC4 background, which are available separately, so the fits can be
# compared on the same Crab events:
#
# - **signal only**: the Crab events alone (no background). The reference for the others: it
#   only depends on the response and the folding (validated in `validation/closure/`).
# - **bkg-only template**: Crab + background events, with the template built from the 3 months
#   of the background simulation alone (an ideal template, without sources).
# - **all-events template**: the same events, with the template built from the 3 months of
#   the mock dataset (background + all the sources), as one would do with real data.
# - **mock data** (first window only, the one of the Crab tutorial): the mock dataset itself,
#   which also contains the other sources (~5% of the events), with the all-events template.
#
# Cuts: Distance >= 1 cm (as the `hist_simple` response) and 200 keV - 5 MeV. The fitted model
# is a power law, which is not exactly the injected spectrum; the comparison is between fits.

# %%
import gc
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.time import Time
from astromodels import Powerlaw, PointSource
from scipy.optimize import minimize

from cosipy.background_estimation import HistBackgroundTemplate, FreeNormHistBackgroundDensity
from cosipy.data_io.EmCDSUnbinnedData import (TimeTagEmCDSEventDataInSCFrameFromArrays,
                                              TimeTagEmCDSDistanceEventDataInSCFrameFromDC3Fits)
from cosipy.event_selection import DistanceSelector, ChainEventSelectors
from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.event_selection.time_selection import TimeSelector
from cosipy.response.broadened_irf_cache import BroadenedIRFCache
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized
from cosipy.spacecraftfile import SpacecraftHistory
from cosipy.threeml.psr_fixed_ei import UnbinnedThreeMLPointSourceResponseTrapz

# divide-by-zero warnings of the empty phase-space bins, once per evaluation
warnings.filterwarnings('ignore', category=RuntimeWarning)

# threeML (imported by cosipy) sets a non-interactive backend without a display


# %% [markdown]
# Inputs: the response, the orientation, the two 3-month templates
# (`COSI-SMEX/develop/Data/Products_and_Templates/Background_Models/HistBackgroundTemplate/`,
# Distance >= 1 cm), the week-4 mock dataset, and the simulated Crab and background events
# (`Crab_DC4_3months...` and `Total_DC4_BG_3months...` in `COSI-SMEX/DC4/Data/`), extracted
# in each window as arrays of the FITS columns.

# %%
data_path = Path('/home/user/dc4data')
irf_file = data_path / 'ResponseContinuum.area.relative.nonsparse_smoothing1p0.h5'
templates_files = {'bkg-only template': data_path / 'bkgonly_template_dist1cm.h5',
                   'all-events template': data_path / 'dc4_mock_bkg_template_dist1cm.h5'}
mock_file = data_path / 'dc4_mock_dataset_week_4_unbinned_data_filtered_with_SAAcut.fits.gz'
crab_file = data_path / 'crabfull/crab_all.npz'
windows = {'2028-03-23 10:36:00': data_path / 'truth/bkg_crabwin.npz'}
for start in (1836210468, 1838456868, 1839580068, 1840703268, 1841826468, 1842949668):
    windows[Time(start, format='unix').utc.iso[:19]] = data_path / f'windows/bkg_{start}.npz'

erange = (200., 5000.)
selection = EnergySelector(erange * u.keV)
sc_full = SpacecraftHistory.open(data_path / 'DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits')

spectrum = Powerlaw(index=-2.2, K=2.5e-6, piv=1000.)
spectrum.K.unit = 1 / u.cm ** 2 / u.s / u.keV
spectrum.piv.unit = u.keV
source = PointSource('Crab', l=184.56, b=-5.78, spectral_shape=spectrum)

irf = IRFRelativeHistUnpolarized.from_h5(irf_file, selections=selection)
cache = BroadenedIRFCache(IRFRelativeHistUnpolarized.from_h5(irf_file), energy_range_keV=(100., 10_000.))
cells = UnbinnedThreeMLPointSourceResponseTrapz.cells_for(source, cache, accuracy=1e-3)

# Smoothing chosen by cross-validation (see the HistBackgroundTemplate docs)
templates = {name: HistBackgroundTemplate.open(f).smooth(psichi_fwhm=4 * u.deg, phi_fwhm=20 * u.deg, time_counts=500)
             for name, f in templates_files.items()}

# %%
def load_events(filename, tstart, tstop):
    """Simulated events of a window as arrays, after the cuts"""
    z = np.load(filename)
    m = ((z['TimeTags'] >= tstart.utc.unix) & (z['TimeTags'] < tstop.utc.unix) & (z['Distance'] >= 1)
         & (z['Energies'] >= erange[0]) & (z['Energies'] < erange[1]))
    return {'unix': z['TimeTags'][m], 'energy': z['Energies'][m], 'lon': z['Chi local'][m],
            'lat': np.pi / 2 - z['Psi local'][m], 'phi': z['Phi'][m]}


def to_data(parts):
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    t = Time(arrays['unix'], format='unix', scale='utc')
    return TimeTagEmCDSEventDataInSCFrameFromArrays(t.jd1, t.jd2, arrays['energy'], arrays['lon'], arrays['lat'],
                                                    arrays['phi'])


def mock_events(sc):
    cut = ChainEventSelectors(DistanceSelector(min_distance=1 * u.cm), TimeSelector(tstart=sc.tstart, tstop=sc.tstop),
                              selection)
    d = TimeTagEmCDSDistanceEventDataInSCFrameFromDC3Fits([mock_file], selection=cut)
    return {'unix': d.time.utc.unix, 'energy': np.asarray(d.energy_keV), 'lon': np.asarray(d.scattered_lon_rad_sc),
            'lat': np.asarray(d.scattered_lat_rad_sc), 'phi': np.asarray(d.scattering_angle_rad)}


def set_spectrum(log_k, index):
    spectrum.K.value = np.exp(log_k)
    spectrum.index.value = index


def fit(psr, idx, bkg_density=None, livetime=None, x0=(np.log(2.5e-6), -2.2, None)):
    """
    Maximum likelihood K, index (and background rate in Hz) with the events `idx`.
    `bkg_density` is the background density per Hz of rate, at all the events.
    """
    def negloglike(x):
        set_spectrum(x[0], x[1])
        s = np.asarray(psr.expectation_density())[idx]
        S = psr.expected_counts()
        if bkg_density is None:
            return -(np.sum(np.log(s)) - S)
        rate = np.exp(x[2])
        return -(np.sum(np.log(s + rate * bkg_density[idx])) - S - rate * livetime)

    start = list(x0[:2]) + ([] if bkg_density is None else [x0[2]])
    result = minimize(negloglike, start, method='Nelder-Mead', options=dict(xatol=1e-5, fatol=1e-4, maxiter=4000))
    errors = np.sqrt(np.diag(np.linalg.inv(hessian(negloglike, result.x))))
    set_spectrum(*result.x[:2])
    S = psr.expected_counts()
    return {'K': np.exp(result.x[0]), 'K error': np.exp(result.x[0]) * errors[0], 'index': result.x[1],
            'index error': errors[1], 'S': S, 'B': None if bkg_density is None else np.exp(result.x[2]) * livetime}


def hessian(f, x, step=1e-2):
    """Central finite differences"""
    n = len(x)
    h = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            def shifted(si, sj):
                y = np.array(x, dtype=float)
                y[i] += si * step
                y[j] += sj * step
                return f(y)
            h[i, j] = h[j, i] = (shifted(1, 1) - shifted(1, -1) - shifted(-1, 1) + shifted(-1, -1)) / (4 * step ** 2)
    return h


import os
os.makedirs('/home/user/dc4data/sref', exist_ok=True)
set_spectrum(np.log(2.6e-6), -2.38)
for w, (start, bkg_file) in enumerate(windows.items()):
    if os.path.exists(f'/home/user/dc4data/sref/w{w}.npz'):
        continue
    tstart = Time(start, scale='utc')
    sc = sc_full.select_interval(tstart, tstart + 3 * u.h)
    bk = load_events(bkg_file, sc.tstart, sc.tstop)
    data = to_data([bk])
    psr = UnbinnedThreeMLPointSourceResponseTrapz(data, irf, sc, cache, cells)
    psr.set_source(source); psr.init_cache()
    np.savez(f'/home/user/dc4data/sref/w{w}.npz', s=np.asarray(psr.expectation_density()), S=psr.expected_counts(),
             start=start, **bk)
    print(start, psr.expected_counts(), flush=True)
    del psr, data; gc.collect()
