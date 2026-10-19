#!/usr/bin/env python
# coding: UTF-8

"""
Build the raw-count HistBackgroundTemplate files used by
FreeNormHistBackgroundDensity out of the DC4 mock dataset weekly files.

For each weekly file (given in time order), the events are read once and two
templates are filled: one with all the events, and one with the events with
Distance >= 1 cm (the quality cut needed by the "hist_simple" response). The
templates are written unsmoothed (raw counts) to

    <prefix>.h5
    <prefix>_dist1cm.h5

Livetime
--------
Each file is paired with its own, non-overlapping slice of the spacecraft
orientation history, so no livetime is counted twice. The slices start at the
first event of the first file, end at the last event of the last file, and
consecutive files are split at the last event of the earlier one. The total
template livetime is compared at the end with the one of the orientation file
over that range.

Binning
-------
The constants below define the binning. Edit them to change it.

Usage
-----
python build_hist_background_template.py ORIENTATION_FILE PREFIX \
    --weeks 1 2 3 ... [--data-dir DIR] [--fetch] [--delete]

Each week is read from DIR/dc4_mock_dataset_week_{i}_unbinned_data_filtered_with_SAAcut.fits.gz.
With --fetch, missing files are downloaded from Wasabi, and with --delete, each
file is removed after it is used.
"""

import argparse
import resource
import time as timer
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.time import Time
from histpy import Axes, Axis, HealpixAxis
from scoords import SpacecraftFrame

from cosipy import SpacecraftHistory
from cosipy.background_estimation import HistBackgroundTemplate
from cosipy.data_io.EmCDSUnbinnedData import (
    TimeTagEmCDSDistanceEventDataInSCFrameFromArrays,
    TimeTagEmCDSDistanceEventDataInSCFrameFromDC3Fits,
)
from cosipy.event_selection import DistanceSelector
from cosipy.util import fetch_wasabi_file

TIME_BIN_S = 300.
RATE_EM_EDGES_KEV = np.geomspace(100., 1e4, 30 + 1)
PHI_EM_EDGES_KEV = np.geomspace(100., 1e4, 30 + 1)
PHI_EDGES_RAD = np.deg2rad(np.arange(0., 180.1, 2.))
PSICHI_EM_EDGES_KEV = np.geomspace(100., 1e4, 8 + 1)
PSICHI_PHI_EDGES_RAD = np.deg2rad(np.arange(0., 180.1, 10.))
PSICHI_NSIDE = 32
ROCKING_EDGES_DEG = [-90., 0., 90.]
MIN_DISTANCE = 1 * u.cm

WEEK_FILE = "dc4_mock_dataset_week_{}_unbinned_data_filtered_with_SAAcut.fits.gz"
WASABI_DIR = "COSI-SMEX/DC4/Data/Mock_Dataset"


def make_axes(tstart_unix, tstop_unix):
    start = np.floor(tstart_unix / TIME_BIN_S) * TIME_BIN_S
    nbins = int(np.ceil((tstop_unix - start) / TIME_BIN_S))
    time = Axis(start + TIME_BIN_S * np.arange(nbins + 1), label='Time')
    rocking = Axis(ROCKING_EDGES_DEG, label='Rocking', unit=u.deg)

    return (Axes([time, Axis(RATE_EM_EDGES_KEV, scale='log', label='Em')]),
            Axes([rocking, Axis(PHI_EM_EDGES_KEV, scale='log', label='Em'),
                  Axis(PHI_EDGES_RAD, label='Phi')]),
            Axes([rocking, Axis(PSICHI_EM_EDGES_KEV, scale='log', label='Em'),
                  Axis(PSICHI_PHI_EDGES_RAD, label='Phi'),
                  HealpixAxis(nside=PSICHI_NSIDE, scheme='ring', coordsys=SpacecraftFrame(), label='PsiChi')]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('orientation', help="Orientation file (SpacecraftHistory)")
    parser.add_argument('prefix', help="Output prefix")
    parser.add_argument('--weeks', type=int, nargs='+', default=list(range(1, 15)),
                        help="Mock dataset weeks, in time order. Default: 1-14")
    parser.add_argument('--data-dir', type=Path, default=Path('.'), help="Directory of the weekly files")
    parser.add_argument('--fetch', action='store_true', help="Download missing weekly files from Wasabi")
    parser.add_argument('--delete', action='store_true', help="Delete each weekly file after using it")
    args = parser.parse_args()

    sc_history = SpacecraftHistory.open(args.orientation)
    axes = make_axes(sc_history.tstart.utc.unix, sc_history.tstop.utc.unix)

    templates = {'': HistBackgroundTemplate.empty(*axes),
                 '_dist1cm': HistBackgroundTemplate.empty(*axes)}
    distance_selector = DistanceSelector(min_distance=MIN_DISTANCE)

    slice_start = None

    for iweek, week in enumerate(args.weeks):
        t0 = timer.time()
        path = args.data_dir / WEEK_FILE.format(week)

        if not path.exists() and args.fetch:
            fetch_wasabi_file(f"{WASABI_DIR}/{WEEK_FILE.format(week)}", output=str(path))

        data = TimeTagEmCDSDistanceEventDataInSCFrameFromDC3Fits(path)
        tevents = data.time
        tmin, tmax = tevents.min(), tevents.max()

        if slice_start is None:
            slice_start = tmin
        elif tmin < slice_start:
            raise ValueError(f"Week {week} starts before the previous one ended. Give the weeks in time order.")

        week_history = sc_history.select_interval(slice_start, tmax)

        selected = TimeTagEmCDSDistanceEventDataInSCFrameFromArrays(
            data.jd1, data.jd2, data.energy_keV, data.scattered_lon_rad_sc, data.scattered_lat_rad_sc,
            data.scattering_angle_rad, data.distance_cm, selection=distance_selector)

        filled = {}
        for label, events in (('', data), ('_dist1cm', selected)):
            before = templates[label].rate_counts.contents.sum()
            templates[label].fill(events, week_history)
            filled[label] = int(templates[label].rate_counts.contents.sum() - before)

        print(f"Week {week}: read {data.nevents} events, filled {filled['']} "
              f"({selected.nevents} read and {filled['_dist1cm']} filled with distance >= {MIN_DISTANCE}), "
              f"livetime {week_history.cumulative_livetime().to_value(u.s):.1f} s, "
              f"{timer.time() - t0:.1f} s", flush=True)

        slice_start = tmax
        covered_start = week_history.tstart if iweek == 0 else covered_start

        del data, selected

        if args.delete:
            path.unlink()

    expected = sc_history.select_interval(covered_start, slice_start).cumulative_livetime().to_value(u.s)

    for label, template in templates.items():
        filename = f"{args.prefix}{label}.h5"
        template.write(filename, overwrite=True)
        print(f"Wrote {filename}: {template.rate_counts.contents.sum():.0f} counts in the Em range of "
              f"the rate histogram, template livetime {template.livetime.contents.sum():.3f} s, "
              f"orientation livetime {expected:.3f} s")

    print(f"Peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")


if __name__ == '__main__':
    main()
