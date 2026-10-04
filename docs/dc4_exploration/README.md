# DC4 mock dataset: exploratory CDS plots

Week 1 of the DC4 mock dataset (13,829,698 events, SAA cut applied) binned in
measured energy, Compton scattering angle phi and the scattered-photon direction
psichi, in three coordinate frames.

    # data: COSI-SMEX/DC4/Data/Mock_Dataset/dc4_mock_dataset_week_1_unbinned_data_filtered_with_SAAcut.fits.gz
    #       COSI-SMEX/DC4/Data/Orientation/DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits
    python bin_dc4.py week1.fits.gz ori15.fits week1_hist.npz 32          # optional 5th arg: min Distance [cm]
    python plot_dc4.py week1_hist.npz figures 4        # last argument: Gaussian FWHM [deg], 0 = none

Needs numpy, astropy, healpy, matplotlib (it does not use cosipy itself).

## Binning
- Energy: 10 log bins, 100 keV - 10 MeV. phi: 5 deg bins (36). psichi: HEALPix nside=32 (12288 pix).
- Frames (the psichi direction is the direction of the scattered gamma ray):
  - **Galactic**: `Chi galactic`, `Psi galactic` columns as (l, b).
  - **Spacecraft**: `Chi local` = longitude, `90 deg - Psi local` = latitude (z = boresight = lat +90).
  - **Earth horizon**: not in the event file, so computed per event. Zenith is the orientation
    file's `EarthZenith`, interpolated to the event time. North is the celestial pole (Earth's spin
    axis) projected onto the horizon plane. Azimuth runs from North towards East, altitude is the elevation above the horizon.
- Checked: local psichi rotated with the per-event x/z pointings (y = z cross x) matches the Galactic columns to < 0.06 deg.

## Figures (`figures/`)
- `overview_psichi.png`: psichi map summed over E and phi, with the slice disks of plot 2 marked.
- `psichi_maps_<frame>.png`: **plot 1**, psichi maps for energy slices (rows) x phi slices (columns), smoothed.
- `energy_vs_phi_<frame>.png`: **plot 2**, energy vs phi for psichi disks (r = 15 deg) around chosen directions.

Longitude increases to the left for Galactic and spacecraft (sky view) and to the right for the Earth frame (azimuth).
Slice choices (`E_SLICES`, `PHI_SLICES`, `SLICE_DIRS`, `DISK_RADIUS`) are at the top of `plot_dc4.py`.

## With a minimum first-to-second-hit distance cut
`figures_dist1cm/` has the same plots after requiring the event `Distance` column >= 1 cm
(`python bin_dc4.py ... week1_hist_d1cm.npz 32 1.0`). This keeps 57% of the week-1 events
(27% in the 100-158 keV bin, about 75% above 1 MeV).

## Earth frame split by survey mode (`figures_survey_modes/`, `figures_survey_modes_dist1cm/`)
The z-axis tilt from the Earth zenith is bimodal: +22 deg towards North ("survey north") or -22 deg
("survey south"), switching every ~12 h with ~8 min slews. `bin_dc4.py` labels each event from the
orientation file (|tilt| > 20 deg; slews, ~1% of events, are in neither mode) and writes
`earth_north` / `earth_south` histograms. `plot_dc4.py` then makes `psichi_maps_earth_survey_{north,south}.png`
with the spacecraft equator (great circle 90 deg from the z axis) overlaid in red, and
`energy_vs_phi_earth_by_mode.png`. The `_dist1cm` directory also applies Distance >= 1 cm.

## Spacecraft psichi matrix for Distance >= 8 cm (`figures_dist8cm/`)
`python bin_dc4.py ... week1_d8.npz 32 8`. Only 4.3% of week-1 events (600k) survive; the 100-158 keV
bin has 23k and the top bin 410 events, so the highest-energy panels are essentially noise.

## Earth-minus-spacecraft decoupling (`figures_earth_minus_sc/`, `figures_earth_minus_sc_dist1cm/`)
`plot_decouple.py` (needs the `spacecraft_north/south` histograms from `bin_dc4.py`). Per survey mode:
the SC-frame psichi map of that mode's events is averaged over chi, painted onto the Earth frame about
the mode's mean z axis, and subtracted from the Earth-frame map (counts/pixel, diverging colormap).
`decoupling_example.png` shows Earth map | chi-averaged SC model | difference for one slice.
Caveat: only the chi-symmetric SC part is removed. chi-dependent SC structure (e.g. the detector-geometry
stripes) remains, and since the SC x axis oscillates between az ~ +/-72 deg once per orbit it shows up
in the residual as azimuthal stripes.

### Full (chi-dependent) spacecraft subtraction
`plot_decouple_full.py hist.npz orientation.fits outdir` adds `psichi_maps_earth_minus_fullsc_survey_{north,south}.png`
and `decoupling_full_example.png` to the same directories. For each survey mode the SC-frame map is rotated into
the Earth frame with the attitude of every 15 s orientation sample (weighted by its event count) and summed, then
subtracted from the Earth map. The rotation code reproduces the axisymmetric model to 0.2% when fed a chi-averaged map.
std(residual)/sqrt(mean counts) in week 1 (1.0 = Poisson): above ~250 keV 1.0-1.1 (was 1.0-1.2 with the chi average);
100-251 keV, phi 30-120 deg: 1.16-1.45 (was 1.4-1.7), so a broad excess remains there.
Caveat: the SC template is built from the same events and still contains the roll-smeared Earth signal.

## Albedo component only (`figures_albedo/`)
Input: `COSI-SMEX/DC4/Data/Backgrounds/AlbedoPhotons_WithDetCstunbinned_data_filtered_with_SAAcut.fits.gz`
(the "albedo" component of the DC4 background; albedo neutrons are a separate, much smaller file).
It holds 27.1M events over the full 3 months (not time-ordered), all of which are binned (no Distance cut):
`python bin_dc4.py AlbedoPhotons...fits ori15.fits albedo.npz 32 0; python plot_dc4.py albedo.npz outdir 4`.
Figures: spacecraft-frame matrix and Earth-frame matrices for survey north / south (red dashed = SC equator).
