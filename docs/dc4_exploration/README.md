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

### Albedo with Distance >= 1 cm (`figures_albedo_dist1cm/`)
`bin_dc4.py AlbedoPhotons...fits ori15.fits albedo_d1.npz 32 1` keeps 18.9M of the 27.1M albedo events (70%).
- `psichi_maps_spacecraft.png`, `psichi_maps_earth_survey_{north,south}.png`: plain matrices with the cut.
- `plot_minus_total.py` -> `psichi_maps_spacecraft_minus_total.png` and `psichi_maps_earth_minus_total_sc_survey_{north,south}.png`:
  each slice minus the component's total (sum over E and phi) SC-frame map scaled to the slice counts; for the Earth frame
  the mode's total SC map is rotated into the Earth frame per orientation sample first (slow: ~10 min for 3 months).
- `narrow_phi_sc.py` -> `narrow_phi_sc_{raw,minus_total}.png`: SC frame, 1 deg phi bins (40, 70, 100, 130, 160 deg) for
  100-251, 251-631 and 631-1585 keV; nside 32 and 4 deg smoothing because the bins hold only 3k-80k events.

## Sensitivity (S/sqrt(B)) vs Distance and ARM (`figures_sensitivity/`)
Signal: `Sources/Crab_DC4_3months_...fits.gz` (3.84M events). Background: `Backgrounds/Total_DC4_BG_3months_..._withSAAbck.fits.gz`
(168.6M events, streamed from Wasabi without a local copy). 6 log energy bins, 200 keV - 5 MeV. Only percentage changes of
S/sqrt(B) relative to no cuts are meaningful. ARM = angle(Crab, scattered-photon direction) - phi, from the Galactic Chi/Psi columns
(Crab peak at 0 +/- 0.4 deg, FWHM 11 deg at 200-342 keV down to 3 deg at 1.7-2.9 MeV). Background events are kept only when the
Crab is above the Earth limb (zenith angle < 113 deg), as in the Crab simulation (76% of the in-range events).
Pipeline: `sensitivity.py` (streams events -> (E, Distance, ARM) histograms in the .npz files, which are included) and
`sensitivity_plots.py` (figures + `summary.txt`). Number of hits and first-hit z are not in the FITS files; pending a Crab file with hit info.

`figures_sensitivity/sensitivity_vs_removed_fraction.png`: change in S/sqrt(B) from a Distance cut with the ARM half-width re-optimized
(0.25-30 deg) at every cut and energy bin, relative to the optimized-ARM no-cut value, vs the fraction of (signal + background)
events removed by the cut. `summary_removed_fraction.txt` lists the values at 0.5, 1, 2 and 5 cm.

## Other background components, spacecraft frame only (`figures_backgrounds_sc_dist1cm/`, flat, one suffix per component)
`run_components.sh outdir 1` streams each component from Wasabi and bins it with Distance >= 1 cm (11 components of the mock dataset:
DC4 albedo neutrons, primary alphas / electrons / positrons / protons, SAA, secondary positrons / protons; DC3 secondary electrons,
Galactic diffuse (`GalTotal_SA100_F98`) and cosmic photons; albedo photons is under `figures_albedo_dist1cm/`). `plot_sc_components.py bindir plotdir`
then makes, per component, `psichi_maps_spacecraft_<Component>.png` and `minus_total_psichi_maps_spacecraft_<Component>.png` (no Earth-frame step; palette-quantized PNGs).
Loop over the plain matrices with the glob `psichi_maps_spacecraft_*.png` (12 files, including `AlbedoPhotons`, copied from `figures_albedo_dist1cm/`).
Events in 200 keV-10 MeV x all phi after the cut (`plot_sc_components.py` output): cosmic photons 48.8M, primary protons 10.2M, primary alphas 5.5M,
secondary positrons 4.2M, Galactic diffuse 4.2M, SAA 3.5M, albedo neutrons 1.2M, secondary electrons 0.77M, secondary protons 0.46M,
primary electrons 26k, primary positrons 1.9k (the last two are noisy). (`plot_component.sh`/`plot_components.sh` also include the slow Earth-frame subtraction.)

## Albedo, spacecraft frame, by survey mode and roll angle (`figures_albedo_roll_modes_dist1cm/`)
`roll_modes.py` (streams the albedo file, Distance >= 1 cm) + `plot_roll_modes.py roll.npz outdir AlbedoPhotons`.
Roll angle = angle, counterclockwise about SC +z, from +x to the projection of the Earth zenith onto the SC xy plane (the SC azimuth chi
of the zenith), in 10 deg bins from -180 to 180. Survey north occupies roll |100-180| deg (zenith towards -x) and survey south -80..+80 deg
(zenith towards +x); slews (1% of events) are excluded. One energy x phi matrix per occupied (mode, roll bin) with >= 20k events:
`psichi_maps_spacecraft_AlbedoPhotons_survey_<north|south>_roll_<lo>to<hi>.png` (16 per mode), plus `roll_distribution_AlbedoPhotons.png`.
Check: the first azimuthal Fourier mode of the -z-side emission (psi 120-155 deg) has phase = roll + 180 deg to within a few deg on average
(std 4-17 deg), amplitude 2-9%, so the roll-dependent part is a modest modulation on top of the roll-independent SC pattern.

## Spacecraft-frame projections on SC latitude and SC longitude (`figures_dist1cm/psichi_spacecraft_vs_{latitude,longitude}_1deg.png`)
Same week-1 data, Distance >= 1 cm, and energy x phi matrix as `figures_dist1cm/psichi_maps_spacecraft.png`, but each panel is a 1-D
histogram in 1 deg bins with no smoothing (shaded band = +/- sqrt(N)): vs SC latitude (90 deg - psi, summed over longitude) and vs
SC longitude (chi, summed over latitude). `sc_lonlat.py events.fits.gz out_prefix 1` (also keeps the 2-D 360 x 180 counts in the .npz).
Counts per latitude bin include the cos(lat) solid-angle factor. Features: a 1-2 deg wide spike at latitude 0 (psi = 90 deg, scatters in the
detector plane) and sharp spikes/dips at longitudes 0, 90, 180, 270 deg plus finer structure at the 1 deg scale (detector geometry).

### Spikes and dips at SC longitude 0/90/180/270 deg (`figures_dist1cm/sc_longitude_spike_fold90.png`, `sc_longitude_spike_vs_distance.png`)
All distances combined (Distance >= 1 cm), the 5 deg bin centred on a spike averages ~17% below the global mean (~21% below its neighbours);
the spike excess covers only ~45% of the dips' deficit, the rest is roughly balanced by slightly elevated shoulders at 5-13 deg, so the total is
conserved to a few % only over about +-13 deg. In bands of the hit separation (`lon_vs_dist.py`) the empty gap around each axis shrinks as 1/Distance:
|offset| < ~3.3 deg for 1-2 cm, ~2.2 for 2-3, ~1.2 for 3-5, ~0.8 for 5-8 and ~0.3 deg for 8-30 cm, i.e. tan(gap) * Distance ~ 1.1 mm,
the signature of quantized hit positions (events with a transverse offset below one position quantum are reconstructed exactly on the axis).
