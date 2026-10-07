# DC4 exploratory analysis: findings and gotchas for background estimation

This directory holds the scripts, figures and notes from an exploration of the COSI Data Challenge 4 (DC4) mock dataset and its background
components (week 1 of the mock dataset, the 3-month component files, and the `.tra` hit-level files). It was written for whoever implements
a **background estimation technique** next. Section 1 is the part to read first (data facts, conventions, detector geometry, the four unusual
features of the psichi maps and what is and is not understood about them, other findings, tooling gotchas). The rest of the file (below the
line "Detailed log") is the chronological log of every analysis with commands and figure locations.

Status tags used below: **[verified]** = checked with numbers in this analysis; **[expert]** = explanation given by a COSI instrument
expert (the user) that is consistent with the data but not independently proven here; **[open]** = not understood / not confirmed.

## 1. Findings and gotchas

### 1.1 Data products and what they contain
- Storage: Wasabi bucket `cosi-pipeline-public` (endpoint `https://s3.us-west-1.wasabisys.com`; the access keys are public, in the
  `cosi-data-challenges/data-products/README.md`). 3 months of data: TimeTags 1835487300 to 1843467255 s (GPS), week 1 = 1835487300 to 1836092100.
- **FITS event files** (mock dataset weekly files, every component, the Crab) have exactly 11 columns: `Energies` [keV], `TimeTags` [s],
  `Xpointings/Ypointings/Zpointings (glon,glat)` [rad], `Phi` [rad], `Chi local`, `Psi local` [rad], `Distance` [cm] (first to second hit),
  `Chi galactic`, `Psi galactic` [deg]. They do **not** contain the number of hits, hit positions, depths, or the true photon direction.
- **`.tra` files** (`COSI-SMEX/DC4/Data/Backgrounds/Trafile/*.extracted.filtered.tra.gz`, backgrounds only; none for the Crab or any source) contain per event:
  `ID`, `TI` (time), `GX/GZ` (pointing of the x and z axes, Galactic deg), `SQ` (number of hits), `CE` (energies), `CD` (positions of the first and
  second hit with errors), `LA` (lever arm = first-to-second-hit distance = FITS `Distance`), and one `CH n x y z E ...` line per hit (`CH 0` = first hit,
  `CH 1` = second hit of the Compton sequence). Positions are in cm in the payload frame (see 1.3). Average 2.48 hits/event. **No true direction** is stored; there are no
  `.sim` files for cosmic photons or Galactic diffuse (DC4 has `.sim` only for albedo photons and primary protons), and diffuse sources have no single direction anyway.
- **Each `.tra` is a concatenation of increments (about 120k events each) and every increment spans the full 3 months**, so the events of one week are
  scattered through the whole file. A week is selected by `TI`, which requires reading the whole file (about 1 minute per GB compressed with a `gunzip | awk` pipe; 14.5 GB for cosmic photons).
- `.tra` and FITS samples are **not event-for-event matched** and their counts differ: week-1 `.tra` events: cosmic photons 8.97M (FITS 3-month 97.55M / 13 = 7.5M), albedo photons 2.90M
  (FITS 27.08M / 13 = 2.08M), primary protons 1.20M (FITS 14.48M / 13 = 1.11M). The relation (SAA / time cuts, extra selection) was not investigated. The unfiltered `SAA_Andreas` `.tra` was not used.
- Components in the mock dataset (rows of the 3-month FITS): cosmic photons 97,554,902 (DC3 file); albedo photons 27,081,364; primary protons 14,481,055; primary alphas 7,783,655; Galactic diffuse 7,171,368 (DC3
  `GalTotal_SA100_F98`); secondary positrons 6,173,396; SAA 4,958,885 (statistically reduced); albedo neutrons 1,654,775; secondary electrons 1,096,211 (DC3 file); secondary protons 650,980; primary electrons 38,994;
  primary positrons 2,959. They sum to the `Total_DC4_BG_...withSAAbck` file (168,648,544 events, 16 GB). Primary electrons/positrons are too sparse for any map.
- The 3-month component FITS files are **not time-ordered** (albedo checked); the weekly mock-dataset files are. Time-ordered orientation file: 15 s bins, columns `TimeStamp`, `XPointings`, `ZPointings`,
  `EarthZenith` (all Galactic lon/lat in degrees), `Altitude`, `LiveTime`.
- Event selection already applied in the products: 100 keV to 10 MeV (nominal range 200 keV to 5 MeV), Compton events only, SAA passages removed, Earth occultation beyond ~113 deg from zenith in the simulation.

### 1.2 Conventions used in all scripts
- **Spacecraft (SC) frame**: longitude = `Chi local`, latitude = 90 deg - `Psi local`; psi is the polar angle from the +z axis (boresight), the direction is that of the *scattered gamma ray*.
  Verified: rotating the local psichi with the per-event x, z pointings (y = z cross x) reproduces the Galactic columns to < 0.06 deg.
  The source of a photon lies on the circle of angular radius `Phi` around its psichi.
- **Earth horizon frame** (not in the files, computed): zenith = `EarthZenith` interpolated to the event time; North = celestial pole (Earth spin axis) projected on the horizon plane; azimuth N to E, altitude from the horizon.
- **Survey modes**: the z axis is tilted +22 deg (survey north) or -22 deg (survey south) from the zenith toward North, switching every ~12 h with ~8 min slews (1.1% of events); classified by |tilt| > 20 deg. Mean z axis in the horizon frame: (az 0, alt 68.1 deg) north, (az 180, alt 68.1 deg) south.
- **Roll angle** (used for the albedo split): angle, counterclockwise about SC +z, from +x to the projection of the Earth zenith onto the SC xy plane (= SC azimuth chi of the zenith). Survey north occupies |roll| > 100 deg, survey south -80 to +80 deg.
  The SC x axis azimuth in the horizon frame swings between about +-72 deg once per orbit (96 min), i.e. the roll is not constant in a survey mode.
- **ARM** = angle(true source direction, psichi) - Phi, computed for the Crab only (known position l = 184.5575, b = -5.7843); peaks at 0 (within 0.4 deg), FWHM 10.8 / 6.0 / 4.2 / 3.2 / 3.0 / 4.8 deg in the six log bins 200 keV to 5 MeV.
- HEALPix nside 32 (1.8 deg pixels, equal area, RING) for all maps; maps are smoothed with a 4 deg Gaussian unless stated. Plain longitude/latitude bins are **not** equal-area (they shrink as cos(lat)).

### 1.3 Detector geometry seen in the hit positions (week 1, 15.3M events, 37.9M hits) [verified]
- 4 layers (z) x 2 x 2 detectors = 16 detectors. Layer z ranges: 12.26-13.77, 14.7-16.3, 17.2-18.8, 19.7-21.45 cm (layer 1 = lowest z). x: -8.39 to -1.06 and 1.06 to 8.39 cm; y: -6.71 to 0.62 and 5.86 to 13.20 cm. Gaps between towers: 2.1 cm in x, 5.2 cm in y. Detector corners are chamfered (about 1 cm).
- **Hit x and y sit on a strip lattice**: pitch 0.1164 cm, 64 strips per detector per axis, ~10% of the hits lie between strips (shared / interpolated positions). A regular histogram beats against this lattice (comb); plot per-strip counts or use bins of one pitch.
- Hit share per layer (bottom to top): all hits 17 / 20 / 25 / 38%, first hits 17 / 17 / 23 / 43%; the 4 detectors of a layer agree to 1-2%; the outer 1 cm of a detector has 73-79% of the central density (all hits).
- **Mid-plane hits**: 1.41% of all hits (2.23% of first hits, 0.79% of second hits) have z exactly equal to the mid-plane of their layer (13.007, 15.573, 18.139, 20.705 cm; one value per layer, ~10x more frequent than any other z). Interpreted as hits with no/failed depth information (see 4 in 1.4).
- In the Galactic-diffuse week 68% of the events have first and second hit in the same layer (45% for Distance >= 1 cm, 16% for Distance > 8 cm).

### 1.4 The four unusual features of the psichi maps (spacecraft frame)
Figures: `figures_dist1cm/` (week 1, Distance >= 1 cm), `figures/` (no cut), `figures_dist8cm/` (Distance >= 8 cm), `figures_backgrounds_sc_dist1cm/` (per component),
`figures_dist1cm/psichi_spacecraft_vs_{latitude,longitude}_1deg.png` (1 deg projections), `figures_hit_positions/` (hit-level plots).

**Feature 1. A bunch of narrow peaks along meridians (constant chi) without a distance cut.** [expert] Discretization of the strips (see Fig. 2.30 of P. Janowski's MSc thesis,
Mainz 2025, `https://cms.zdv.uni-mainz.de/fb08-xenon-physik/wp-content/uploads/sites/107/2026/03/oberlack-group-thesis_2025_MSc_Janowski-Pascal.pdf`; I did not read the thesis).
They disappear for Distance > 1 cm. [verified] The narrow azimuth peaks of the first-to-second-hit lateral displacement (Galactic diffuse, no cut, folded to 0-45 deg) are at 14.25, 18.25, 26.75, 33.75 and 44.75 deg,
i.e. at atan(1/4), atan(1/3), atan(1/2), atan(2/3), atan(1/1) = 14.04, 18.43, 26.57, 33.69, 45 deg: small-integer strip offsets, as expected for a lattice. Short hops (< 1 cm lateral) are where the lattice matters.

**Feature 2. For Distance > 1 cm, strong dips along the x and y axes (chi = 0, 90, 180, 270 deg), around a spike exactly on the axis.** [expert] Also strip discretization: every direction whose lateral
offset is within one strip pitch in one coordinate is reconstructed exactly on the axis, producing a spike on the axis and a depletion of the small angles next to it. (The first, rejected, explanation was only the gaps between towers plus
material shielding: it is not needed.) [verified] (a) the spike occupies the single 1 deg bin starting at each axis ([0,1), [89,90)... depending on the bin edge) with roughly 1.8x the neighbouring counts and is surrounded by a V-shaped depletion of about +-5 deg;
(b) the empty gap around the axis shrinks as 1/Distance: |offset| < ~3.3 deg for Distance 1-2 cm, ~2.2 (2-3), ~1.2 (3-5), ~0.8 (5-8), ~0.3 deg (8-30 cm), and tan(gap) x Distance ~ 1.1 mm, i.e. one strip pitch (1.164 mm);
(c) the events are only partly "migrated" onto the spike: summed over all distances the 5 deg bin centred on the spike is ~17% below the global mean, the spike covers ~45% of the dips' deficit and the total is conserved to a few % only over about +-13 deg (the rest sits in slightly elevated shoulders at 5-13 deg).
**Gotcha**: the dips are wider than one HEALPix nside-32 pixel at small Distance (+-3.3 deg for 1-2 cm), so templates for a background model must be built with the same quality cuts as the data and cannot be smooth near the axes. Figures: `figures_dist1cm/sc_longitude_spike_{fold90,vs_distance}.png`.

**Feature 3. For Distance > 8 cm the axis dips turn into broad maxima along x and y (not along the diagonals, where the most material is).** [expert, verified] 8 cm is about the size of one detector, so most first-second-hit pairs must be in different towers:
89% of the events (Distance > 8 cm) have the two hits in different detectors (same / x-neighbour / y-neighbour / diagonal: 11 / 32 / 34 / 23%) against 7% without the cut. Neighbouring towers along x and y are closer than diagonal ones, the number of events falls steeply with distance,
and the cut sits at the steep part, so the axis neighbours (separations 8-10 cm) dominate; diagonal pairs are present but subdominant (their azimuth peaks at about +-50 and +-130 deg because the gaps differ in x and y). The first and second hits concentrate on the detector edges facing the gaps
(34.9% of the first hits within 1.5 cm of the y-gap edges vs 22.0% without the cut, area 21%; 25.2% vs 20.5% for the x-gap edges); the (dx, dy) displacement map is a ring of radius ~8.5-9 cm (the cut) with hot spots on the axes. A pure geometry Monte Carlo (uniform hits in the 2 x 2 layout,
weight exp(-rho/lambda)/rho^2, rho > 8 cm) reproduces the four axis maxima and the diagonal minima, but over-weights x-neighbours (50-56% vs 32%) and under-weights diagonal and same-detector pairs because it ignores the vertical separation (the data have dz peaks at multiples of the 2.56 cm layer spacing). In the limit of far-apart
towers and a cut above the tower diagonal only the neighbour directions would remain; with flush towers the first intuition (maximum along the diagonals) would hold. Figures: `figures_hit_positions/dist8cm/`.
**Gotcha**: the Distance cut therefore acts as a *geometry* selection: it changes the psichi pattern through the tower layout, not only through angular resolution.

**Feature 4. A very bright band along the equator (psi = 90 deg) in all maps, even though the full instrument has detectors stacked above and below.** [expert] a depth-reconstruction problem: a bias of the reconstructed depth toward the top and bottom of the detector, and hits reconstructed exactly at the mid-plane
of the detector; because it is not always both hits that are mid-plane, the effect is a broad band rather than a narrow spike, and one mis-reconstructed hit is enough to bias psichi to the equator.
[verified, partial] mid-plane hits exist (1.4% of hits, 2.2% of first hits; sec. 1.3); events with *both* of the first two hits at the mid-plane (0.01%) lie 80% within 1 deg of the equator, as expected. [open] I could not confirm that *one* mid-plane hit pushes the event to the equator: in the Galactic week the events with exactly one mid-plane hit (3.0% of the events)
are less equatorial than the others (|lat| < 5 deg: 6.7% vs 12.4%; with Distance >= 1 cm 10.7% vs 11.1%). The layer-edge bias is also weak in the aggregate z distribution (density in the outer 0.1 cm of a layer / inner reference: 0.82-1.09), but the true depth distribution is not uniform (absorption), so this does not exclude a reconstruction bias.
Same-layer pairs show a mild excess at dz ~ 0 relative to independent uniform depths (P(|dz| < 0.05 cm) = 8.0% vs 6.6% for Distance >= 1 cm; 12.8% vs 6.6% with no cut, dominated by very short hops).
[verified, part of the explanation] even with perfect depth the band is partly geometric: pairs in the same 1.5 cm thick layer have |lat| < atan(1.5 cm / lateral separation) (about 17 deg at 5 cm), and 68% of the events (45% for Distance >= 1 cm) have both hits in the same layer.
In the Galactic-diffuse week, 11.1% of the events (Distance >= 1 cm) lie within |lat| < 5 deg vs 8.7% for an isotropic distribution. A way to settle this: match `.tra` events to simulated truth (or to depth-resolved single-site data) and compare reconstructed and true depth.

### 1.5 Other findings that matter for a background model
- **Components differ strongly in the SC psichi maps** (`figures_backgrounds_sc_dist1cm/`, `psichi_maps_spacecraft_<Component>.png`, Distance >= 1 cm): the photon components (cosmic photons 48.8M events after the cut, Galactic diffuse) show a smooth +z to -z gradient with phi (small phi at +z, large phi at -z: photons from above);
  the albedo photons are the mirror image (photons from the Earth below, small phi at -z); the particle/instrumental components (primary protons and alphas, SAA, secondary positrons) are dominated by the equatorial band and chi stripes with a weak gradient. Primary electrons/positrons are noise.
- **Earth-fixed and SC-fixed parts**: in the Earth horizon frame the albedo is a pattern fixed to the nadir (ring of radius phi around the nadir) while the equatorial ring (psi = 90 deg) is SC-fixed and appears on the great circle 90 deg from the z axis (survey north and south are mirror images). The azimuthal asymmetry of the albedo emission in the SC frame rotates with the roll
  (first Fourier mode phase = roll + 180 deg to within a few degrees on average, amplitude 2-9%): the roll-dependent part is a modest modulation on top of a roll-independent SC pattern (`figures_albedo_roll_modes_dist1cm/`, 32 matrices, 10 deg roll bins).
- **Separating the Earth from the SC part** (`figures_earth_minus_sc*/`): subtracting the chi-averaged SC map leaves chi stripes in the Earth frame; subtracting the full SC template rotated with the per-sample attitude reduces the residual to the Poisson level above ~250 keV (std/sqrt(counts) 1.0-1.1) and leaves an excess at 100-250 keV (1.16-1.45).
  Caveat: the SC template is built from the same events and contains the Earth signal smeared by the roll. Cost: ~10 minutes per 3-month component because of the 15 s orientation samples.
- **Distance cut effects**: Distance >= 1 cm keeps 57% of the week-1 mock-dataset events (27% at 100-158 keV, ~75% above 1 MeV; the Crab and the background keep nearly the same fraction in each energy bin) and removes the strip comb in chi (Feature 1). Distance > 8 cm keeps 4.3-4.5% of the events overall (cosmic photons only 2.8%, primary protons 7.3%, albedo photons 7.0%, albedo neutrons 10.9%).
- **Sensitivity (S/sqrt(B), Crab vs total background, percentage changes only, `figures_sensitivity/`)**: a minimum-Distance cut essentially never improves S/sqrt(B) (-10 to -25% at 1 cm; the only gain is +1.6% at 0.5 cm in the 2.9-5 MeV bin) because signal and background keep the same fraction; an ARM window of half-width ~1-1.3 FWHM gives +8% (2.9-5 MeV) to +56% (1-1.7 MeV), +20% over all bins in quadrature;
  re-optimising the ARM window after a Distance cut does not recover the loss. The number of hits and the first-hit z could not be studied (no Crab `.tra`).
- The Crab events are all above the Earth limb (zenith angle < 112 deg); for the background the same selection (< 113 deg) was applied in the sensitivity study (76% of the events).

### 1.6 Practical gotchas (tooling)
- `cosipy` is not installed and is not used: the scripts need only numpy, scipy, astropy, healpy, matplotlib (histpy/mhealpy are not needed).
- Never decompress the big files to disk (~23 GB free): `sensitivity.stream_rows` parses the gzipped FITS table (112 bytes/row, big-endian, column order above) from a stream; `.tra` files are reduced by `gunzip | awk` to `TI idx x y z E` lines, `hit_positions.py` histograms them.
  Streaming from Wasabi runs at tens of MB/s (16 GB total-background file in ~6 min). Python-level binning of 100M events takes ~2 minutes.
- Histogram bins must not alias with the strip lattice (pitch 0.1164 cm): use per-strip counts (peaks of a 0.05 cm histogram, each strip falls in one bin). `np.histogram` and `floor((x - x0)/dx)` differ at the 1e-12 level for hits exactly on bin edges (negligible).
- In this session a safety check blocked `sh -c` / `bash -c` wrappers (e.g. `xargs sh -c ...`); put loops in script files. `pkill -f <pattern>` kills the calling shell if the pattern is in its own command line (use `pkill -f "[p]attern"` in a separate command). Container restarts kill background jobs; scripts write their `.npz` only at the end of each component.
- A background `until`/`while` waiter is killed after its time limit (1 h); the job it waits for keeps running.
- PNG sizes: the per-component figures are palette-quantized (128 colours) to keep the repository small.

### 1.7 Suggestions for the background estimation (my recommendations, not verified)
1. Build SC-frame templates per component from the data/simulation **with the same quality cuts** as the analysis (the Distance cut reshapes the maps through the tower geometry and the strip lattice; the narrow axis structures (Features 1, 2) are not smooth).
2. Model the background as an SC-fixed, roll-independent template plus an Earth-fixed (albedo/atmospheric) component rotated with the attitude, separately for the two survey modes, and weight by the exposure in each roll bin.
3. Keep chi/psi bins or smoothing consistent with the axis structure (spike width 1 deg, dip width up to ~3 deg at 1-2 cm), or avoid chi bins that straddle the axes.
4. Treat the equatorial band (Feature 4) as an instrument/reconstruction feature that must be reproduced by the simulation of the background; resolving its origin (depth reconstruction) would tell whether a cut on the depth quality removes it.
5. Component fractions in the maps are not those of the sky: cosmic photons dominate the counts but are strongly suppressed by a large Distance cut.

### 1.8 Index
- Scripts: `bin_dc4.py` (psichi histograms from FITS, streaming; frames, survey modes), `plot_dc4.py`, `plot_decouple*.py`, `plot_minus_total.py`, `plot_roll_modes.py`/`roll_modes.py`, `plot_sc_components.py`, `run_components.sh`, `sc_lonlat.py` (1 deg lon/lat projections),
  `lon_vs_dist.py`, `sensitivity.py`/`sensitivity_plots.py` (S/sqrt(B)), `narrow_phi_sc.py`, `hit_positions.py`/`run_hit_positions.sh` (`.tra` scan; `DMIN=8` for a Distance cut), `plot_hit_positions.py`, `plot_hit_xy.py`, `plot_hit_geometry.py`.
- Figure directories: `figures/` (first analysis), `figures_dist1cm/`, `figures_dist8cm/`, `figures_albedo*/`, `figures_survey_modes*/`, `figures_earth_minus_sc*/`, `figures_backgrounds_sc_dist1cm/`, `figures_albedo_roll_modes_dist1cm/`, `figures_sensitivity/`, `figures_hit_positions/` (with `dist8cm/`).

---

# Detailed log (commands and figures, in the order the analyses were made)

## First analysis: DC4 mock dataset, exploratory CDS plots

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

## Hit positions (x, y, z), week 1, all background components (`figures_hit_positions/`)
Hit-level positions exist only in the `.tra` files, so `run_hit_positions.sh` scans the `.tra` of every background component that has an
`extracted.filtered.tra` (albedo photons / neutrons, cosmic photons, Galactic diffuse (`inc1.id1`), primary alphas / electrons / positrons / protons,
secondary electrons / positrons / protons; the unfiltered `SAA_Andreas` tra is not included) and keeps the events with TI in week 1
(1835487300 to +604800 s). Each `.tra` is a concatenation of increments that each span the whole 3 months, so the whole file has to be read.
`hit_positions.py` histograms the `CH` hit lines (x, y, z of all hits, of the first hit `CH 0` and of the second hit `CH 1`; per-component
histograms in `histograms/*.npz`) and `plot_hit_positions.py` draws `hit_positions_week1_all_components.png`: 15.3M events, 37.9M hits.
x and y hits sit on a strip lattice (pitch 0.1164 cm, 64 strips per detector, 2 detectors per axis), so they are plotted as counts per strip
(`hit_strip_lattice_zoom.png` shows the lattice; ~10% of the hits lie between strips). z shows 4 detector layers (12.26-13.77, 14.7-16.3,
17.2-18.8, 19.7-21.45 cm) with spikes at the layer mid-planes (z = 13.01, 15.57, 18.13, 20.71 cm).

### x-y hit density per detector layer (`figures_hit_positions/hit_xy_density_by_layer_{all_hits,first_hit,second_hit}.png`)
Same week-1 `.tra` scan, now also filling 2-D (x, y) histograms (0.05 cm bins) per detector layer (z boundaries 12.0 / 14.2 / 16.9 / 19.3 / 22.0 cm;
`histograms/*.npz` now hold these as `xy_<kind>_L<layer>` besides the 1-D x, y, z histograms). `plot_hit_xy.py` reads the (x strip, y strip) cells
(strip lattice as above, ~10% of hits between strips not shown) and draws one panel per layer, top layer first: 2 x 2 detectors with chamfered corners.
Share of the hits per layer (top to bottom): all hits 38 / 25 / 20 / 17%, first hits 43 / 23 / 17 / 17%. The four detectors of a layer are balanced to 1-2%
(layer 1: 26.0 / 24.1 / 25.7 / 24.1% for x<0,y<3 / x>0,y<3 / x<0,y>3 / x>0,y>3); the outer 1 cm of a detector has 73-79% of the central density for all hits.

### Distance > 8 cm: hit-position density and hit-pair geometry (`figures_hit_positions/dist8cm/`)
`DMIN=8 run_hit_positions.sh outdir` repeats the week-1 `.tra` scan keeping only events whose first-to-second-hit distance (from the `CH 0`/`CH 1`
positions) is > 8 cm (686,797 of 15.32M events, 4.5%; cosmic photons 2.8%, primary protons 7.3%, albedo photons 7.0%) and also stores diagnostics of the first->second
displacement (azimuth by lateral-distance band and by detector relation, dx-dy map, dz, lateral separation, layer pairs). Plots: `plot_hit_xy.py` (xy density per layer, all /
first / second hits) and `plot_hit_geometry.py` (+ a simple geometric Monte Carlo of the 2 x 2 detector layout).
Findings: 89% of the pairs are in different detectors (same / x-neighbour / y-neighbour / diagonal: 11 / 32 / 34 / 23%; without the cut 93% are in the same detector);
the first and second hits sit near the detector edges facing the gaps (34.9% of the first hits within 1.5 cm of the y-gap edges vs 22.0% without the cut, area 21%; 25.2% vs 20.5% for the x-gap edges);
the azimuth of the lateral displacement peaks along +-x and +-y (18.6% of the events within 15 deg of the x axis and 23.5% of the y axis, vs 16.7% each for a uniform azimuth; 57.9% on the diagonals vs 66.7%) with minima on the diagonals, as the psichi lobes do.
The geometric Monte Carlo (uniform hits in one layer, weight exp(-rho/lambda)/rho^2, rho > 8 cm) reproduces the four axis maxima and the diagonal minima but over-weights x-neighbours and under-weights diagonal / same-detector pairs (it ignores the vertical separation).
