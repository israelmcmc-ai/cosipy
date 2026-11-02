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
  `Xpointings/Ypointings/Zpointings (glon,glat)` [rad], `Phi` [rad], `Chi local`, `Psi local` [rad], `Distance` [cm] (= |hit2 - hit1|, the first-to-second-hit distance = MEGAlib `FirstLeverArm()`; [verified] by matching 1270 events of a week-1 `.tra` and the FITS file by time and energy: FITS `Distance` == |hit2 - hit1| for all of them, and != `LA` for the 28% with 3+ hits),
  `Chi galactic`, `Psi galactic` [deg]. They do **not** contain the number of hits, hit positions, depths, or the true photon direction.
- **`.tra` files** (`COSI-SMEX/DC4/Data/Backgrounds/Trafile/*.extracted.filtered.tra.gz`, backgrounds only; none for the Crab or any source) contain per event:
  `ID`, `TI` (time), `GX/GZ` (pointing of the x and z axes, Galactic deg), `SQ` (number of hits), `CE` (energies), `CD` (positions of the first and
  second hit with errors), `LA` (lever arm = MEGAlib `LeverArm()` = the minimum consecutive-hop distance of the Compton sequence; NOT the FITS `Distance`: it equals |hit2 - hit1| only for 2-hit events and is used by mimrec's "Any lever arm" cut), and one `CH n x y z E ...` line per hit (`CH 0` = first hit,
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
- **Mid-plane hits** (middle plane of each detector layer = the 4 spikes in the z histogram): 1.41% of all hits (2.23% of first hits, 0.79% of second hits) have z exactly equal to the mid-plane of their layer (13.007, 15.573, 18.139, 20.705 cm; one value per layer, ~10x more frequent than any other z). Interpreted as hits with no/failed depth information (Feature 4 in 1.4); the top and bottom surfaces of each layer show a weaker pile-up (1.4).
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

**Feature 4. A very bright band along the equator (psi = 90 deg) in all maps, even though the full instrument has detectors stacked above and below.** [expert] A depth-reconstruction problem **in each detector (each of the 4 z layers) individually**: the reconstructed depth is biased toward the
top and bottom surfaces of the detector, and some hits are reconstructed exactly at the middle plane of the detector (the 4 spikes inside the 4 z blocks of the z histogram in `figures_hit_positions/hit_positions_week1_all_components.png`).
Because it is not always both hits that are on the middle plane, the effect is a broad band rather than a narrow spike, and one mis-reconstructed hit is enough to bias psichi toward the equator (dz = 0).
[verified] the depth anomalies (`figures_hit_positions/hit_z_profile_by_layer.png`, week 1, all components; density relative to the median of the layer, spike excluded):
- *Middle plane*: one exact z value per layer (13.0067, 15.5727, 18.1387, 20.7047 cm) with 1.4% of all hits (all hits 0.9 / 0.9 / 1.0 / 1.55% of the hits of layers 1-4 in the Galactic-diffuse sample; 2.2% of the first hits, 0.8% of the second hits overall); the spike is 1.8-2.6x the local density for all hits, 2.2-3.8x for first hits, 1.5x for second hits (0.02 cm bins).
- *Top and bottom surfaces*: a weaker, broader excess just inside the surfaces, with a sharp drop at the surface itself (outermost bin only partly filled). Density 0.03-0.11 cm inside the surface relative to 0.30-0.50 cm inside: top surface 1.08 / 1.13 / 1.16 / 0.90 (layers 1-4, all hits; first hits 1.07 / 1.15 / 1.22 / 0.94),
  bottom surface 0.96 / 1.02 / 1.00 / 0.95 (first hits 1.12 / 1.05 / 0.98 / 0.87), i.e. at most +10-20% and not the same in every layer (the real depth profile is not flat: absorption, and the top layer sees more photons), so by itself this is a modest signature. There are no repeated exact z values at the surfaces (unlike the middle plane).
[verified, partial] the mechanism for the equator: for events with both of the first two hits on the middle plane (0.01% of the events) 80% lie within 1 deg of the equator, as expected. For events with **one** middle-plane hit it depends on the topology (Galactic-diffuse week, pairs in the same layer):
with lateral separation >= 3 cm and Distance >= 1 cm, |lat| < 10 deg is 93.0% (398 events) vs 81.2% (26,928 events) without a middle-plane hit and rms(dz) is 0.43 vs 0.58 cm, as the mechanism predicts (the middle-plane z pins dz to < ~0.75 cm);
integrated over all same-layer pairs with Distance >= 1 cm the change is not significant (|lat| < 5 deg: 22.5% vs 24.5%), and for all distances the flagged events are *less* equatorial (8.3% vs 18.3%, dominated by very short hops); cross-layer pairs are not equatorial either way (|lat| < 10 deg: 0.6% vs 1.3%).
Only 2-3% of the events have one middle-plane hit among the first two (3.0% for no Distance cut, 1.8% for Distance >= 1 cm), so the middle-plane hits alone cannot be the whole band: [open] how much of the band is due to the top/bottom pile-up and how much to the geometry below.
[verified, part of the explanation] even with perfect depth the band is partly geometric: pairs in the same 1.5 cm thick layer have |lat| < atan(1.5 cm / lateral separation) (about 17 deg at 5 cm), and 68% of the events (45% for Distance >= 1 cm) have both hits in the same layer.
In the Galactic-diffuse week 11.1% of the events (Distance >= 1 cm) lie within |lat| < 5 deg vs 8.7% for an isotropic distribution; same-layer pairs with lateral separation >= 3 cm are 55% within 5 deg.
Same-layer pairs also show a mild excess at dz ~ 0 relative to independent uniform depths (P(|dz| < 0.05 cm) = 8.0% vs 6.6% for Distance >= 1 cm; 12.8% vs 6.6% with no cut, dominated by very short hops).
A way to settle the rest: compare reconstructed and true depth with simulated truth, or look at how the equatorial excess changes when hits within the middle-plane / surface depth bins are excluded in the event selection.

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

## With a minimum Distance (= |hit2 - hit1|) cut
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

## Sensitivity vs event-quality cuts with the response simulations (`figures_quality/`)
**Distance = first lever arm = |hit2 - hit1|** (= the FITS `Distance` and MEGAlib `FirstLeverArm()`, the quantity of the response creator's `dmin`/`dmax`). It is NOT the `.tra` `LA` (minimum consecutive hop, mimrec "Any lever arm"): an earlier version of this study used LA by mistake; it is kept for reference in `figures_quality_LA/` (`DISTVAR=LA` selects it in `bkg_quality.py` / `sensitivity_quality.py`).
Signal: 10 `COSI-SMEX/DC3/Data/Responses/TrasFiltered/ResponseContinuum.inc*.id1.filtered.tra.gz` files (3.85M Compton events; far-field isotropic, flat 10-10000 keV in E_true, re-weighted by E_true^-2.1; true direction from the `OI` line, ARM peaks at 0 only with d = the OI direction, i.e. the propagation direction).
Background: all 11 components of the week-1 `.tra` files (SAA-filtered), scanned with `tra_events.awk` + `bkg_quality.py` (`run_bkg_quality.sh`, about 1 h; the files hold the whole 3 months so the scan reads everything even though a fraction of the events would give enough statistics). Background in the ARM window is the isotropic-source average B(w) = sin(w) * sum(sin phi).
FoM = max over the half-width w of the |ARM| window of S/sqrt(B), 6 log energy bins 200-5000 keV, percentage change vs no cut (window re-optimised for every cut), errors = jackknife over the 10 signal files (statistical only). Scripts: `tra_events.awk`, `tra_quality.py`, `resp_signal.py`/`run_signal.sh`, `bkg_quality.py`/`run_bkg_quality.sh`, `sensitivity_quality.py`, `plot_quality_study.py`, `plot_quality_arm_opt.py`. All tables: `figures_quality/quality_summary.md`, `quality_distance_arm_optimised.md`.

Results (within the model; the sky-averaged sin(phi) background is an approximation):
- **Validation**: the optimal ARM window gains +13, +18, +32, +45, +26, -1% (Crab study: +16, +25, +43, +56, +38, +8%), half-widths 13.5, 7, 4.5, 3.5, 4, 10.5 deg (Crab: 10, 6, 4, 3, 3, 8): same trend, somewhat smaller gains.
- **Minimum Distance**: always hurts. Over all bins in quadrature -5% (0.5 cm), -15% (1 cm), -30% (2 cm), -64% (5 cm), -81% (8 cm); at 1 cm from -21% (200-342 keV) to -3% (above 1.7 MeV), i.e. it hurts least at high energy; at 0.5 cm the 2.9-5 MeV bin is +0.2%. Consistent with the Crab study (-10 to -25% at 1 cm). The cut removes 29% of the background at 1 cm.
- **ARM with the window re-optimised for each Distance cut** (`figures_quality/quality_distance_arm_optimised.png`): the optimal half-width does shrink with the cut (D >= 2 cm: 8, 4, 3, 2, 2, 6 deg vs 13.5, 7, 4.5, 3.5, 4, 10.5 deg without; D >= 4 cm: 8, 3.5, 2.5, 1.5, 1.5, 5 deg), so the background inside the window falls faster than the signal (D >= 1 cm: background 0.41-0.58 vs signal 0.50-0.74 of the no-cut values, versus 0.58-0.76 for the background in the unchanged window). That is not enough to compensate for the lost signal: S/sqrt(B) still falls, so re-optimising the window recovers only part of the loss.
- **Number of hits**: no cut is best. SQ <= 3: -9%, SQ <= 4: -2%, SQ = 2 only: -27% (-14% at 200-342 keV to -56% above 1.7 MeV); events with 3+ hits are mostly signal at high energy.
- **First-hit depth**: dropping the bottom layer (layers 2-4) gains +1.0% over all bins (+3.2% at 342-585 keV, but -4% above 1.7 MeV); keeping only upper layers or a single layer is strongly negative. Excluding only middle-plane hits is neutral (-0.4%); excluding middle-plane and surface hits costs -6.5% (first hit) / -12% (either hit).
- **Combined search** (1200 cuts: Distance x SQ x layer set x depth flags): the best common cut is +1.0% (no Distance cut, drop the bottom layer); per-bin best at most +3.2% (342-585 keV); at 2.9-5 MeV the best is D >= 0.5 cm with +0.2%. Variants: +1.7% (gamma 3), -0.4% (gamma 1.5), +4.1% (sources within 60 deg of +z); D >= 1 cm gives -11% to -19% for these.
- **ARM width** (68% containment of events within 30 deg, all quality-cut events): 12.5, 12, 10.5, 9, 10.5, 12.5 deg without a cut; D >= 4 cm: 12, 11, 10, 8, 8.5, 9 deg (modest narrowing, most at high energy); SQ cuts do not narrow it; a top-layer first hit narrows the lowest 4 bins (11.5, 10.5, 9, 8 deg) at the cost of 62% of the signal.
- **Background composition**: albedo photons dominate at high energy (67% at 2.9-5 MeV), cosmic photons at low (48% at 342-585 keV), primary protons and alphas matter near 1-1.7 MeV (22% + 12%). The cuts barely change the composition; a D >= 1 cm cut keeps 55-76% of each component.
- **Bottom line**: simple per-event quality cuts do not improve S/sqrt(B) beyond the optimal ARM window; the only small gains are from dropping bottom-layer first hits (+1%) at low energy.
Caveats: the hit-level cuts rely on the response simulation matching the mock-data geometry; background statistics are one week; the sequence reconstruction is taken from the files (not re-run).

**Keeping events with at least n hits, and the best cut that removes >= 90% of the events** (`plot_quality_keep.py`, `figures_quality/keep_*.png`, `keep_summary.md`; Distance = first lever arm; "removed" = fraction of the week-1 background events in 200-5000 keV, which dominate the event count).
- SQ >= n only: SQ >= 3 removes 58% of the background and costs -30% in S/sqrt(B) (-48% at 200-342 keV, -6 to -12% above 585 keV); SQ >= 4: 86% removed, -57%; SQ >= 5: 96% removed, -77%. High-multiplicity events are mostly signal-poor at low energy and only mildly so above 1 MeV.
- Best combination removing >= 90% (1200 -> 5000+ combinations of Distance x hit range x first-hit layer x depth flags, window re-optimised): Distance >= 1 cm, SQ >= 3, **first hit in the top layer** (z 19.7-21.45 cm), no depth flag: 91% of the background removed, 13% of the signal kept, S/sqrt(B) -41.1 +- 0.4% (-59% at 200-342 keV, -24 to -33% in the other bins). Depth flags and an upper SQ limit do not help. Per-energy-bin optima are similar (D >= 0.5-2 cm, SQ >= 3, top layer; -57% at 200-342 keV, -24 to -34% above 342 keV).
- Best achievable S/sqrt(B) change versus the background removed: 50% -> -9%, 70% -> -20%, 80% -> -34%, 90% -> -41%, 95% -> -53%.

**Classifier on the quality variables, 90% background removal** (`plot_quality_classifier.py`, `figures_quality/classifier_summary.md`, `classifier_tradeoff.png`).
Per energy bin, a Poisson regression models the signal-to-background ratio of every (Distance bin x SQ x depth category of hit 1 x of hit 2) cell (one-hot main effects plus pairwise interactions, ridge 1; signal = simulated events inside the baseline-optimal ARM window, background = sum sin(phi)); cells are ranked by the predicted ratio and kept until 90% of that bin's background events are removed, and the ARM window is re-optimised afterwards.
- S/sqrt(B) changes by -36.8 +- 0.5% (all bins; -54, -29, -19, -24, -25, -25% from the lowest to the highest bin); trained on 5 signal files and evaluated on the other 5: -37.5% (no sign of overfitting). The best rectangular cut was -41.1%, so the classifier gains only about 4 points.
- The classifier keeps mostly top-layer first hits (90% of the kept signal), SQ >= 3 and Distance 1-8 cm, but the kept signal and the kept background have almost identical distributions in every variable: after the first-hit layer, the quality variables carry very little extra signal-versus-background information.
- Trade-off (all bins): 50% removed -8%, 70% -17%, 80% -26%, 90% -37%, 95% -49%, 98% -63% (rectangular cuts: -9, -20, -34, -41, -53% at 50, 70, 80, 90, 95%).

**What improves or worsens the ARM** (`arm_vs_characteristics.py`, `figures_quality/arm_vs_characteristics.{md,png}`; response simulation, within each energy bin). Core width w50 = half-width containing 50% of the events with |ARM| < 30 deg, relative to the all-event value, averaged over the 6 energy bins (< 1 better):
- Distance: <0.5 cm 1.63, 0.5-1 1.25, 1-2 1.01, 2-4 0.81, 4-8 **0.73**, >8 cm 0.92 (best at 4-8 cm; the very long hops get worse again, tails 1.11).
- Scatter angle phi: <30 deg 0.64, 30-60 0.74, 60-90 1.12, 90-120 1.18, >120 1.35 (but phi < 30 deg has the largest tails, 1.46).
- Number of hits: 2 hits 1.29, 3 0.95, 4 0.76, 5+ 0.68 (confounded with Distance and energy; tails also fall: 1.18, 0.90, 0.81, 0.80).
- Second hit at a middle plane 1.56, first hit at a middle plane 1.34 (1% of the events); surface hits are neutral (0.99-1.03).
- First-hit layer: bottom 1.28, 2 1.11, 3 0.98, top 0.90; second-hit layer: top 1.12, bottom 1.11, middle layers 0.91-0.93.
- Hit 2 above hit 1 (scattered upwards) 1.19 vs downwards 0.87.
- True source direction (not observable): from below (theta > 90) 1.39, from above 0.89-0.92.
Many of these are correlated (long Distance, many hits and small phi go together), so they are not independent effects.

**Scatter angle phi: sensitivity vs removed background** (`bkg_phi.py`/`run_bkg_phi.sh`, `plot_phi_study.py`, `figures_quality/phi_*.png`, `phi_summary.md`). Same signal/background setup, but the background inside the |ARM| < w window uses the exact isotropic-source sphere fraction (cos(max(phi-w,0)) - cos(min(phi+w,pi)))/2 per event, valid for small phi too; cells are (phi 2.5 deg, Distance bin, SQ, first-hit layer); the classifier is the Poisson ratio regression described above with phi features (10 deg groups) and interactions. All numbers: S/sqrt(B) change [%], all bins in quadrature, window re-optimised, every energy bin removes the quoted fraction of its background events.
- Simple cuts on phi alone always hurt: phi < 30 deg removes 81% of the background (and 82% of the signal) for -60%; phi < 60 deg 54% for -25%; phi > 60 deg 46% for -27%; phi > 90 deg 69% for -44%; phi > 15 deg removes 9% for -0.7%. Small angles give the sharpest ARM but the signal and background phi distributions are similar, so neither side of a phi cut helps (`phi_profile.png`).
- Best cuts at a given removal (all bins): 50% -> phi only -19%, no phi (Distance, hits, first layer) -8%, with phi -7%; 70% -> -29 / -20 / -17%; 80% -> -39 / -28 / -24%; 90% -> -54 / -43 / -34%; 95% -> -66 / -56 / -44%; 98% -> -80 / -68 / -57%.
- phi alone is a poor selector (-54% at 90%) but adds about 10 points when combined with Distance, number of hits and first layer (-43% -> -34% at 90%, -56% -> -44% at 95%); held-out check at 90%: -34.2% (trained on 5 files, tested on the other 5) vs -33.7% in-sample. The classifier here uses only the first-hit layer (not the second hit), so it is not directly comparable with the -36.8% classifier above.

**Most background removed for a limited loss of sensitivity (all-bin S/sqrt(B) loss <= 20%)** (`plot_phi_study.py`, `figures_quality/phi_constraint_summary.md`): ranked-cell classifier with phi + Distance + number of hits + first-hit layer: **75% of the background removed**, 31% of the signal kept (per bin -31, -13, -9, -12, -9, -11%); the same without phi: 69%; phi bins alone: 53%. Best explicit rectangular cuts: Distance >= 1 cm, SQ <= 5, first hit not in the bottom layer, phi < 90 deg: 66% removed (-18.9%); or top-layer first hit, phi >= 20 deg, SQ <= 5: 66% (-18.7%). The loss is concentrated in the lowest energy bin for the classifier (-31%).

**Dropping first hits in the bottom layers** (`layers_gain.py`, Distance = first lever arm): removing only the bottom layer gains +1.0% (all bins); removing the bottom two layers alone removes 40% of the background and 35% of the signal for -2.3% (+1.6% at 342-585 keV, -7.7 to -13% above 1 MeV). The benefit is in reaching large reductions cheaply: best S/sqrt(B) change at >= 50% / 70% / 90% background removed, searching Distance x SQ range x first-hit layer set: all layers -30 / -36 / -59%; without the bottom 2 layers -15 / -26 / -49%; top layer only -17 / -22 / -41%.

**Keeping only events whose second hit is below the first (hit ordering)** (`bkg_order.py`/`run_bkg_order.sh`, `plot_order_study.py`, `figures_quality/order_summary.md`, `order_tradeoff.png`). The ordering is almost uninformative on its own, but it removes events cheaply: 57-59% of the signal and 53-57% of the background have the second hit below the first at 200-585 keV, 53% vs 47-51% above 1 MeV; cosmic photons and Galactic diffuse 57-59%, albedo and secondaries 44-49%.
- Alone: removes 49% of the background and keeps 55% of the signal for -12.9% in S/sqrt(B) (-8.5% to -16% per bin). For comparison Distance >= 1 cm removes 29% for -15%, SQ >= 3 removes 47% for -30%.
- On top of other cuts: first hit in the upper two layers 45% -> 65% removed (-2.3% -> -13.3%); Distance >= 1 cm 29% -> 64% (-15% -> -21%); Distance >= 1 cm, SQ <= 5, upper two layers, phi < 90 deg 74% -> 82% removed (-19.2% -> -17.5%, per bin it helps 342-1000 keV and costs at 200-342 keV).
- Most background removed for an all-bin loss of at most 10 / 20 / 30%: rectangular cuts 40 / 74 / 85% without the ordering cut and 68 / 81 / 88% with it (Distance >= 1 cm, SQ <= 5, first hit in the upper two layers, phi < 90 deg, second hit below the first at 20%: 81% removed, -17.5%); classifier on (phi, Distance, SQ, both hit layers) 54 / 71 / 82% without and 59 / 76 / 88% with the ordering feature; held-out check at 75% removal: -19.5% (vs -18.9% in-sample).

**Splitting the events by Distance bin and adding the S/sqrt(B) in quadrature** (`order_common.py`, `plot_split_study.py`, `figures_quality/split_summary.md`, `split_distance_vs_energy.png`). Each subset (10 Distance bins) gets its own optimal |ARM| window; subsets are independent so their S/sqrt(B) add in quadrature; change relative to the no-cut single-window value, per energy bin (200-342 ... 2924-5000 keV; half-sample differences <= 0.9 points):
- No cut, split only: +2.0, +3.9, +4.7, +6.2, +9.1, +9.1% (all bins +3.4%) vs 0 for one window.
- Distance >= 1 cm, SQ <= 5, upper two layers, phi < 90 deg, second hit below the first (81% of the background removed): single window -28.4, -7.2, -8.9, -15.7, -18.7, -19.2% (all -17.5%); split -27.8, -5.6, -5.9, -12.7, -13.7, -14.1% (all -15.9%).
- The same cuts without the Distance requirement (74% removed; keep all Distance bins and split): single window -18.6, -3.0, -7.8, -15.6, -20.1, -19.3% (all -11.9%); split **-15.7, +2.2, -0.8, -8.6, -10.4, -9.8% (all -7.2%)**: dropping the Distance cut and using the split is much cheaper than cutting Distance for the same removal.
- Contribution of each Distance bin for the 81% cut: Distance 1-3 cm carries 20-29% of the S/sqrt(B)^2 at 200-585 keV, 4-8 cm 24-37% above 585 keV; bins above 12 cm carry < 5%.

**ARM versus Distance with the other cuts** (`plot_arm_vs_distance.py`, `figures_quality/arm_vs_distance_{w50,wopt,tail}.png`, `arm_vs_distance.md`). Distance bins used in the split study and here: edges 0, 0.5, 1, 2, 3, 4, 6, 8, 12, 20 cm and > 20 cm (10 bins). The core width w50 (50% of the events within 30 deg) is smallest at 3-8 cm in every energy bin (e.g. 3.0 deg at 1-1.7 MeV vs 7.5 deg below 0.5 cm), rises again above 8-12 cm, and with SQ <= 5, upper-two-layers, phi < 90 deg and second hit below the first it falls by about 30% (1000-1710 keV, 3-8 cm: 3.0 -> 2.0 deg). Bins above 12 cm have few events (noisy).

**Overlap of the Distance < 1 cm cut with the other cuts** (`plot_distance_overlap.py`, `figures_quality/distance_overlap.md`). Distance < 1 cm holds 34% of the background (43% at 200-342 keV, 24-27% above 585 keV) and 37% of the signal. Share of the Distance < 1 cm background events removed by each other cut (vs the share of Distance >= 1 cm events): phi < 90 deg 43% (33%; 47 vs 37% below 585 keV), second hit below first 48% (46%), upper two layers 36% (42%), SQ <= 5 1% (1%); all four 78%. Only phi < 90 deg is correlated with short hops; the others remove short and long hops alike, so the Distance cut after them is roughly "34% of what is left": extra removal by Distance >= 1 cm as cuts are added cumulatively: 34 points (alone), 34 (+SQ), 22 (+layers), 12 (+phi), 7 (+ordering) of all background events, i.e. still 28-36% of the events that remain (the other cuts already removed 74%).

**Source transiting over the instrument zenith: loss from using only |theta| < 45 deg** (`plot_zenith_loss.py`, `figures_quality/zenith_loss.md`, `zenith_transit.png`). Method: response simulation (far-field isotropic beam, so effective area ~ detected events / (sin(theta) dtheta)), 5 deg zenith bins, ARM window optimised per zenith and energy bin, constant angular speed (equal time per theta bin), SNR^2 = sum over zenith bins of A(theta,w)^2/B(w) (quadrature). Loss = 1 - SNR(theta < 45 deg)/SNR(all): no cut, vs theta < 90 deg: 7, 7, 9, 10, 12, 15% per energy bin (200-342 ... 2924-5000 keV), 7% over all energies; vs all angles (< 180 deg): 7, 8, 10, 12, 16, 22%. With the combined cut (Distance >= 1 cm, SQ <= 5, upper two layers, phi < 90 deg, second hit below the first) the loss is 5, 6, 7, 8, 9, 12% (it removes sources below the instrument, so the reference does not depend on theta_max). SNR^2 of the no-cut case: 87 / 85 / 82 / 77 / 70 / 61% comes from theta < 45 deg; the effective area drops steeply above ~45 deg (to ~10-30% beyond 75 deg).

**Psichi cut consistent with a source within 45 deg of the SC zenith** (`bkg_psichi.py`/`run_bkg_psichi.sh`, `plot_psichi_study.py`, `figures_quality/psichi_summary.md`, `psichi_tradeoff.png`). The psichi direction p = -(scattered direction) has polar angle psi from +z (psi = arccos(-(z2-z1)/|hit2-hit1|)); the source lies on the ring at angle phi from p, whose smallest zenith angle is |psi - phi|, so an event is consistent with a source within 45 deg of the zenith if |psi - phi| < 45 deg + margin ([verified] on the response simulation: 95% of the events with |ARM| < 10 deg have the true zenith angle in the allowed band). Sensitivity is for a zenith transit using only theta < 45 deg (signal = simulation with true zenith angle < 45 deg, quadrature over 5 deg zenith bins), all-bin S/sqrt(B) change vs no cut:
- Alone (no other cut): margin 0 deg removes 39% of the background and keeps 79% of the signal for **+21%** (+16% at 200-342 keV to +41% at 2.9-5 MeV); margin 5 / 10 / 20 deg: +17 / +14 / +9% (34 / 30 / 23% removed). The hit order alone is -8.5% (46% removed).
- Base cuts (Distance >= 1 cm, SQ <= 5, upper two layers, phi < 90 deg): -13.7% at 74% removed; + hit order -5.3% (81%); + psichi margin 0 / 5 / 10 deg -1.0 / -4.0 / -6.3% (82 / 80 / 79%); + psichi 10 deg + hit order -2.6% (82%). Without the Distance requirement: + psichi margin 0 deg **+6.4% at 73% removed**, + hit order +2.5% (74%).
- Most background removed for an all-bin loss of at most 10 / 20 / 30%: no extra cut 62 / 77 / 90%; hit order 81 / 87 / 91%; psichi margin 0 deg 84 (D >= 2 cm, -9.4%) / 88 (-15.0%) / 92 (-22.4%)%; psichi 10 deg + hit order 82 (-2.6%) / 88 / 92%.
- The gain is at >= 342 keV (up to +40% at 2.9-5 MeV); the 200-342 keV bin always loses (-15 to -27% with the base cuts), so the cuts could be applied only above ~340 keV.
- A source outside the 45 deg cap loses sensitivity with this cut, as intended (it keeps 61% of the events from sources beyond 45 deg at margin 0 deg, 73% at 10 deg).

**Individual cuts vs S/sqrt(B), background rejection and ARM** (`plot_individual_cuts.py`, `figures_quality/individual_cuts.png`, `individual_cuts.md`; one cut at a time, transit theta < 45 deg, each row has its own no-cut baseline; w50 = half-width of 50% of the signal events with |ARM| < 30 deg, w_opt = optimal window for the zenith-pooled signal). Main points: minimum Distance monotonically costs S/sqrt(B) (-5% at 0.5 cm to -80% at 8 cm) while rejecting 9-17% to 91-97% of the background and slightly narrowing w50 / w_opt (e.g. 1-1.7 MeV: 3.5 -> 2.5 deg); SQ <= n and SQ >= n both cost sensitivity (SQ <= 3: -7%); first-hit layer "not bottom" gains +3.5% (14-25% rejected), top only -16%; phi < x costs -12% at 90 deg with a clear ARM narrowing (w50 3.5 -> 2.5 deg at 1-1.7 MeV) but equal signal loss; phi > x only hurts and widens the ARM; the zenith cap theta0 of the psichi cut gives +96% at 15 deg, +42% at 30, +21% at 45, +11% at 60 and +3% at 90 deg (74-86% rejected at 15 deg), and the psichi margin gives +21% (margin 0 deg) to +3% (margin 45 deg); the psichi cut does not change the ARM resolution (w50 and w_opt are flat vs margin), it only rejects background (in the zenith-cap row the ARM changes because the signal sample itself is restricted to theta < theta0).

**CORRECTION: the psichi/zenith-cap gains above (+21% etc., `psichi_summary.md`, the zenith-cap and psichi-margin rows of `individual_cuts.png`, `zenith_cap_total.md`) are an artifact of the isotropic-source background approximation.** B(w) = sum over events of the sphere fraction inside the window assumes the source is anywhere on the sphere; for a source at a known zenith angle theta_s an event only enters the window if |angle(source, psichi direction) - phi| < w, so events whose ring misses the cap were never background for that source, and removing them cannot improve S/sqrt(B). `plot_psichi_pos.py` (`psichi_pos_common.py`, `figures_quality/psichi_pos_summary.md`, `zenith_cap_pos.png`) evaluates the background at the source zenith angle from the (phi, psi) distribution (azimuthally symmetric: P(|alpha - phi| < w) = (arccos u1 - arccos u2)/pi, u = (cos bound - cos theta_s cos psi)/(sin theta_s sin psi); the sphere average reproduces the isotropic factor to 0.1%). Results (transit theta < 45 deg, all-bin SNR change vs no cut):
- Zenith-cap cut |psi - phi| < theta0 on every event: 0.0% for theta0 >= 55 deg, -0.1% at 45 deg (47% of the background events removed), then falling: -4.5% at 30, -12.5% at 20, -19% at 15, -27% at 10, -40% at 5 deg (for the whole transit theta < 90 deg: -2.0% at 45 deg, -8% at 30 deg). So the cap cut is a free event reduction (about 40-47% of the background events at no cost) but gives no sensitivity gain, and the curve is flat then falling, not rising.
- Base cuts (Distance >= 1 cm, SQ <= 5, upper two layers, phi < 90 deg): -23.4% at 74% removed; + zenith cap 45 deg: -23.5% at 83% removed; + hit order -25.1% at 82%; + both -25.1% at 85%. The hit order alone costs -13.9% at 49% removed.
- The background at the source position differs from the sphere average by up to +34% (200-342 keV, near the zenith) to -9% (1.7-2.9 MeV); the loss from using only theta < 45 deg is unchanged (8%).
The other studies that used the all-sky average background (Distance, SQ, layers, phi, hit order, split-by-Distance, classifier) assumed a source direction uniformly distributed over the sphere, as agreed; the transit-based numbers (individual cut rows with theta < 45 deg, zenith loss) carry the isotropic-B approximation, which is a small effect for cuts that do not depend on psichi.
