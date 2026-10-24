# Diagnostics of the histogram background in the 3 h Crab fits

Scratch scripts from the investigation of the bias of the Crab fits with
`FreeNormHistBackgroundDensity` (`../crab_hist_background.ipynb`). They are kept
for reproducibility, not as library code; paths point to the sandbox layout below.

## Data (`/home/user/dc4data`)

- `ResponseContinuum.area.relative.nonsparse_smoothing1p0.h5`: `hist_simple` IRF.
- `DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits`: orientation.
- Templates (3 months, Distance >= 1 cm): `bkgonly_template_dist1cm.h5` (background
  simulation only), `bkgcrab_template_dist1cm.h5` (background + Crab),
  `dc4_mock_bkg_template_dist1cm.h5` (mock dataset: background + all sources).
- `crabfull/crab_all.npz`: the 3-month DC4 Crab events (FITS columns).
- `truth/bkg_crabwin.npz` and `windows/bkg_<unix start>.npz`: the true background
  events of the 7 windows (`extract_windows.py`, one streaming pass of
  `Total_DC4_BG_3months...`).
- `sref/w<i>.npz`: background events of window `i` after the cuts (200 keV - 5 MeV,
  Distance >= 1 cm) and the Crab density `s` at them (power law K = 2.6e-6 at 1 MeV,
  index -2.38, Trapz folding), with `S` (`sref.py`).

## Scripts

| script | what |
|---|---|
| `crab_src.py` | source of `../crab_hist_background.ipynb` (`py2nb.py crab_src.py out.ipynb`) |
| `bkg_em.py`, `bkg_quad.py`, `bkg_integral.py` | the integral of the density vs the expected counts B (Em rate, p(phi), p(PsiChi); importance sampling) |
| `nearest_vs_linear.py` | log-likelihood of the true background, nearest vs linear interpolation per factor |
| `band_test.py` | one free background norm per Em band |
| `proj_test.py`, `where.py` | `sum(s/b)` over the true background events vs S, by region (Em, phi, elevation, rocking, time) |
| `variants.py` | `sum(s/b)/S` for smoothing variants of the templates, over the 7 windows (uses `sref/`) |
| `fixed_norm.py`, `fixed_norm_bkgcrab.py` | fits with the background norm free, fixed at the template, fixed at the true count |

## Results so far

- The density now integrates to B exactly (before: +0.16% Em, +-0.15% phi).
- Crab + true background, 7 windows: K / K_signal_only - 1 with the norm free / fixed
  at the template / fixed at the true count: bkg-only template -52% / -52% / -48%,
  all-events -41% / -91% / -34%, bkg+Crab -73% / -82% / -65%. Pulls -2 to -10.
- `sum(s/b)/S` over the true background (1 for a perfect template): 0.981 +- 0.003
  (bkg-only), 0.971 (all-events). The shift of the source counts is about
  `(sum(s/b)/S - 1) sigma_S^2`, with `sigma_S` ~ 250 counts, so the template must match
  to a few 0.1% where the source is.
- Smoothing changes `sum(s/b)/S` by < 0.003. Per-Em-band norms don't help.
- The excess is at scattered-direction elevations 0-15 deg (-219 +- 46 of ~2100 Crab
  counts) and 60-90 deg (-123 +- 39), phi 40-90 deg, and in the north rocking state
  (-287 +- 87 vs -4 +- 80 south): direction shapes that vary along the orbit?
