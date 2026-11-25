# Spectral bias of the relative-histogram IRF on DC3 data

Scripts used to locate the soft-index bias of signal-only fits with `IRFRelativeHistUnpolarized`
(`hist_nn`, `hist_simple`) on the DC3 Crab (`crab_powerlaw`) and GRB bn090424592.

They compare the **measured-energy spectrum** of the data with the one each response predicts at
the injected spectrum, independent of the unbinned folding, and fit the injected parameters from
`Em` alone. All of them take the work directory `W` as their first argument and use `W/data`,
`W/out` and `W/scripts` (a link to this folder).

```bash
W=/path/to/workdir; mkdir -p $W/out; ln -s $PWD $W/scripts
python download.py $W                 # ~45 GB peak, ~20 GB kept (both hist IRFs unzipped)
python project.py $W/data/ResponseContinuum.area.relative.nonsparse_smoothing1p0.h5 $W/out/proj_simple.npz
python project.py $W/data/relative_hist_irf_from_nf_response.h5 $W/out/proj_nn.npz
python exposure.py $W                 # dwell maps of the Crab (3 h, 3 months) and the GRB
python reduced_irf.py $W simple $W/data/ResponseContinuum.area.relative.nonsparse_smoothing1p0.h5
python reduced_irf.py $W nn $W/data/relative_hist_irf_from_nf_response.h5
# Em response matrices with the real IRF code (#31: PYTHONPATH=<irf-normalization worktree>), one at a time (memory)
for n in nn simple; do for w in crab3m crab3h grb; do python em_matrix.py $W $n ${w}_n32_occ0 $W/out/M_${n}_${w}.npz; done; done
python fit_em.py $W                   # Crab: index from Em alone, hist_nn / hist_simple / DC4 binned
python fit_grb.py $W                  # GRB: Band from Em alone
python cmp_matrix.py $W               # total area and photopeak fraction vs the DC4 binned response
python raw_disp.py $W                 # raw Epsilon distributions per Ei bin
for s in 1 2 3 4 5 6; do python nn_mc.py $W 100000 $s; done; python nn_mc_cmp.py $W   # NN response itself
```

- `reduced_irf.py` collapses Phi/Theta/Zeta to one bin each. The `Ei`/`Epsilon` behavior of the IRF
  (normalization, hats, selected fraction) is unchanged, so `em_matrix.py` evaluates the exact
  `_effective_area_cm2` with an `Em` selection of each bin, for photons on the exposure map.
- `predict_em.py`/`cmp_crab.py`/`cmp_phi.py` are an independent numpy version (bin contents as step
  functions); they agree with the real code to a few %.
- The Crab simulation has no Earth occultation (`occ0` exposure reproduces the counts to 3%).

## Results

See the PR/hand-off notes for the full numbers. In short (3 months of Crab, 5.26 M events,
injected index -2.00), fitting the index from the `Em` spectrum alone:

| Response | Data | Index |
|---|---|---|
| `hist_nn` | all events | -2.087 +- 0.001 |
| `hist_simple` | Distance >= 1 cm | -2.062 +- 0.001 |
| NN response `unpolarized_nfresponse_v1-01` (400k MC samples) | all events | -2.046 +- 0.004 |
| DC4 binned MEGAlib continuum response (`m2284`) | all events | -2.002 +- 0.001 |

- The injected truth is right: the DC3 sources were simulated over 100 keV - 10 MeV (the folding
  range too), and the MEGAlib binned response recovers -2.00 from the same events.
- `hist_nn` has the right total area (within 1% of the binned response for Ei >= 158 keV), but
  too little incomplete absorption at high Ei: the fraction of Em in the same 0.2-dex bin as Ei is
  0.25 vs 0.15 at 4-6.3 MeV. Half of it comes from the NN response itself, the rest from the
  midpoint-rule conversion (`relative_hist_irf_from_nf_response.py`), and it is already in the raw
  bin contents (Ei interpolation doesn't matter).
- `hist_simple` (Distance >= 1 cm) predicts 1.6x the observed events at Em = 100-126 keV (21% of
  the data passes the cut there, the response says 34%). This drives the GRB Band alpha to its
  upper limit (+3) in an Em-only fit, as in the unbinned fit with #31; `hist_nn` gives alpha
  -1.13 and the binned response -1.05 (injected -1.02).
