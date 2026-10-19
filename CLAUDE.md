# CLAUDE.md

Notes for Claude Code sessions working on this repo, in particular on the
relative-coordinates histogram IRF (`IRFRelativeHistUnpolarized`) and the
histogram-based background (`FreeNormHistBackgroundDensity`).

## Repo and branch workflow

- This is `israelmcmc-ai/cosipy`, a fork of `cositools/cosipy`. The working
  branch is `develop_israel`: `cositools/develop` plus this CLAUDE.md, which
  must not go upstream. (`rel_irf_hist` was merged upstream as
  `cositools/cosipy#641`.)
- To update `develop_israel`, merge `cositools/develop` into it (remote
  `upstream` = `https://github.com/cositools/cosipy.git`) with a merge commit,
  never a rebase, and only when asked. Then merge `develop_israel` into the
  open PR branches that conflict, keeping upstream's behavior and wording.
- **Don't commit directly to `develop_israel`** (except that sync, when asked).
  Put work on a new branch and open a PR with base `develop_israel`.
  Unrelated side fixes (e.g. to `EnergySelector`, `DistanceSelector`, the
  chain selector, or the IRF while working on the background) go on their
  own branch with their own PR.
- The maintainer often pushes to the same PR branch while you work (e.g.
  comment edits, scratch scripts). Always `git fetch` before pushing, and
  only rebase *your own unpushed* commits on top; never force-push over
  their commits. Resolve conflicts keeping their wording.
- If asked to "put X in a separate PR and leave branch Z as it was": create
  the new branch at Z's tip, `git reset --hard <old tip>` on Z, then
  `git push --force-with-lease origin Z`, and open the PR from the new
  branch.
- Closed-unmerged PRs in this fork (e.g. #7, #8, #10, #11) were closed on
  purpose; don't redo them unless asked.
- The maintainer prefers simple code: refinements that "don't matter much"
  get reverted. Verify a change actually moves the needle before adding
  complexity.

## Python version for the tests

- cosipy requires Python >= 3.12 (`pyproject.toml`), but the sandbox's default
  `python` may be 3.11. There, `isinstance()` against the
  `@runtime_checkable` protocols in `cosipy/interfaces` evaluates properties
  (e.g. `EventDataInterface.nevents`), so 12 tests in `tests/event_selection/`
  fail with `TypeError: iter() returned non-iterator of type 'NoneType'`.
  They pass on 3.12; this is not a real failure.
- Run the tests in a 3.12 venv. `uv` also builds `antlr4-python3-runtime`
  fine, so the full install (astromodels/threeML included) works and the
  stub loader below is only a fallback:

  ```bash
  uv venv -p 3.12 $SCRATCH/venv312
  VIRTUAL_ENV=$SCRATCH/venv312 uv pip install -e . pytest
  $SCRATCH/venv312/bin/python -m pytest tests/event_selection/
  ```

## Running the tests in a sandbox without astromodels

`import cosipy` pulls in `astromodels`/`threeML` (via
`cosipy/__init__.py` → `response` → `threeml`), which may fail to install
(`antlr4-python3-runtime==4.9.3` build error with Debian's setuptools,
`AttributeError: install_layout`). A full install works if that wheel is
built in a clean venv first:

```bash
pip download "antlr4-python3-runtime==4.9.3" --no-deps --no-binary :all: -d $SCRATCH/antlr
tar xzf $SCRATCH/antlr/*.tar.gz -C $SCRATCH/antlr
python -m venv $SCRATCH/bvenv && $SCRATCH/bvenv/bin/pip install setuptools wheel
$SCRATCH/bvenv/bin/pip wheel --no-deps $SCRATCH/antlr/antlr4-python3-runtime-4.9.3 -w $SCRATCH/wheels
pip install $SCRATCH/wheels/antlr4_python3_runtime-4.9.3-py3-none-any.whl && pip install -e .
```

The unbinned tutorials also need the `[ml]` extras even in the hist modes
(`UnbinnedThreeMLPointSourceResponseIRFAdaptive` imports torch):
`pip install torch normflows "sphericart[torch]" torch_geometric` (several GB
with the CUDA libraries; don't use `--no-deps` for torch, it then fails to
import). Running the `nn` modes on CPU is very slow (> 1 h for the GRB cache).

If the full install isn't possible, install the light deps and load just the
needed submodules with stub packages:

```bash
pip install histpy scoords mhealpy h5py tqdm yayc pytest typing_extensions
```

Save as e.g. `$SCRATCH/load_relative_irf_hist.py` (not in the repo):

```python
import sys, types, importlib.util
ROOT = "/home/user/cosipy"  # adjust

def load(fullname, path):
    spec = importlib.util.spec_from_file_location(fullname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod

def stub_pkg(name, subdir):
    pkg = types.ModuleType(name)
    pkg.__path__ = [f"{ROOT}/{subdir}"]
    sys.modules[name] = pkg
    return pkg

def export(pkg, mod):
    for n in dir(mod):
        if not n.startswith('_'):
            setattr(pkg, n, getattr(mod, n))

stub_pkg('cosipy', 'cosipy')
stub_pkg('cosipy.util', 'cosipy/util')
load('cosipy.util.iterables', f'{ROOT}/cosipy/util/iterables.py')

pol = stub_pkg('cosipy.polarization', 'cosipy/polarization')
for m in ['conventions', 'polarization_angle', 'polarization_axis']:
    export(pol, load(f'cosipy.polarization.{m}', f'{ROOT}/cosipy/polarization/{m}.py'))

iface = stub_pkg('cosipy.interfaces', 'cosipy/interfaces')
for m in ['event', 'data_interface', 'event_selection', 'photon_parameters',
          'instrument_response_interface']:
    export(iface, load(f'cosipy.interfaces.{m}', f'{ROOT}/cosipy/interfaces/{m}.py'))

stub_pkg('cosipy.event_selection', 'cosipy/event_selection')
for m in ['time_selection', 'energy_selection', 'distance_selection']:
    load(f'cosipy.event_selection.{m}', f'{ROOT}/cosipy/event_selection/{m}.py')

stub_pkg('cosipy.response', 'cosipy/response')
load('cosipy.response.relative_coordinates', f'{ROOT}/cosipy/response/relative_coordinates.py')
load('cosipy.response.relative_irf_hist', f'{ROOT}/cosipy/response/relative_irf_hist.py')
```

Then run pytest from the same interpreter so the stubs stay in `sys.modules`:

```python
exec(open('load_relative_irf_hist.py').read())
import pytest, sys
sys.exit(pytest.main(['-v', 'tests/response/test_relative_irf_hist.py']))
```

The ~28 `RuntimeWarning: divide by zero` warnings from histpy come from
existing code and are harmless.

Real response files live on COSI's Wasabi bucket
(`s3.us-west-1.wasabisys.com`). Whether you can reach it depends on the
environment's network policy, so **try first** rather than assuming:

```python
from cosipy.util import fetch_wasabi_file  # needs full cosipy; otherwise load
                                           # cosipy/util/data_fetching.py directly
fetch_wasabi_file('COSI-SMEX/develop/Data/Responses/relative_hist_irf_from_nf_response.h5.zip',
                  output='relative_hist_irf_from_nf_response.h5.zip', unzip=True,
                  checksum='d9093daeaf56a386095a42ae40c6635e')             # hist_nn
fetch_wasabi_file('COSI-SMEX/develop/Data/Responses/ResponseContinuum.area.relative.nonsparse_smoothing1p0.h5.zip',
                  output='ResponseContinuum.area.relative.nonsparse_smoothing1p0.h5.zip', unzip=True,
                  checksum='bd2dfa700d0d382b052ea428a7eeabe1')             # hist_simple
```

A quick reachability probe is
`curl -sS -o /dev/null -w '%{http_code}\n' https://s3.us-west-1.wasabisys.com`:
`CONNECT tunnel failed, response 403` means an egress proxy blocks it,
while any HTTP status from Wasabi itself means it's reachable. If it's
reachable, validate energy-selection changes on these real files as well;
if it's blocked, say so and validate with synthetic histograms.

## `IRFRelativeHistUnpolarized` (`cosipy/response/relative_irf_hist.py`)

- 6D histogram with axes `[NuLambda, Ei, Epsilon, Phi, Theta, Zeta]`, where
  `Epsilon = (Em - Ei)/Ei`. `Ei` is usually a log-scaled axis.
- **Bin contents are per-bin effective area (cm²), not densities.** Both
  builders write it that way:
  - `scripts/IRFRelativeHist/relative_hist_irf_from_rsp.py`: counts ×
    EFF_AREA (MEGAlib simulation, `hist_simple` mode).
  - `scripts/IRFRelativeHist/relative_hist_irf_from_nf_response.py`:
    `tot_aeff · density · phase_space_cds · Ei_center·ΔEps` (`hist_nn` mode).

  `__init__` divides by the phase space (CDS volume and `Ei_center·ΔEps`)
  to get `_diff_aeff`.
- `_tot_aeff` is either `irf.project('NuLambda', 'Ei')` or a separate `aeff`
  histogram (usually finer). `from_h5` reads an `AEFF` group automatically
  if present. `_tot_aeff` is linearly interpolated in `Ei` at evaluation
  time.
- The tutorial is
  `docs/tutorials/spectral_fits/continuum_fit/grb/example_grb_fit_relative_hist_response.ipynb`
  (`irf_mode` = `hist_simple` / `hist_nn` / `nn`).

### Energy selections (`selections=EnergySelector(...)`)

- A tuple of selectors is OR'd via `EnergySelector.union`. `EnergySelector`
  (`cosipy/event_selection/energy_selection.py`) has
  `energy_ranges_keV`, `min/max_energy[_keV]`, `union`, `intersect`,
  `except_`.
- `_apply_energy_selection` scales `_tot_aeff` per `(NuLambda, Ei)` by the
  fraction of area with `Em` inside the cut. `_diff_aeff` is never modified.
  - **Same grid** (no separate `aeff`): the fraction is computed at irf's own
    `Ei` centers and the grid is not refined.
  - **Separate `aeff`**:
    1. `_refine_ei_edges` adds `Ei` edges at `E_cut / (1 + Epsilon)` for
       every Epsilon edge and center (the fraction's kinks), keeping all the
       original edges.
    2. `_regrid_ei` resamples `aeff` onto the refined grid using the axis's
       own `interp_weights`.
    3. irf's `(NuLambda, Ei, Epsilon)` projection is interpolated onto that
       grid one Epsilon center at a time.
    4. `_selection_fraction` evaluates the cut at each exact target `Ei`.

    So `_tot_aeff` can end up with more `Ei` bins than the `aeff` passed in.
- `_integrate_piecewise_linear` integrates the per-Epsilon density
  (content/ΔEps), linear between centers and flat beyond the first/last
  center. Integrating over the full range is **not** exactly the content
  sum when bins are non-uniform, so `content.sum()` is used as the total.

### Lessons from debugging narrow energy cuts

- The two changes that mattered:
  1. Evaluate the fraction at the target `Ei` instead of interpolating a
     fraction computed on irf's coarse grid.
  2. Refine the `aeff` grid. With `geomspace(50, 10000, 41)` and a
     495–505 keV cut, sampling only at bin centers made the integral over
     `Ei` about 2× too large; refining brings it within 0.2%.
- Integrating over `dEps` vs `dEm` at a fixed `Ei` is just a change of
  variables (numerically identical), so it's not a source of error.
- Interpolating content as a density across `Ei` (divide by native `Ei`,
  multiply by the target) was tried and reverted as not worth the
  complexity.
- A synthetic test whose Epsilon profile has the same shape and scale at
  every `Ei` can't catch `Ei`-interpolation problems. Use shapes that vary
  with `Ei` when testing those.

## Validating on real files in a sandbox

- The full hist IRF files hold 9.6 GB of contents. Without
  https://github.com/israelmcmc-ai/cosipy/pull/24 (`from_h5` defaults to `copy=False`,
  non-finite cleanup in place) loading one needs > 14 GB and gets OOM-killed in a 15 GB
  sandbox; with it, the peak is ~11 GB, enough to run the relative-hist tutorials. For
  line validation, slicing the `Ei` axis of the h5 with h5py (e.g. 916-5000 keV for
  Al-26) is enough.
- Run notebooks headless with `nbclient` (`NotebookClient(nb, timeout=None,
  resources={'metadata': {'path': ...}}).execute()`) on a copy with `data_path`
  changed; check `dmesg` for "Memory cgroup out of memory" if the kernel dies.
- Large Wasabi files (e.g. the 16 GB `Total_DC4_BG`) can be streamed with boto3 +
  gzip and filtered by time without a local copy (fixed-size FITS rows, column names
  from the header), but the connection resets every few GB: reopen with
  `Range=bytes=<pos>-` and keep going.

## Histogram background (`cosipy/background_estimation/free_norm_hist_background.py`)

- `HistBackgroundTemplate` holds raw counts (`rate_counts` [Time, Em], `livetime`
  [Time], `phi_counts` [Rocking, Em, Phi], `psichi_counts` [Rocking, Em, Phi,
  PsiChi]) built from the data itself, in the SC frame; `fill`/`remove`/`smooth`/`+`
  and HDF5 `write`/`open`. `FreeNormHistBackgroundDensity` evaluates it with one free
  norm (Hz) per component. Only the SC frame is supported for now.
- 3-month DC4 mock templates (all events / Distance >= 1 cm) are on Wasabi under
  `COSI-SMEX/develop/Data/Products_and_Templates/Background_Models/HistBackgroundTemplate/`,
  built with `cosipy/background_estimation/scripts/HistBackgroundTemplate/`.
- Smoothing was chosen by cross-validation (template from a random half of the events,
  log-likelihood of the other half in 1 h windows): `psichi_fwhm=4 deg`,
  `phi_fwhm=20 deg`, `time_counts=500`. Use that, not eyeballing, to retune.
- The template contains all the sources in the data. For a transient, `remove` its
  on-time window (GRB tutorial). In the DC4 mock data the sources other than the
  target are ~5% of the events, so the data-driven template overpredicts the true
  background by a few %; the Crab is only ~1.6% of the events in 3 h, so the Crab
  tutorial is the real test of the background model.
- Injected spectra: the mock dataset uses the DC4 sources (e.g. the DC4 Crab is
  nebula + 3 pulsar components, in cosi-sim `Source_Library/DC4`), not the DC3 ones.

## histpy gotchas

- `Histogram.interp()` / `Axis.interp_weights()` on a `scale='log'` axis
  interpolate linearly in **log(x)** between bin centers, and clamp to the
  first/last center.
- `Histogram.copy()` doesn't deep-copy `Axis` objects, and
  `IRFRelativeHistUnpolarized.__init__` strips units from axes in place.
  So after constructing a model, the caller's original `irf`/`aeff`
  histograms already have unitless (keV) axes, even with `copy=True`.
- Projections sum contents over the dropped axes.

## Conventions

- Default to no code comments unless the "why" is non-obvious. Docstrings in
  this module are numpy-style.
- PR replies and review comments in this fork end with the Claude Code
  attribution footer.
