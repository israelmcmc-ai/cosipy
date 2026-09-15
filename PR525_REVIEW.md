# Review of cositools/cosipy#525 — Stokes polarization fitting + MDP

**PR:** https://github.com/cositools/cosipy/pull/525
**Head reviewed:** `38b3c07` ("Fix unit test")
**Merge base with `develop`:** `e3d4d83`

Scope of the diff (excluding the tutorial notebook):

| File | Change |
|---|---|
| `cosipy/threeml/util.py` | `to_linear_polarization` now converts `StokesPolarization` instead of raising |
| `cosipy/response/instrument_response.py` | **Actually applies the PA-convention rotation** in the inertial path (`_rot_psr_pol`); drops the `polarization` argument from `_differential_effective_area_inertial` |
| `cosipy/response/PointSourceResponse.py` | `get_expectation` handles `StokesPolarization` directly, with an `exp((pd-1)²)` barrier for `pd > 1` |
| `cosipy/sensitivity/mdp.py` | New `compute_mdp` |
| `tests/polarization/test_polarization_mlm.py` | Reparameterized to Q/U; new `test_mdp` |

## Verdict

The core change is a genuine bug fix and it's the right one. Before this PR, the
inertial (galactic-PsiChi) path built a `Pol` axis in `out_axes` and then called
`_rot_psr`, which rotates **only** `PsiChi`. The polarization-angle bins were
carried straight from the spacecraft frame into the source frame with no
convention transform at all. `_differential_effective_area_inertial` now mirrors
`FullDetectorResponse.get_point_source_response` and calls `_rot_psr_pol`. That
part should go in.

What I'd hold the PR on is everything built on top of it. The Stokes
reparameterization sits on a likelihood that is **piecewise constant in
polarization angle**, and the `exp((pd-1)²)` barrier turns the resulting random
walk into `inf`/`NaN`. Both are demonstrated below with numbers from the PR's own
test and notebook. The MDP module has three API-level bugs that make it unusable
outside the exact shape of the unit test.

**Environment used for all measurements below:** Python 3.12, `pip install .`
from `38b3c07`, astromodels 2.6.0, threeml 2.6.1, histpy 2.0.7, numpy 2.5.3.
`pytest tests/response tests/polarization` → **37 passed, 5 skipped** in 57 s,
so nothing existing regresses. `test_mdp` takes **14 s**, which is fine for CI.

---

## A. Correctness — things that can produce a wrong number

### A1. The likelihood is a 12-step staircase in polarization angle, so the fit covariance is singular

`get_expectation` picks the polarized weight with a **nearest-bin lookup**
([`PointSourceResponse.py:117-118`](https://github.com/cositools/cosipy/pull/525/files#diff-ad7707b33ae0be2fca555f4be3d1203fab920056794d6a5408083ab7336f680bR117)):

```python
polarization_bin_index = pol_axis.find_bin(polarization_angle * u.deg)
weights[polarization_bin_index] += polarization_level
```

The expectation is therefore *identical* for every PA inside a response bin. I
scanned `-logL` over PA at fixed PD = 0.5 using the unit-test data and response:

```
PA=  0..15 deg  -logL = -16280.534367
PA= 20..30 deg  -logL = -16173.308311
PA= 35..40 deg  -logL = -16012.277644
PA= 45..60 deg  -logL = -15843.395157
...
distinct -logL values over 180 integer PAs at fixed PD: 12
```

Exactly 12 values — one per response `Pol` bin. `∂(-logL)/∂PA = 0` almost
everywhere.

This pre-dates the PR (the `TODO: this could also be interpolated` comments are
already there), but the Q/U reparameterization is what makes it bite. With
`(degree, angle)` the flat direction was confined to one parameter. With
`(Q, U)` **both** parameters mix the flat direction, so MINUIT's Hessian is
singular in both. That is visible in the notebook this PR ships:

| | before (merge base) | after (this PR) |
|---|---|---|
| parameters | `degree = 26.5 ± 3.2` %, `angle = 89.9998 ± 0.0013` deg | `Q = -0.2 ± 0.7`, `U = 0.1 ± 2.8` |
| correlation matrix | `1.00 / -0.53` | `1.00 / 1.00` |
| `-log(likelihood)` | **21743.138** | **21753.190** |

`U = 0.1 ± 2.8` on a parameter bounded by |U| ≤ 1 is not a usable uncertainty,
and a correlation of exactly 1.00 is MINUIT telling you the covariance is
degenerate. (The old `± 0.0013 deg` on the angle was the same pathology wearing
a different hat.)

**Recommendation:** interpolate over the `Pol` axis (`pol_axis.interp_weights`,
which `PolarizationAxis` already provides) instead of `find_bin`, in both
`get_expectation` and the `_rot_psr_pol` bin mapping. Until the model is smooth
in PA, reported polarization uncertainties from this path should not be
trusted, and I'd say so in the notebook.

### A2. `exp((pd - 1)²)` overflows to `inf`, then `inf × 0` → `NaN` in the likelihood

[`PointSourceResponse.py:96`](https://github.com/cositools/cosipy/pull/525/files#diff-ad7707b33ae0be2fca555f4be3d1203fab920056794d6a5408083ab7336f680bR96):

```python
factor = np.exp((pd - 1.)**2)
```

`Q` and `U` are `Constant.k` parameters with no bounds, so `pd` is unbounded.
`np.exp((pd-1)²)` overflows at **pd ≥ 28**, and this is not hypothetical — it
happens in this PR's own unit test:

```
tests/polarization/test_polarization_mlm.py::test_mdp
  WARNING RuntimeWarning: overflow encountered in exp
  WARNING RuntimeWarning: invalid value encountered in multiply
  WARNING RuntimeWarning: invalid value encountered in subtract
```

Running `compute_mdp(50, ...)` over six seeds:

```
seed=0 mdp=14.799%  overflow_warnings= 0  nan_warnings= 0
seed=1 mdp=13.177%  overflow_warnings= 4  nan_warnings= 8
seed=2 mdp=15.274%  overflow_warnings= 3  nan_warnings= 9
seed=3 mdp=17.177%  overflow_warnings= 4  nan_warnings=12
seed=4 mdp=15.268%  overflow_warnings=18  nan_warnings=54
seed=5 mdp=16.959%  overflow_warnings= 1  nan_warnings= 3
```

`inf × 0 = nan` in the `tensordot`, and the `NaN` propagates into the Poisson
likelihood (`invalid value encountered in subtract`). The minimizer is being fed
`NaN` objective values and nobody notices, because the barrier is a silent
multiplicative factor. This is also *why* the minimizer wanders out to pd ≈ 28
in the first place: per A1 there is no gradient in the PA direction to pull it
back.

**Recommendation:** enforce the physical constraint where it belongs — bound
`Q` and `U` (or fit `pd, PA` with `pd ∈ [0,1]`), or clamp the exponent
(`np.exp(np.minimum((pd-1)**2, 50))`). A soft barrier hidden inside the forward
model also means anyone calling `get_expectation` for simulation or display
silently gets inflated counts rather than an error.

### A3. Related: the penalty is applied to the model, not to the likelihood

`expectation = tensordot(...) * factor`
([`PointSourceResponse.py:139`](https://github.com/cositools/cosipy/pull/525/files#diff-ad7707b33ae0be2fca555f4be3d1203fab920056794d6a5408083ab7336f680bR139))
scales the predicted counts. With a free background normalization the fitter can
partially absorb `factor` by lowering the background, which weakens the barrier
in exactly the configuration where you need it. A penalty term added to the
objective would not have that leak.

### A4. The notebook's own result is worth a second look before merging

The tutorial states the GRB "was simulated with 80% polarization at an angle of
90 degrees in the IAU convention". The committed outputs give:

- **before:** `degree = 26.5 ± 3.2` %, `angle = 89.9998` deg — angle right, degree 3× low
- **after:** `Q = -0.2 ± 0.7`, `U = 0.1 ± 2.8` → 23.3 %, 83.3 deg, and `-logL` **worse by 10.05**

The response rotation changed, so the two `-logL` values aren't strictly
comparable — but a corrected response landing at a *worse* minimum, with a
degenerate covariance, reads more like a convergence failure (A1) than a better
model. Separately, the notebook reports **MDP(99%) = 8.11 %** while the same fit
reports σ_Q ≈ 0.7 (70 %). Those two numbers cannot both be right: if the MDP
really were 8 %, a 23 % measurement would be a many-σ detection with a small
error bar. I'd want that reconciled before this ships as a tutorial, since the
80 %/90 % truth is stated right there on the page.

### A5. `compute_mdp` seeds each fit at the user's polarization

`this_model = copy.deepcopy(model)`
([`mdp.py:56`](https://github.com/cositools/cosipy/pull/525/files#diff-1540a6c3b1be8e585d96f84ff539aed18ee4a94ca000333a5c3fe2055472c9ed))
keeps the polarization of the **input** model, while the injected data is always
`LinearPolarization(angle=0, degree=0)`. So the MDP depends on a property of the
input model that should be irrelevant to it — in the notebook the seed is 80 % at
150°, in the unit test 0.5 % at 100°. With a local minimizer on the staircase
likelihood of A1, the starting point matters. Either start every fit
unpolarized, or document that the model's polarization is used as the fit seed.

The one thing I checked and found **correct**: the Stokes → linear math in
`to_linear_polarization`. `(.5 * arctan2(U,Q)) % π` parses as intended (`*` and
`%` are same-precedence, left-associative) and round-trips properly:

```
Q=+0.50 U=+0.00 -> 50.00% @   0.00 deg
Q=+0.00 U=+0.50 -> 50.00% @  45.00 deg
Q=-0.50 U=+0.00 -> 50.00% @  90.00 deg
Q=+0.00 U=-0.50 -> 50.00% @ 135.00 deg
```

---

## B. Effects outside the stated scope of the PR

### B1. `differential_effective_area(polarization=...)` is now silently ignored on the inertial path

The argument was removed from `_differential_effective_area_inertial`
([`instrument_response.py:194`](https://github.com/cositools/cosipy/pull/525/files#diff-23d0650d3f53fe0365d21dc5ed827fac7b78883ff981d878841dd29bdcc5c4c4R194))
but is still accepted by the public `differential_effective_area`, still
documented there, and still declared in
`cosipy/interfaces/instrument_response_interface.py`. The output binning is now
always the response's own `Pol` binning.

`BinnedThreeMLPointSourceResponse.__init__` documents `polarization_axis` as
"the **desired** effective binning of the photon polarization angle ... This also
defines the polarization coordinate system and convention". That promise no
longer holds. Passing a coarser axis:

```
(b) coarse 3-bin Pol axis FAILS: ValueError operands could not be broadcast
    together with shapes (4,3,4,30,12) (4,12,4,30,12) (4,3,4,30,12)
```

A raw broadcast error is a poor diagnostic for "your requested binning is
ignored". Either honour the requested axis, or validate it against
`self._dr.axes['Pol']` up front and raise something readable — and update both
docstrings.

### B2. The output `Pol` axis is hardcoded to `iau`, and that label is then dropped anyway

[`instrument_response.py:236`](https://github.com/cositools/cosipy/pull/525/files#diff-23d0650d3f53fe0365d21dc5ed827fac7b78883ff981d878841dd29bdcc5c4c4R236)
builds the out-axis with `convention='iau'` while the *caller* (the unit test,
the notebook, `mdp.py`) supplies a `RelativeZ` axis to `from_scatt_map`, and that
caller axis is what ends up on the returned `PointSourceResponse`. So the PSR is
labelled `RelativeZ` while its contents are indexed by IAU angle. It happens to
work because the bin edges are numerically identical and `get_expectation` looks
up a bare `Quantity`, which skips `PolarizationAxis._standardize_value`. Pass a
`PolarizationAngle` instead and you would silently get the wrong bin.

Note also that `FullDetectorResponse.get_point_source_response` does the
*opposite* — it leaves `psr_axes['Pol']` on the response's own convention while
filling it with IAU-indexed contents. Two code paths, two different labellings
of the same quantity. Worth settling on one.

There's a reason nobody has noticed: **`PolarizationAxis._copy` loses
`_convention` entirely**, so `Axes([...])` (which copies by default) strips it:

```
Axes([pol]) keeps convention:                   False
Axes([pol], copy_axes=False) keeps convention:  True
pol.copy() keeps:                               False
from response file (Axes.open):                 True
```

```python
>>> psr.axes['Pol'].centers
AttributeError: 'PolarizationAxis' object has no attribute '_convention'
```

The comment in `polarization_axis.py:67` says *"self._convention is not copied.
It's safe to share it."* — but `super()._copy` never runs
`PolarizationAxis.__init__`, so it isn't shared either. This is pre-existing and
outside the diff, but it is exactly the mechanism that hides B2, and it means the
`convention='iau'` on line 236 is currently dead metadata. Worth a one-line fix
in `_copy` (`new._convention = self._convention`) in this PR or a companion.

### B3. `to_linear_polarization` no longer raising opens paths that aren't ready

Two other call sites gate on it:

- `threeml_point_source_response.py:140` (`set_source`)
- `threeml_extended_source_response.py:111` (`set_source`)

Both previously got a hard `ValueError` for a Stokes source and now proceed.
Neither knows about the `pd > 1` case, and the extended-source path has no
equivalent of the `factor` handling at all. If Stokes support is meant to be
point-source-only for now, that's worth an explicit check rather than a silent
divergence in behaviour.

Also, the conversion assumes `Q`/`U` are `Constant` functions:

```python
Q = polarization.Q.value.k.value   # .k only exists on Constant
```

Any other astromodels function gives a confusing error and energy-dependent
Q/U are silently unsupported:

```
Line raises: AttributeError Accessing an element k of the node that does not exist
```

A `isinstance(..., Constant)` check with a clear message would save someone an
afternoon.

### B4. Implicit assumption: response axis order is `[Ei, Pol] + data axes`

`_rot_psr_pol` resolves `label_to_index('Pol')` against `out_axes`, but indexes
the array returned by `_get_pixel`, which is in `_rest_axes` order. These agree
only by luck of file layout:

```
DR axes:  ['NuLambda' 'Ei' 'Pol' 'Em' 'Phi' 'PsiChi']
rest:     ['Ei' 'Pol' 'Em' 'Phi' 'PsiChi']
out_axes: ['Ei' 'Pol' 'Em' 'Phi' 'PsiChi']
```

A response file with `Pol` anywhere else, or binned data whose axes aren't in
`Em, Phi, PsiChi` order (`__init__` checks the label *set*, not the order),
would take along the wrong axis and produce a silently wrong PSR. One
`assert out_axes.labels == self._dr._rest_axes.labels` would close this.

### B5. Unpolarized analysis against a polarized response is still broken in the inertial path

The local path handles `polarization=None` on a polarized response by projecting
`Pol` out and dividing by `nbins`. The inertial path keys off
`self.is_polarization_response` (a property of the *response*, not of the
request), so it always emits a `Pol` axis:

```
after  this PR: ValueError operands could not be broadcast together with
                shapes (4,4,30,12) (4,12,4,30,12) (4,4,30,12)
before this PR: AttributeError 'NoneType' object has no attribute 'centers'
```

**Not a regression** — broken both ways — but a plain spectral fit with a
polarized response in galactic coordinates is a real use case, and this PR is
touching exactly that code.

Adjacent, also pre-existing: `out = Quantity(np.zeros(out_axes.shape),
dr_pix.unit, ...)` at
[`instrument_response.py:282`](https://github.com/cositools/cosipy/pull/525/files#diff-23d0650d3f53fe0365d21dc5ed827fac7b78883ff981d878841dd29bdcc5c4c4R282)
references an undefined `dr_pix`, so any call with `out=None` is a `NameError`.
Cheap to fix while you're in the file.

---

## C. `cosipy/sensitivity/mdp.py`

### C1. The source must literally be named `source` — confirmed failure

`model.source` (used three times) is attribute access on the astromodels node
tree, so it only resolves for a source named `"source"`. Both the unit test and
the notebook name it `'source'`, which is why nobody hit it:

```
compute_mdp with a source named 'grb':
  AttributeError: Accessing an element source of the node that does not exist
```

Use `model.sources[name]` / `list(model.sources.values())[0]`, or take the source
name as an argument.

### C2. The documented default `response_pa_convention=None` raises — confirmed failure

```
compute_mdp(..., response_pa_convention=None):
  TypeError: Input must be str or subclass of PolarizationConvention
```

from `PolarizationAxis(dr.axes['Pol'], convention=None)`. Since MDP only makes
sense for a polarization response, and the response file already carries its
convention, the default should be `dr.axes['Pol'].convention` rather than
`None`.

### C3. Wrong check, wrong message

```python
if len(model.source.spectrum.to_dict()) > 1:
    raise RuntimeError('Model cannot contain more than one source.')
```

This counts **spectral components of one source**, not sources. `len(model.sources) > 1`
is what the message describes.

### C4. Bare `except:`

`except:` at `mdp.py:113` swallows `KeyboardInterrupt` and `SystemExit`, and
hides *why* fits fail — which matters a lot given A2. `except Exception as e:`
plus a `logger.debug(e)` would make the NaN problem visible instead of silently
incrementing a counter.

### C5. The point source response is rebuilt on every iteration

`SourceInjector`, `inject_model`, `BinnedInstrumentResponse`,
`BinnedThreeMLPointSourceResponse` and `BinnedThreeMLModelFolding` are all
constructed inside the `for i in range(n)` loop, but none of their inputs change
across iterations — only the Poisson draw does. `from_scatt_map` (the expensive
part) therefore runs `n` times instead of once. The docstring in the notebook
says "This currently takes a long time to run, so this only runs 100
simulations"; hoisting the invariant work out of the loop is most of that time.

### C6. Smaller things

- `np.percentile(degrees, confidence)` on an empty list if every fit fails.
- `logger.warning(f'{failed_fits}/{n} fits failed')` fires unconditionally, including `0/n`.
- `background.expectation()` is called three times per iteration.
- No RNG seeding or `rng` argument — results are irreproducible run to run.
- `degree` can exceed 100 % (`to_linear_polarization` doesn't clamp), so the returned MDP can too.
- `StokesPolarization` is imported but unused; ditto `PolarizationAngle` in `instrument_response.py:13`.
- File has no trailing newline.
- Tabs for indentation (85 lines in `mdp.py`, 43 in the test) where the rest of the repo uses 4 spaces. The test diff is mostly whitespace churn for this reason, which makes the real change harder to see.

---

## D. Test coverage

**Covered.** `test_polarization_fit` and `test_mdp` both exercise the new
rotation code end to end — the binned test data has `PsiChi` in galactic, so
`_differential_effective_area_inertial` → `_rot_psr_pol` is genuinely hit, as is
the `StokesPolarization` branch of `to_linear_polarization` and the
`polarization_level is None` branch of `get_expectation`. Both run fast (0.3 s
and 14 s). The rest of `tests/response` and `tests/polarization` still passes.

**Not covered:**

1. **The rotation itself is never checked against a known answer.** The only
   assertions are end-to-end fit results on one dataset at one attitude. There's
   no test that `_rot_psr_pol` does the right thing — e.g. an identity attitude
   where IAU and local conventions coincide should reproduce `_rot_psr`'s PSR
   bin-for-bin, and a 90° roll should shift the `Pol` bins by a predictable
   amount. That's the actual physics fix in this PR and it's untested.
2. **The `pd > 1` branch** (`PointSourceResponse.py:94-98`) — the `factor`,
   the `polarization_level = 1.` clamp, and the overflow. It's *hit* during
   `test_mdp` (that's where the warnings come from) but nothing asserts on it.
   A direct test would be three lines and would have caught A2.
3. **`to_linear_polarization` has no unit test at all** — not the Stokes branch,
   not the `None` branch, not the base-`Polarization` branch. The four-quadrant
   round-trip in A5 above is worth committing.
4. **The local / spacecraft-frame polarization path** is untested with this
   change; all polarization tests use galactic `PsiChi`.
5. **`compute_mdp` argument handling** — C1 and C2 are both one-line tests.
6. **`test_mdp`'s assertion is mis-centred and has no power.**
   `np.allclose([mdp], [25.], atol=[15.])` accepts anything in `[10, 40]`. The
   measured distribution over six seeds is **13.2–17.2 %, mean 15.4 %** — the
   nominal 25 is ~10 points off centre, and the test would still pass at 39 %
   (2.5× the true value). With no seed, it's also nondeterministic. Seed the RNG
   and tighten to something like `25 → 15 ± 3`, or assert on a property that
   doesn't need a wide band.
7. The comparison values in `test_polarization_fit` (`Q, U = 0.74, 0.0`) are
   regression locks, not physics. Given A4, it's worth stating in a comment what
   the simulated truth for `polarization_data_binned.hdf5` actually is.

---

## E. Summary of requested changes

**Blocking**

- A2 — bound `Q`/`U` or clamp the exponent so the likelihood can't go `NaN`.
- C1, C2 — `compute_mdp` fails outright for any source not named `source`, and for its own documented default.
- A4 — reconcile the notebook's fitted 23 % @ 83° / MDP 8.11 % against the stated 80 % @ 90° truth, or drop the claim.

**Should fix in this PR**

- A1 — interpolate the `Pol` axis, or document that fitted uncertainties are unreliable.
- B1 — honour or reject the caller's `polarization_axis`; fix the docstrings either way.
- B2 — settle the IAU-vs-response-convention labelling; fix `PolarizationAxis._copy`.
- B4 — assert the axis-order assumption.
- D1, D2, D3 — unit tests for the rotation, the `pd > 1` branch, `to_linear_polarization`.
- D6 — seed and tighten `test_mdp`.
- C3, C4, C5 — the source-count check, bare `except`, and hoisting the PSR out of the loop.

**Nice to have**

- B3, B5, C6, plus converting tabs to spaces to make the test diff readable.

Nothing here argues against the central fix — `_rot_psr_pol` in the inertial
path is correct and needed. It's the layer on top that needs another pass.
