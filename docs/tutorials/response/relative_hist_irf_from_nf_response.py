#!/usr/bin/env python
# coding: UTF-8

"""
Build the HDF5 input expected by IRFRelativeHistUnpolarized.from_h5() out of
an UnpolarizedNFFarFieldInstrumentResponseFunction (the neural-network / flow
based response).

IRFRelativeHistUnpolarized stores a 6D histogram (NuLambda, Ei, Epsilon, Phi,
Theta, Zeta) whose *contents* are per-bin effective area (cm^2); at load time
it divides those contents by each bin's phase-space volume to get the
differential effective area it actually interpolates on (see
IRFRelativeHistUnpolarized.__init__ and its docstring).

This script does the inverse operation with 0-order (midpoint rule)
integration: for every bin of a target 6D grid it

  1. evaluates the NF response's differential effective area
     (= effective_area(NuLambda, Ei) * event_probability_density(...)) at
     the bin *center*, and
  2. multiplies it by that bin's phase-space volume,

which is exactly the per-bin content IRFRelativeHistUnpolarized expects.
The phase-space formulas used here are copied verbatim from
IRFRelativeHistUnpolarized.__init__ so that re-loading the produced file
divides out precisely what was multiplied in here.

Caveats of the 0-order approximation:
  - Only accurate if bins are narrow enough that the response doesn't vary
    much across a bin. Coarse bins (e.g. wide Ei bins at low energy, where
    the response changes quickly) will be biased.
  - A bin whose (Phi, Theta) corner is only partially in the physical region
    (Phi+Theta in [0, pi]) is still filled by evaluating at the (clipped) bin
    center, weighted by its (correctly clipped) phase space -- bins that are
    *entirely* unphysical are skipped and left at zero, matching how
    IRFRelativeHistUnpolarized itself zeroes those bins.

Grid size / runtime
--------------------
The bulk of the work is one call to NFResponse.evaluate_density() per chunk
of `grid_batch_size` (Phi, Theta)-physical bin combinations of
(NuLambda, Ei, Epsilon, Zeta). The total number of such combinations is
roughly:

    npix * nEi * nEpsilon * nPhi * nTheta * nZeta / 2

(the /2 accounts for only ~half of the Phi-Theta plane being physical).
This can get very large very quickly -- e.g. nside=8 (768 pixels) with
10/30/50/50/50 bins in Ei/Epsilon/Phi/Theta/Zeta is already ~1.4e10
evaluations. Pick your target binning with that in mind; `grid_batch_size`
only bounds the *memory* used per chunk, not the total runtime.

Two ways to define the target grid are supported (see `__main__` below):
  - `axes_from_h5_template()`: reuse the NuLambda/Ei/Epsilon/Phi/Theta/Zeta
    axes (including healpix nside and the Zeta polarization convention) of
    an existing IRFRelativeHistUnpolarized-compatible file, e.g. so the NF
    response can be compared bin-for-bin against a MEGAlib-derived histogram
    response.
  - Building an `Axes` object by hand with custom bin edges.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

import astropy.units as u

from histpy import Histogram, Axes, Axis, HealpixAxis
from scoords import SpacecraftFrame

from cosipy.response.ml.NFResponse import NFResponse
from cosipy.response.ml.nf_instrument_response_function import UnpolarizedNFFarFieldInstrumentResponseFunction
from cosipy.response.relative_coordinates import RelativeCDSCoordinates

logger = logging.getLogger(__name__)

_EXPECTED_LABELS = ['NuLambda', 'Ei', 'Epsilon', 'Phi', 'Theta', 'Zeta']


def _to_value(quantity_like, unit=None):
    """Like Quantity.to_value(), but also accepts plain arrays/PolarizationAngle."""

    if hasattr(quantity_like, 'angle'):
        # PolarizationAngle (e.g. PolarizationAxis.centers/.edges)
        quantity_like = quantity_like.angle

    if hasattr(quantity_like, 'to_value'):
        return quantity_like.to_value(unit) if unit is not None else quantity_like.to_value(u.dimensionless_unscaled)

    return np.asarray(quantity_like, dtype=float)


def axes_from_h5_template(template_path: Union[str, Path]) -> Axes:
    """
    Load the NuLambda/Ei/Epsilon/Phi/Theta/Zeta axes (bin edges, healpix
    nside, Zeta polarization convention, units -- everything except the
    contents) from an existing IRFRelativeHistUnpolarized-compatible HDF5
    file, to reuse as the target grid for build_relative_hist_irf().

    Parameters
    ----------
    template_path : str or Path
        Path to an HDF5 file with an "IRF" group holding a 6D histogram with
        axes ['NuLambda', 'Ei', 'Epsilon', 'Phi', 'Theta', 'Zeta'].
    """

    template = Histogram.open(template_path, "IRF")

    _validate_axes(template.axes)

    return template.axes


def _validate_axes(axes: Axes):
    if not np.array_equal(axes.labels, _EXPECTED_LABELS):
        raise ValueError(f"axes labels must be {_EXPECTED_LABELS}, got {list(axes.labels)}")

    if not isinstance(axes['NuLambda'], HealpixAxis):
        raise ValueError("NuLambda axis must be a HealpixAxis")

    if not isinstance(axes['NuLambda'].coordsys, SpacecraftFrame):
        raise ValueError("NuLambda axis must be defined in the spacecraft frame")

    zeta_axis = axes['Zeta']
    if not hasattr(zeta_axis, 'convention'):
        raise ValueError("Zeta axis must be a PolarizationAxis")

    if not isinstance(zeta_axis.convention.frame, SpacecraftFrame):
        raise ValueError("Zeta axis polarization convention must be defined in the spacecraft frame")


def _phase_space_cds(phi_axis: Axis, theta_axis: Axis, zeta_axis) -> np.ndarray:
    """Phase-space volume of every (Phi, Theta, Zeta) bin. Same formula as
    IRFRelativeHistUnpolarized.__init__."""

    phi_edges_mesh, theta_edges_mesh, zeta_edges_mesh = np.meshgrid(
        _to_value(phi_axis.edges, u.rad),
        _to_value(theta_axis.edges, u.rad),
        _to_value(zeta_axis.edges, u.rad),
        indexing='ij')

    return RelativeCDSCoordinates.get_relative_cds_phase_space(
        phi_edges_mesh[:-1, :-1, :-1], phi_edges_mesh[1:, :-1, :-1],
        theta_edges_mesh[:-1, :-1, :-1], theta_edges_mesh[:-1, 1:, :-1],
        zeta_edges_mesh[:-1, :-1, :-1], zeta_edges_mesh[:-1, :-1, 1:])


def _phase_space_em(ei_axis: Axis, epsilon_axis: Axis) -> np.ndarray:
    """Phase-space volume of every (Ei, Epsilon) bin. Same formula as
    IRFRelativeHistUnpolarized.__init__ (dEm = Ei_center * dEpsilon)."""

    ei_centers_mesh, epsilon_widths_mesh = np.meshgrid(
        _to_value(ei_axis.centers, u.keV),
        _to_value(epsilon_axis.widths),
        indexing='ij')

    return ei_centers_mesh * epsilon_widths_mesh


def build_relative_hist_irf(nf_response: NFResponse,
                            axes: Axes,
                            output_path: Union[str, Path],
                            devices: Optional[List] = ("cpu",),
                            grid_batch_size: int = 200_000,
                            dtype=np.float32,
                            overwrite: bool = False,
                            show_progress: bool = True) -> Path:
    """
    Build an IRFRelativeHistUnpolarized-compatible HDF5 file from an
    UnpolarizedNFFarFieldInstrumentResponseFunction, by evaluating it at the
    center of every bin of `axes` and multiplying by the bin's phase-space
    volume (0-order / midpoint-rule integration). See module docstring.

    Parameters
    ----------
    nf_response : NFResponse
        Must be unpolarized (nf_response.is_polarized == False).
    axes : histpy.Axes
        Target 6D grid, labels ['NuLambda', 'Ei', 'Epsilon', 'Phi', 'Theta',
        'Zeta']. See axes_from_h5_template() to reuse an existing file's
        binning, or build one by hand (NuLambda must be a HealpixAxis in the
        spacecraft frame, Zeta a PolarizationAxis whose convention is also
        defined in the spacecraft frame).
    output_path : str or Path
        Where to write the resulting HDF5 file (loadable with
        IRFRelativeHistUnpolarized.from_h5(output_path)).
    devices : list, optional
        Passed to NFResponse.init_compute_pool(). Defaults to a single CPU
        worker; pass e.g. ["cuda:0"] or [0, 1] to use GPU(s) instead.
    grid_batch_size : int, optional
        Upper bound on the number of (NuLambda, Ei, Epsilon, Phi, Theta,
        Zeta) bin combinations evaluated by the NF response in one call.
        This only bounds memory use -- the total number of evaluations is
        set by `axes` (see module docstring). It is unrelated to
        nf_response's own area_batch_size/density_batch_size, which control
        how each such call is internally split across compute-pool workers.
    dtype : numpy dtype, optional
        dtype of the output histogram contents.
    overwrite : bool, optional
        Passed to Histogram.write().
    show_progress : bool, optional
        Print a progress bar over the (Phi, Theta)-bin loop.

    Returns
    -------
    Path
        `output_path`, for convenience.
    """

    if nf_response.is_polarized:
        raise ValueError("nf_response is polarized; build_relative_hist_irf() only supports "
                          "unpolarized NF responses (see UnpolarizedNFFarFieldInstrumentResponseFunction).")

    _validate_axes(axes)

    nulambda_axis, ei_axis, epsilon_axis, phi_axis, theta_axis, zeta_axis = (
        axes[label] for label in _EXPECTED_LABELS)

    npix = nulambda_axis.nbins
    nEi = ei_axis.nbins
    nEps = epsilon_axis.nbins
    nZeta = zeta_axis.nbins

    pol_convention = zeta_axis.convention

    # Photon directions (spacecraft frame) for every NuLambda pixel, and
    # their (lon, colatitude) in radians -- colatitude because that is the
    # convention UnpolarizedNFFarFieldInstrumentResponseFunction._get_context
    # / ._get_source use internally (lat -> -lat + pi/2).
    nulambda_dir = nulambda_axis.pix2skycoord(np.arange(npix))
    nulambda_lon_rad = nulambda_dir.lon.rad
    nulambda_colat_rad = np.pi / 2 - np.clip(nulambda_dir.lat.rad, -np.pi / 2, np.pi / 2)

    ei_centers_keV = _to_value(ei_axis.centers, u.keV)
    epsilon_centers = _to_value(epsilon_axis.centers)
    phi_centers_rad = _to_value(phi_axis.centers, u.rad)
    theta_centers_rad = _to_value(theta_axis.centers, u.rad)
    zeta_centers_rad = _to_value(zeta_axis.centers, u.rad)

    phase_space_cds = _phase_space_cds(phi_axis, theta_axis, zeta_axis)  # (nPhi, nTheta, nZeta)
    phase_space_em = _phase_space_em(ei_axis, epsilon_axis)  # (nEi, nEpsilon)

    # (Phi, Theta) pairs with at least one physical Zeta bin. Physicality
    # (Phi + Theta in [0, pi]) does not depend on Zeta, so this is constant
    # across the Zeta axis.
    valid_phi_theta = np.argwhere(np.any(phase_space_cds > 0, axis=-1))
    n_valid_pairs = len(valid_phi_theta)

    logger.info(f"{n_valid_pairs}/{phi_axis.nbins * theta_axis.nbins} (Phi, Theta) bin pairs are physical; "
               f"evaluating {n_valid_pairs * npix * nEi * nEps * nZeta:.4g} "
               f"(NuLambda, Ei, Epsilon, Phi, Theta, Zeta) bins in total.")

    contents = np.zeros(axes.nbins, dtype=dtype)

    # Keep a single compute pool alive for every chunked call below, instead
    # of letting evaluate_effective_area()/evaluate_density() spin up (and
    # tear down) a temporary one on every call.
    nf_response.init_compute_pool(list(devices))
    try:
        # --- Step 1: total effective area on the (NuLambda, Ei) grid. It
        # only depends on those two axes, so it is evaluated once,
        # independently of the (Epsilon, Phi, Theta, Zeta) loop below.
        ipix_grid, iei_grid = np.meshgrid(np.arange(npix), np.arange(nEi), indexing='ij')
        ipix_flat = ipix_grid.ravel()
        iei_flat = iei_grid.ravel()

        tot_aeff = np.empty(npix * nEi, dtype=float)
        for start in range(0, len(ipix_flat), grid_batch_size):
            end = min(start + grid_batch_size, len(ipix_flat))
            sl = slice(start, end)
            context = torch.stack([
                torch.as_tensor(nulambda_lon_rad[ipix_flat[sl]], dtype=torch.float32),
                torch.as_tensor(nulambda_colat_rad[ipix_flat[sl]], dtype=torch.float32),
                torch.as_tensor(ei_centers_keV[iei_flat[sl]], dtype=torch.float32),
            ], dim=1)
            tot_aeff[sl] = np.asarray(nf_response.evaluate_effective_area(context))
        tot_aeff = tot_aeff.reshape(npix, nEi)

        # --- Step 2: differential effective area (density) on the full 6D
        # grid, restricted to physical (Phi, Theta) pairs, processed in
        # grid_batch_size-sized chunks so memory use stays bounded
        # regardless of how large the grid is.
        inner_shape = (npix, nEi, nEps, nZeta)
        inner_size = int(np.prod(inner_shape))
        n_valid = n_valid_pairs * inner_size

        iterator = range(0, n_valid, grid_batch_size)
        if show_progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, total=-(-n_valid // grid_batch_size), desc="Filling relative hist IRF", unit="chunk")

        for start in iterator:
            end = min(start + grid_batch_size, n_valid)

            lin = np.arange(start, end)
            iK, ipix, iEi, iEps, iZeta = np.unravel_index(lin, (n_valid_pairs,) + inner_shape)
            iPhi = valid_phi_theta[iK, 0]
            iTheta = valid_phi_theta[iK, 1]

            # Photon (context): direction/energy at NuLambda, Ei bin centers.
            context = torch.stack([
                torch.as_tensor(nulambda_lon_rad[ipix], dtype=torch.float32),
                torch.as_tensor(nulambda_colat_rad[ipix], dtype=torch.float32),
                torch.as_tensor(ei_centers_keV[iEi], dtype=torch.float32),
            ], dim=1)

            # Event (source): measured energy, kinematic scattering angle,
            # and scattered direction (psichi) at the bin center.
            em_keV = ei_centers_keV[iEi] * (1 + epsilon_centers[iEps])
            phi_kin_rad = phi_centers_rad[iPhi]

            phi_geo_rad = np.clip(phi_centers_rad[iPhi] + theta_centers_rad[iTheta], 0, np.pi)
            az_rad = zeta_centers_rad[iZeta]

            relcoords = RelativeCDSCoordinates(nulambda_dir[ipix], pol_convention)
            psichi = relcoords.to_cds(phi_geo_rad, az_rad)

            psichi_lon_rad = psichi.lon.rad
            psichi_colat_rad = np.pi / 2 - np.clip(psichi.lat.rad, -np.pi / 2, np.pi / 2)

            source = torch.stack([
                torch.as_tensor(em_keV, dtype=torch.float32),
                torch.as_tensor(phi_kin_rad, dtype=torch.float32),
                torch.as_tensor(psichi_lon_rad, dtype=torch.float32),
                torch.as_tensor(psichi_colat_rad, dtype=torch.float32),
            ], dim=1)

            density = np.asarray(nf_response.evaluate_density(context, source))

            bin_content = (tot_aeff[ipix, iEi] * density
                          * phase_space_cds[iPhi, iTheta, iZeta]
                          * phase_space_em[iEi, iEps])

            contents[ipix, iEi, iEps, iPhi, iTheta, iZeta] = bin_content
    finally:
        nf_response.shutdown_compute_pool()

    hist = Histogram(axes, contents=contents, unit=u.cm * u.cm)

    output_path = Path(output_path)
    hist.write(str(output_path), name="IRF", overwrite=overwrite)

    return output_path


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    from cosipy.util import fetch_wasabi_file

    data_path = Path("./")

    # NF response
    nf_response_path = data_path / "unpolarized_nfresponse_v1-01.pt"
    fetch_wasabi_file('COSI-SMEX/DC4/Data/Responses/unpolarized_nfresponse_v1-01.pt',
                      output=str(nf_response_path),
                      checksum='bf2d0c16eac5954fb56489480c2602ca')

    nf_response = NFResponse(
        path_to_model=nf_response_path,
        area_batch_size=300_000,
        density_batch_size=100_000,
        devices=["cpu"],
        area_compile_mode=None,
        density_compile_mode=None,
        show_progress=True)

    # --- Option A: reuse the axes of an existing histogram-based response file ---
    template_zip_path = data_path / "ResponseContinuum.area.relative.nonsparse.h5.zip"
    template_path = data_path / "ResponseContinuum.area.relative.nonsparse.h5"
    fetch_wasabi_file('COSI-SMEX/develop/Data/Responses/ResponseContinuum.area.relative.nonsparse.h5.zip',
                      output=str(template_zip_path),
                      unzip=True,
                      checksum='7c082917a3bcb22d6f7116b0a6831007')

    axes = axes_from_h5_template(template_path)

    # --- Option B: build a custom (typically coarser) grid instead ---
    # import astropy.units as u
    # from histpy import Axis, Axes, HealpixAxis
    # from scoords import SpacecraftFrame
    # from cosipy.polarization import PolarizationAxis, StereographicConvention
    #
    # axes = Axes([
    #     HealpixAxis(nside=8, scheme='ring', coordsys=SpacecraftFrame(), label='NuLambda'),
    #     Axis(np.geomspace(100, 10000, 11) * u.keV, label='Ei', scale='log'),
    #     Axis(np.linspace(-1, 1, 21), label='Epsilon'),
    #     Axis(np.linspace(0, 180, 31) * u.deg, label='Phi'),
    #     Axis(np.linspace(-180, 180, 31) * u.deg, label='Theta'),
    #     PolarizationAxis(np.linspace(0, 360, 31) * u.deg, convention=StereographicConvention(), label='Zeta'),
    # ])

    output_path = build_relative_hist_irf(
        nf_response,
        axes,
        output_path=data_path / "relative_hist_irf_from_nf_response.h5",
        devices=["cpu"],
        grid_batch_size=200_000,
        overwrite=True)

    print(f"Wrote {output_path}")

    # Sanity check: load it back and compare against the NF response directly
    from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized
    from cosipy.response.ml.nf_instrument_response_function import UnpolarizedNFFarFieldInstrumentResponseFunction
    from cosipy.response.photon_types import PhotonWithDirectionAndEnergyInSCFrame

    hist_irf = IRFRelativeHistUnpolarized.from_h5(output_path)
    nf_irf = UnpolarizedNFFarFieldInstrumentResponseFunction(nf_response)

    test_photon = PhotonWithDirectionAndEnergyInSCFrame(0.3, 0.2, 511)
    print("Effective area (hist):", hist_irf.effective_area_cm2(test_photon))
    print("Effective area (NF):  ", nf_irf.effective_area_cm2(test_photon))
