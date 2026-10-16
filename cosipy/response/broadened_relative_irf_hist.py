"""
Broadened version of an :class:`IRFRelativeHistUnpolarized`, for folding a
spectrum with the same ``Ei`` nodes for the per-event density and the total
expected counts.

The spectrum ``F`` is written as ``F = w * (F/w)``, with a fixed weight
``w ~ E^weight_index``, and ``F/w`` is interpolated linearly in ``ln(E)``
between the nodes ``E_n``: ``F/w ~ sum_n (F/w)(E_n) h_n(E)``, with hat
functions ``h_n``. Then, for any event ``x`` and photon direction,

    int F(E) Aeff(E) p(x|E) dE ~ sum_n F(E_n) Delta_n A_n p_n(x)

with ``Delta_n = int h_n(E) dE``, the broadened effective area

    A_n = 1 / Delta_n int h_n(E) w(E) / w(E_n) Aeff(E) dE

and the broadened event probability ``p_n(x)``, proportional to
``int h_n w Aeff p(x|E) dE``. ``p_n`` is stored as an IRF histogram whose
``Ei`` axis centers are the nodes, in the same format as the original
(contents per bin, in cm2), with ``Epsilon = Em / E_n - 1``. When it is
loaded as an :class:`IRFRelativeHistUnpolarized` it is normalized like any
other IRF, so its density integrates over the data space, inside any energy
selection, to exactly the selected area that the same IRF computes. A folding
that uses it at the nodes for both the density and the expected counts is then
consistent for any number of nodes: the error of the ``F/w`` interpolation only
changes the shape of the model, not its normalization.
"""

import h5py as h5
import numpy as np
from astropy import units as u
from histpy import Axes, Axis
from scipy import sparse

from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.response.relative_coordinates import RelativeCDSCoordinates

BROADENED_IRF_GROUP = 'BROADENED_IRF'
BROADENED_AEFF_GROUP = 'BROADENED_AEFF'


def hat_functions(nodes_keV, energy_keV):
    """
    Hat functions, linear in ``ln(E)``, of each node evaluated at each
    energy. The first and last ones are one-sided: they are zero outside of
    the nodes' range.

    Returns
    -------
    numpy.ndarray, shape (len(energy_keV), len(nodes_keV))
    """

    log_nodes = np.log(np.asarray(nodes_keV, dtype=float))
    log_e = np.log(np.asarray(energy_keV, dtype=float))
    hats = np.zeros((log_e.size, log_nodes.size))

    inside = (log_e >= log_nodes[0]) & (log_e <= log_nodes[-1])
    j = np.clip(np.searchsorted(log_nodes, log_e, side='right') - 1, 0, log_nodes.size - 2)
    t = (log_e - log_nodes[j]) / (log_nodes[j + 1] - log_nodes[j])
    rows = np.nonzero(inside)[0]
    hats[rows, j[rows]] = 1 - t[rows]
    hats[rows, j[rows] + 1] += t[rows]

    return hats


def hat_integrals_keV(nodes_keV):
    """
    ``Delta_n``: integral over ``E`` of the hat function of each node (see
    :func:`hat_functions`), in keV.
    """

    nodes = np.asarray(nodes_keV, dtype=float)
    delta = np.zeros(nodes.size)

    # int_0^L (1 - s/L) e^s ds and int_0^L (s/L) e^s ds, with s = ln(E/E_j)
    log_step = np.diff(np.log(nodes))
    falling = (np.expm1(log_step) - log_step) / log_step
    rising = (log_step * np.exp(log_step) - np.expm1(log_step)) / log_step
    delta[:-1] += nodes[:-1] * falling
    delta[1:] += nodes[:-1] * rising

    return delta


def broadened_epsilon_edges(log_step, epsilon_max, steps_per_node=4, log_min=-1.):
    """
    Edges of the ``Epsilon = Em / E_n - 1`` axis of the broadened IRF.

    The broadened photopeak spans about one node step at each side of
    ``Em = E_n``, so the edges are uniform in ``ln(1 + Epsilon)``, with
    ``steps_per_node`` bins per node step, from ``log_min`` up to the largest
    reachable value, ``ln(1 + epsilon_max) + log_step``. Below that there are
    two coarse bins down to ``Epsilon = -1``, for the Compton continuum.
    """

    step = log_step / steps_per_node
    j = np.arange(np.floor(log_min / step - 0.5), np.ceil((np.log1p(epsilon_max) + log_step) / step - 0.5) + 1)
    fine = np.expm1((j + 0.5) * step)

    return np.concatenate([[-1., 0.5 * (fine[0] - 1)], fine])


def _broadened_aeff(irf, nodes, weight_index, nulambda, pixel_batch=32):
    """
    ``A_n = 1 / Delta_n int h_n(E) (E / E_n)^weight_index Aeff(E) dE`` at the
    center of each pixel of ``nulambda``, shape (npix, nnodes).
    """

    log_step = np.log(nodes[1] / nodes[0])
    sub = np.linspace(-1, 1, 1025)
    energy = np.unique(np.concatenate([n * np.exp(sub * log_step) for n in nodes]))
    energy = energy[(energy >= nodes[0]) & (energy <= nodes[-1])]
    trapz = np.zeros(energy.size)
    trapz[:-1] += np.diff(energy) / 2
    trapz[1:] += np.diff(energy) / 2
    weight = (trapz[:, None] * hat_functions(nodes, energy) * (energy[:, None] / nodes[None, :]) ** weight_index
              / hat_integrals_keV(nodes)[None, :])

    broadened = np.zeros((nulambda.nbins, nodes.size))
    for p0 in range(0, nulambda.nbins, pixel_batch):
        pix = np.arange(p0, min(p0 + pixel_batch, nulambda.nbins))
        lon, lat = (np.deg2rad(np.asarray(a)) for a in nulambda.pix2ang(pix, lonlat=True))
        photons = PhotonListWithDirectionAndEnergyInSCFrame(np.repeat(lon, energy.size), np.repeat(lat, energy.size),
                                                            np.tile(energy, pix.size))
        aeff = np.asarray(irf._effective_area_cm2_unselected(photons), dtype=float).reshape(pix.size, energy.size)
        broadened[pix] = aeff @ weight

    return broadened


def write_broadened_irf(filename,
                        nodes_per_decade=6,
                        energy_range_keV=(100., 10_000.),
                        weight_index=-2.,
                        steps_per_node=4,
                        nsub=16,
                        pixel_batch=8,
                        dtype=np.float32,
                        overwrite=False):
    """
    Compute the broadened IRF from the ``IRF`` (and ``AEFF``, if present)
    groups of an HDF5 file, and add it to the same file as the groups
    ``BROADENED_IRF`` and ``BROADENED_AEFF``. Load it with
    ``IRFRelativeHistUnpolarized.from_h5(filename, broadened=True)``.

    The nodes are log-spaced over ``energy_range_keV``, which is also the
    range of the integral over ``Ei``. ``BROADENED_AEFF`` has the
    ``NuLambda`` axis of ``AEFF`` (or ``IRF``), and ``BROADENED_IRF`` the
    ``NuLambda``, ``Phi``, ``Theta`` and ``Zeta`` axes of ``IRF``.

    The broadened probability is evaluated at the centers of its bins, from
    the linearly interpolated original (as :class:`IRFRelativeHistUnpolarized`
    evaluates it), and the integral over ``Ei`` is done, at fixed ``Em``, over
    the original ``Epsilon``, split at every kink of the integrand (``Epsilon``
    and ``Ei`` centers, the node) and in ``nsub`` sub-intervals in between.

    Parameters
    ----------
    filename : str or path-like
        HDF5 file with the ``IRF`` group, and optionally ``AEFF``.
    nodes_per_decade : int
        Number of nodes per decade of energy.
    energy_range_keV : tuple of float
        First and last node.
    weight_index : float
        Index of the weight ``w = E^weight_index``.
    steps_per_node : int
        Bins of the broadened ``Epsilon`` axis per node step (see
        :func:`broadened_epsilon_edges`).
    nsub : int
        Sub-intervals between kinks for the integral over ``Ei``.
    pixel_batch : int
        Number of ``NuLambda`` pixels computed at a time.
    dtype : numpy dtype
        Of the stored contents. Single precision halves the size, and it
        doesn't affect the consistency between the density and the selected
        area, since both are computed from the same table when loaded.
    overwrite : bool
        Whether to replace existing broadened groups.
    """

    from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized

    with h5.File(filename, 'r') as f:
        orig_axes = Axes.open(f['IRF/axes'])
        aeff_axes = Axes.open(f['AEFF/axes']) if 'AEFF' in f else None
        existing = [g for g in (BROADENED_IRF_GROUP, BROADENED_AEFF_GROUP) if g in f]

    if existing and not overwrite:
        raise ValueError(f"{filename} already has {existing}. Use overwrite=True to replace them.")

    irf = IRFRelativeHistUnpolarized.from_h5(filename)
    table = irf._event_prob
    axes = table.axes
    ei_axis, eps_axis = axes['Ei'], axes['Epsilon']

    emin, emax = energy_range_keV
    nnodes = int(round(np.log10(emax / emin) * nodes_per_decade)) + 1
    nodes = np.geomspace(emin, emax, nnodes)
    log_step = np.log(nodes[1] / nodes[0])

    new_ei_edges = np.concatenate([nodes * np.exp(-log_step / 2), nodes[-1:] * np.exp(log_step / 2)])
    new_ei_axis = Axis(new_ei_edges * u.keV, label='Ei', scale='log')
    nodes = np.asarray(new_ei_axis.centers, dtype=float)
    w_nodes = nodes ** weight_index

    eps_edges_orig = np.asarray(eps_axis.edges, dtype=float)
    new_eps_axis = Axis(broadened_epsilon_edges(log_step, eps_edges_orig[-1], steps_per_node), label='Epsilon')
    new_eps_centers = np.asarray(new_eps_axis.centers, dtype=float)
    new_eps_widths = np.asarray(new_eps_axis.widths, dtype=float)

    kinks_eps = np.asarray(eps_axis.centers, dtype=float)
    kinks_ei = np.asarray(ei_axis.centers, dtype=float)

    # Quadrature nodes q, in the original Epsilon, of each (node n, broadened Epsilon center c)
    q_combo, q_e, q_eps, q_weight = [], [], [], []
    for n in range(nnodes):
        e_lo, e_hi = nodes[max(n - 1, 0)], nodes[min(n + 1, nnodes - 1)]
        for c, eps_c in enumerate(new_eps_centers):
            em = (1 + eps_c) * nodes[n]
            lo = max(em / e_hi - 1, eps_edges_orig[0])
            hi = min(em / e_lo - 1, eps_edges_orig[-1])
            if hi <= lo:
                continue
            breaks = np.concatenate([[lo, hi], kinks_eps, em / kinks_ei - 1, [em / nodes[n] - 1]])
            breaks = np.unique(breaks[(breaks >= lo) & (breaks <= hi)])
            eps = np.unique(np.concatenate([np.linspace(a, b, nsub + 1) for a, b in zip(breaks[:-1], breaks[1:])]))
            trapz = np.zeros(eps.size)
            trapz[:-1] += np.diff(eps) / 2
            trapz[1:] += np.diff(eps) / 2
            energy = em / (1 + eps)
            # dE = E / (1 + eps) dEps, and the table is per unit Epsilon times Ei: p = table / E
            weight = trapz * hat_functions(nodes, energy)[:, n] * (energy ** weight_index / w_nodes[n]) / (1 + eps)
            q_combo.append(np.full(eps.size, n * new_eps_centers.size + c))
            q_e.append(energy)
            q_eps.append(eps)
            q_weight.append(weight)

    q_combo, q_e, q_eps, q_weight = map(np.concatenate, (q_combo, q_e, q_eps, q_weight))
    ncombo = nnodes * new_eps_centers.size

    # Sparse map from quadrature nodes to the (Ei, Epsilon) bins of the original table
    ei_bins, ei_w = (np.asarray(a) for a in ei_axis.interp_weights(q_e))
    eps_bins, eps_w = (np.asarray(a) for a in eps_axis.interp_weights(q_eps))
    n_eps_orig = eps_axis.nbins
    rows, cols, vals = [], [], []
    for a in range(ei_bins.shape[0]):
        for b in range(eps_bins.shape[0]):
            rows.append(np.arange(q_e.size))
            cols.append(q_combo * (ei_axis.nbins * n_eps_orig) + ei_bins[a] * n_eps_orig + eps_bins[b])
            vals.append(q_weight * ei_w[a] * eps_w[b])
    to_bins = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                shape=(q_e.size, ncombo * ei_axis.nbins * n_eps_orig))
    to_bins_t = to_bins.T.tocsr()

    # Phase space of the CDS bins, to store contents per bin as the IRF expects
    phi_m, theta_m, zeta_m = np.meshgrid(axes['Phi'].edges, axes['Theta'].edges, axes['Zeta'].edges, indexing='ij')
    cds_volume = np.nan_to_num(np.asarray(RelativeCDSCoordinates.get_relative_cds_phase_space(
        phi_m[:-1, :-1, :-1], phi_m[1:, :-1, :-1], theta_m[:-1, :-1, :-1], theta_m[:-1, 1:, :-1],
        zeta_m[:-1, :-1, :-1], zeta_m[:-1, :-1, 1:]), dtype=float))
    em_volume = nodes[:, None] * new_eps_widths[None, :]

    # Broadened effective area, on the NuLambda axis of the separate aeff if there is one
    aeff_nulambda = (aeff_axes or orig_axes)['NuLambda']
    broadened_aeff = _broadened_aeff(irf, nodes, weight_index, aeff_nulambda)

    nulambda = axes['NuLambda']
    npix = nulambda.nbins
    cds_shape = table.contents.shape[3:]

    new_irf_axes = Axes([orig_axes['NuLambda'], new_ei_axis, new_eps_axis,
                         orig_axes['Phi'], orig_axes['Theta'], orig_axes['Zeta']])

    def pixel_photons(lon_deg, lat_deg, energy):
        n = lon_deg.size
        return PhotonListWithDirectionAndEnergyInSCFrame(np.repeat(np.deg2rad(lon_deg), energy.size),
                                                          np.repeat(np.deg2rad(lat_deg), energy.size),
                                                          np.tile(energy, n))

    with h5.File(filename, 'a') as f:
        for group in existing:
            del f[group]

        group = f.create_group(BROADENED_IRF_GROUP)
        group.attrs['unit'] = 'cm2'
        group.attrs['format'] = 'dense'
        group.attrs['weight_index'] = weight_index
        new_irf_axes.write(group.create_group('axes'))
        contents = group.create_dataset('contents', shape=tuple(new_irf_axes.nbins), dtype=dtype, track_times=False)

        for p0 in range(0, npix, pixel_batch):
            pix = np.arange(p0, min(p0 + pixel_batch, npix))
            lon, lat = nulambda.pix2ang(pix, lonlat=True)
            aeff = np.asarray(irf._effective_area_cm2_unselected(pixel_photons(np.asarray(lon), np.asarray(lat), q_e)),
                              dtype=float).reshape(pix.size, q_e.size)
            # (ncombo * nEi * nEps, npix_batch)
            weights = (to_bins_t @ aeff.T).T.reshape(pix.size, ncombo, ei_axis.nbins * n_eps_orig)
            density = np.matmul(weights, table.contents[pix].reshape(pix.size, ei_axis.nbins * n_eps_orig, -1))
            density = density.reshape(pix.size, nnodes, new_eps_centers.size, *cds_shape)
            contents[pix[0]:pix[-1] + 1] = (density * em_volume[None, :, :, None, None, None]
                                            * cds_volume[None, None, None])

    new_aeff_axes = Axes([aeff_nulambda, new_ei_axis])
    with h5.File(filename, 'a') as f:
        group = f.create_group(BROADENED_AEFF_GROUP)
        group.attrs['unit'] = 'cm2'
        group.attrs['format'] = 'dense'
        group.attrs['weight_index'] = weight_index
        new_aeff_axes.write(group.create_group('axes'))
        group.create_dataset('contents', data=broadened_aeff, track_times=False)
