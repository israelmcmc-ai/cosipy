"""
Hierarchical cache of an IRF broadened over energy cells, to fold a spectrum with
the same energy nodes for the density of each event and for the expected counts.

The spectrum is written as ``F = w g``, with a fixed weight ``w = E^weight_index``,
and ``g`` is interpolated with hat functions linear in ``ln E`` between nodes:
``g ~ sum_n g(E_n) h_n(E)``. Then, for any event ``x`` and photon direction,

    int F(E) Aeff(E) p(x|E) dE ~ sum_n g(E_n) K_n(x),   K_n(x) = int h_n w Aeff p(x|E) dE.

The nodes are the edges of energy cells. Cells are equal in ``log10 E``: level 0 has
``cells_per_decade`` cells per decade over ``energy_range_keV``, and each level splits
every cell in two. In a cell ``[a, b]`` the hats of its two edges are the rising piece
``R`` (``t = ln(E/a) / ln(b/a)``, of the node ``b``) and the falling piece ``L``
(``1 - t``, of the node ``a``), so the kernel of a node is the ``R`` piece of the cell
below it plus the ``L`` piece of the cell above it. A grid is any set of cells that
tiles the energy range, of any levels.

The pieces of a cell are exact weighted sums of its children's: with ``1`` and ``2``
the lower and upper children, ``R = R1/2 + L2/2 + R2`` and ``L = L1 + R1/2 + L2/2``.
So a coarse cell can be built from finer ones that are already computed, which is much
cheaper than computing it from the IRF.

Each piece is stored as an IRF in the same format as :class:`IRFRelativeHistUnpolarized`
(contents per bin in cm2, plus a separate total effective area), with a single ``Ei``
bin, the cell, and ``Epsilon' = Em / a - 1`` instead of ``Epsilon``. It is evaluated as
an :class:`IRFRelativeHistUnpolarized` at the photon energy ``a``, which normalizes it
and computes its selected effective area from the same table. The integral of each
piece's differential effective area over the selected data space is then exactly its
selected effective area, so a folding that uses the pieces for both the density and
the expected counts is consistent for any grid: the error of the interpolation of ``g``
only changes the shape of the model, not its normalization.

HDF5 layout, under ``/BROADENED`` (the attributes describe how it was built, and are
checked when the cache is opened again)::

    BROADENED/                attrs: energy_range_keV, cells_per_decade, weight_index,
                                     energy_degree (1: hats linear in ln E),
                                     direction_degree (1: linear between pixel centers),
                                     steps_per_cell, max_epsilon_step, quadrature_step,
                                     irf_id, version
        axes/                 Epsilon, Phi, Theta, Zeta axes of the IRF (the floor of the
                              Epsilon' binning and the CDS axes of every piece)
        e{level}/c{index}/point/
                              attrs: energy_range_keV (the cell)
            R/, L/            IRF histograms [NuLambda, Ei (the cell), Epsilon', Phi,
                              Theta, Zeta], one chunk per NuLambda pixel
            AEFF_R/, AEFF_L/  total effective area histograms [NuLambda, Ei], in cm2 keV
            computed          bool per NuLambda pixel, set after the rows are written
        hp{order}/            reserved for extended sources (pixel-integrated kernels)

Pixels are computed on demand and the rest of each dataset is left unallocated.
"""

import hashlib
import os
from pathlib import Path
from typing import Iterable, Tuple, Union

import h5py as h5
import numpy as np
from astropy import units as u
from histpy import Axes, Axis, Histogram

from cosipy.event_selection.energy_selection import EnergySelector
from cosipy.polarization import PolarizationAxis
from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized

ROOT = 'BROADENED'
VERSION = 1
PIECES = ('R', 'L')


class BroadenedIRFCache:
    """
    Broadened pieces of the cells of an :class:`IRFRelativeHistUnpolarized`, computed
    on demand and stored in an HDF5 file. See the module docstring.

    Parameters
    ----------
    irf : IRFRelativeHistUnpolarized
        The response. Its selections, if any, are not used: they are applied when the
        pieces are evaluated (see :meth:`piece_irfs`).
    filename : str or path-like
        HDF5 file of the cache. It can be the file of the IRF. If it already has a
        cache, it must have been built with the same parameters and IRF.
    energy_range_keV : tuple of float
        Energy range of the integral over the photon energy, and of the level-0 cells.
    cells_per_decade : int
        Number of level-0 cells per decade. The range must be a whole number of them.
    weight_index : float
        Index of the weight ``w = E^weight_index``, ideally close to the spectra to fit.
    steps_per_cell : int
        The ``Epsilon'`` axis of each cell has bins uniform in ``ln(1 + Epsilon')``, of
        width ``ln(b/a) / steps_per_cell``, over the broadened photopeak, plus the
        edges of the ``Epsilon`` axis of the IRF everywhere (see :meth:`epsilon_edges`).
    max_epsilon_step : float
        Maximum width of those bins. The error of the kernels decreases as the square of
        the width: with the default, 1/64 of a decade, it is ~0.5%.
    quadrature_step : float, optional
        Step in ``ln E`` of the midpoint rule of the integral over the photon energy. By
        default, a quarter of the narrowest ``Epsilon`` bin of the IRF (in
        ``ln(1 + Epsilon)``), so that the photopeak is resolved.
    max_level : int
        Finest level allowed.
    """

    def __init__(self,
                 irf: IRFRelativeHistUnpolarized,
                 filename: Union[str, Path],
                 energy_range_keV: Tuple[float, float] = (100., 10_000.),
                 cells_per_decade: int = 1,
                 weight_index: float = -2.,
                 steps_per_cell: int = 16,
                 max_epsilon_step: float = np.log(10) / 64,
                 quadrature_step: float = None,
                 max_level: int = 6):

        self._irf = irf
        self._filename = Path(filename)
        self._log10_range = np.log10(np.asarray(energy_range_keV, dtype=float))
        self._cells_per_decade = int(cells_per_decade)
        self._weight_index = float(weight_index)
        self._steps_per_cell = int(steps_per_cell)
        self._max_epsilon_step = float(max_epsilon_step)
        self._max_level = int(max_level)

        ncells0 = (self._log10_range[1] - self._log10_range[0]) * self._cells_per_decade
        if not np.isclose(ncells0, np.round(ncells0)) or np.round(ncells0) < 1:
            raise ValueError(f"The energy range {energy_range_keV} keV must be a whole number of "
                             f"level-0 cells ({cells_per_decade} per decade)")
        self._ncells0 = int(np.round(ncells0))

        axes = irf.axes
        self._epsilon_edges = np.asarray(axes['Epsilon'].edges, dtype=float)

        if quadrature_step is None:
            quadrature_step = np.min(np.diff(np.log1p(self._epsilon_edges[self._epsilon_edges > -1]))) / 4
        self._quadrature_step = float(quadrature_step)

        self._attrs = {'energy_range_keV': np.asarray(energy_range_keV, dtype=float),
                       'cells_per_decade': self._cells_per_decade,
                       'weight_index': self._weight_index,
                       'energy_degree': 1,
                       'direction_degree': 1,
                       'steps_per_cell': self._steps_per_cell,
                       'max_epsilon_step': self._max_epsilon_step,
                       'quadrature_step': self._quadrature_step,
                       'irf_id': self._irf_id(irf),
                       'version': VERSION}

        self._open_or_create()

    @property
    def energy_range_keV(self) -> np.ndarray:
        """Energy range of the integral over the photon energy, in keV."""
        return 10 ** self._log10_range

    @property
    def weight_index(self) -> float:
        """Index of the weight ``w = (E/keV)^weight_index``."""
        return self._weight_index

    @property
    def max_level(self) -> int:
        """Finest level allowed."""
        return self._max_level

    # Geometry

    def ncells(self, level: int) -> int:
        """Number of cells at ``level``."""

        return self._ncells0 * 2 ** level

    def cell_edges_keV(self, level: int, index: int) -> np.ndarray:
        """Lower and upper edge of a cell, in keV."""

        width = 1 / (self._cells_per_decade * 2 ** level)
        return 10 ** (self._log10_range[0] + width * np.array([index, index + 1]))

    def epsilon_edges(self, level: int, index: int) -> np.ndarray:
        """
        Edges of the ``Epsilon' = Em / a - 1`` axis of a cell ``[a, b]``: those of the
        IRF's ``Epsilon`` axis, plus a uniform grid in ``ln(1 + Epsilon')`` from one cell
        width below the photopeak to the highest reachable value,
        ``ln(b/a) + ln(1 + eps_max)``. Its step is ``ln(b/a) / steps_per_cell``, but at
        most ``max_epsilon_step``: through the photopeak (``Em ~ E``), the kernel follows
        the variation of the IRF with ``Ei``, whatever the width of the cell.
        """

        lo, hi = self.cell_edges_keV(level, index)
        width = np.log(hi / lo)
        step = min(width / self._steps_per_cell, self._max_epsilon_step)
        u_max = width + np.log1p(self._epsilon_edges[-1])

        k = np.arange(np.floor(-width / step), np.ceil(u_max / step) + 1)
        edges = np.concatenate([self._epsilon_edges, np.expm1(k * step), [np.expm1(u_max)]])
        edges = np.unique(edges[(edges >= self._epsilon_edges[0]) & (edges <= np.expm1(u_max))])

        # Drop edges that only differ by rounding
        return edges[np.concatenate([[True], np.diff(edges) > 1e-9 * np.maximum(1, np.abs(edges[1:]))])]

    # File

    @staticmethod
    def _irf_id(irf: IRFRelativeHistUnpolarized) -> str:
        """Hash of the axes of the response, to detect a cache built from another one."""

        digest = hashlib.sha1()
        for axis in irf.axes:
            digest.update(axis.label.encode())
            digest.update(np.ascontiguousarray(axis.edges if axis.label != 'NuLambda' else [axis.nside],
                                               dtype=float).tobytes())
        return digest.hexdigest()

    def _open_or_create(self):

        with h5.File(self._filename, 'a') as f:
            if ROOT not in f:
                root = f.create_group(ROOT)
                for key, value in self._attrs.items():
                    root.attrs[key] = value
                self._write_axes(root.create_group('axes'))
                return

            root = f[ROOT]
            for key, value in self._attrs.items():
                stored = root.attrs.get(key)
                same = (np.allclose(stored, value) if isinstance(value, (float, np.ndarray))
                        else stored == value)
                if not same:
                    raise ValueError(f"The cache in {self._filename} has {key} = {stored}, "
                                     f"but {value} was requested")

    def _write_axes(self, group):
        axes = self._irf.axes
        Axes(self._cds_axes(Axis(self._epsilon_edges, label='Epsilon'))).write(group)

    def _cds_axes(self, epsilon_axis: Axis):
        """The ``Epsilon``-like, ``Phi``, ``Theta`` and ``Zeta`` axes of a piece, with units."""

        axes = self._irf.axes
        return [epsilon_axis,
                Axis(np.asarray(axes['Phi'].edges) * u.rad, label='Phi'),
                Axis(np.asarray(axes['Theta'].edges) * u.rad, label='Theta'),
                PolarizationAxis(np.asarray(axes['Zeta'].edges) * u.rad, convention=self._irf._pol_convention,
                                 label='Zeta')]

    def _cell_path(self, level: int, index: int) -> str:
        return f'{ROOT}/e{level}/c{index}/point'

    def _create_cell(self, f, level: int, index: int):
        """Empty (unallocated) datasets of a cell."""

        path = self._cell_path(level, index)
        group = f.require_group(path)
        lo, hi = self.cell_edges_keV(level, index)
        group.attrs['energy_range_keV'] = [lo, hi]

        nulambda = self._irf.axes['NuLambda']
        ei_axis = Axis([lo, hi] * u.keV, label='Ei', scale='log')
        irf_axes = Axes([nulambda, ei_axis] + self._cds_axes(Axis(self.epsilon_edges(level, index), label='Epsilon')))
        aeff_axes = Axes([nulambda, ei_axis])

        for piece in PIECES:
            for name, axes, dtype in ((piece, irf_axes, np.float32), (f'AEFF_{piece}', aeff_axes, np.float64)):
                g = group.create_group(name)
                g.attrs['unit'] = 'cm2'
                g.attrs['format'] = 'dense'
                axes.write(g.create_group('axes'))
                shape = tuple(axes.nbins)
                g.create_dataset('contents', shape=shape, dtype=dtype, fillvalue=0, track_times=False,
                                 chunks=(1,) + shape[1:])

        group.create_dataset('computed', shape=(nulambda.nbins,), dtype=bool, fillvalue=False, track_times=False)

    def computed(self, level: int, index: int) -> np.ndarray:
        """Whether each ``NuLambda`` pixel of a cell has been computed."""

        with h5.File(self._filename, 'r') as f:
            path = self._cell_path(level, index)
            if path not in f:
                return np.zeros(self._irf.axes['NuLambda'].nbins, dtype=bool)
            return f[path]['computed'][:]

    def _write_rows(self, level: int, index: int, pixels: np.ndarray, rows: dict):
        """
        Write the rows of ``pixels``, and only then mark them as computed, so that an
        interrupted write leaves them as not computed. One writer at a time.
        """

        lock = Path(str(self._filename) + '.lock')
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"{lock} exists: another process is writing to the cache, or one was "
                               f"interrupted while writing (then delete it)")

        try:
            with h5.File(self._filename, 'a') as f:
                path = self._cell_path(level, index)
                if path not in f:
                    self._create_cell(f, level, index)
                group = f[path]

                for name, values in rows.items():
                    dataset = group[name]['contents']
                    for pixel, value in zip(pixels, values):
                        dataset[pixel] = value

                f.flush()
                group['computed'][np.sort(pixels)] = True
        finally:
            os.close(fd)
            lock.unlink()

    def _read_rows(self, level: int, index: int, pixels: np.ndarray) -> dict:

        with h5.File(self._filename, 'r') as f:
            group = f[self._cell_path(level, index)]
            return {name: np.stack([group[name]['contents'][pixel] for pixel in pixels])
                    for name in ('R', 'L', 'AEFF_R', 'AEFF_L')}

    # Computation

    def ensure(self, cells: Iterable[Tuple[int, int]], pixels: Iterable[int]):
        """
        Compute the ``pixels`` of each ``(level, index)`` cell that are not in the cache
        yet: from its children if they are both computed for those pixels, else from
        the IRF.
        """

        pixels = np.unique(np.asarray(list(pixels), dtype=int))

        for level, index in cells:
            if not 0 <= level <= self._max_level or not 0 <= index < self.ncells(level):
                raise ValueError(f"No cell ({level}, {index})")

            missing = pixels[~self.computed(level, index)[pixels]]
            if len(missing) == 0:
                continue

            if level < self._max_level:
                from_children = (self.computed(level + 1, 2 * index)[missing]
                                 & self.computed(level + 1, 2 * index + 1)[missing])
            else:
                from_children = np.zeros(len(missing), dtype=bool)

            if from_children.any():
                self._write_rows(level, index, missing[from_children],
                                 self._merge_children(level, index, missing[from_children]))
            if (~from_children).any():
                self._write_rows(level, index, missing[~from_children],
                                 self._direct(level, index, missing[~from_children]))

    def precompute(self, level: int, pixels: Iterable[int] = None):
        """
        Compute all the cells of ``level`` from the IRF, and every coarser level from
        them, for ``pixels`` (default: all).
        """

        if pixels is None:
            pixels = np.arange(self._irf.axes['NuLambda'].nbins)

        for lev in range(level, -1, -1):
            self.ensure([(lev, i) for i in range(self.ncells(lev))], pixels)

    def _direct(self, level: int, index: int, pixels: np.ndarray) -> dict:
        """
        The pieces of a cell for ``pixels``, from the IRF.

        For each pixel, the normalized table of the IRF (per unit ``Epsilon``, at the
        ``Ei``, ``Phi``, ``Theta`` and ``Zeta`` centers) is contracted with weights that
        integrate, with a midpoint rule in ``ln E``, the hat times ``w Aeff`` over the
        cell, and exactly, through the ``Epsilon`` hats of the IRF, the interpolated
        density over each ``Epsilon'`` bin. The result is the density at the CDS centers
        integrated over each ``Epsilon'`` bin, which is multiplied by the phase space of
        the CDS bins to store contents per bin, as the IRF format expects.
        """

        irf = self._irf
        axes = irf.axes
        lo, hi = self.cell_edges_keV(level, index)
        eps_edges = self.epsilon_edges(level, index)

        width = np.log(hi / lo)
        nq = max(int(np.ceil(width / self._quadrature_step)), 1)
        t = (np.arange(nq) + 0.5) / nq
        energy = lo * np.exp(t * width)
        de_w = (width / nq) * energy * energy ** self._weight_index  # dE w(E)
        hats = {'R': t, 'L': 1 - t}

        # Ei interpolation weights of each quadrature energy, as a dense (nq, nEi) matrix
        ei_bins, ei_weights = (np.asarray(a) for a in axes['Ei'].interp_weights(energy))
        ei_matrix = np.zeros((nq, axes['Ei'].nbins))
        for b, w in zip(ei_bins, ei_weights):
            np.add.at(ei_matrix, (np.arange(nq), b), w)

        # Integral of each Epsilon hat over each Epsilon' bin: Epsilon = a (1 + Epsilon') / E - 1
        cumulative = irf._epsilon_hats.cumulative(lo * (1 + eps_edges)[None, :] / energy[:, None] - 1)
        hat_bins = np.diff(cumulative, axis=1)  # (nq, nEps', nEps)

        nulambda = axes['NuLambda']
        phase_space = np.nan_to_num(np.asarray(irf._phase_space_cds, dtype=float))
        rows = {name: [] for name in ('R', 'L', 'AEFF_R', 'AEFF_L')}

        for pixel in pixels:
            table = irf._pixel_table(pixel)  # (nEi, nEps, nPhi, nTheta, nZeta)
            aeff = irf._effective_area_at_pixel(pixel, energy)

            for piece in PIECES:
                f = hats[piece] * de_w * aeff
                weights = np.einsum('q,qk,qji->jki', f, ei_matrix, hat_bins)
                density = weights.reshape(len(eps_edges) - 1, -1) @ table.reshape(weights[0].size, -1)
                rows[piece].append((density.reshape((-1,) + phase_space.shape) * phase_space)[None])
                rows[f'AEFF_{piece}'].append([f.sum()])

        return {name: np.asarray(values) for name, values in rows.items()}

    def _merge_children(self, level: int, index: int, pixels: np.ndarray) -> dict:
        """
        The pieces of a cell from its two children: ``R = R1/2 + L2/2 + R2`` and
        ``L = L1 + R1/2 + L2/2``. The tables are rebinned to the cell's ``Epsilon'``
        axis, conserving their contents and assuming a uniform density within each
        child bin.
        """

        lo = self.cell_edges_keV(level, index)[0]
        parent_edges = self.epsilon_edges(level, index)
        children = []

        for child in (2 * index, 2 * index + 1):
            rows = self._read_rows(level + 1, child, pixels)
            child_lo = self.cell_edges_keV(level + 1, child)[0]
            rebin = self._rebin_matrix(child_lo / lo * (1 + self.epsilon_edges(level + 1, child)) - 1, parent_edges)
            for piece in PIECES:
                rows[piece] = np.einsum('mj,pej...->pem...', rebin, rows[piece].astype(float))
            children.append(rows)

        c1, c2 = children
        return {'R': c1['R'] / 2 + c2['L'] / 2 + c2['R'],
                'L': c1['L'] + c1['R'] / 2 + c2['L'] / 2,
                'AEFF_R': c1['AEFF_R'] / 2 + c2['AEFF_L'] / 2 + c2['AEFF_R'],
                'AEFF_L': c1['AEFF_L'] + c1['AEFF_R'] / 2 + c2['AEFF_L'] / 2}

    @staticmethod
    def _rebin_matrix(source_edges: np.ndarray, target_edges: np.ndarray) -> np.ndarray:
        """
        ``M[m, j]``: fraction of source bin ``j`` that falls in target bin ``m``.
        """

        lo = np.maximum(source_edges[None, :-1], target_edges[:-1, None])
        hi = np.minimum(source_edges[None, 1:], target_edges[1:, None])

        return np.clip(hi - lo, 0, None) / np.diff(source_edges)[None, :]

    # Evaluation

    def piece_irfs(self, level: int, index: int, selections=None) -> Tuple[IRFRelativeHistUnpolarized,
                                                                          IRFRelativeHistUnpolarized]:
        """
        The ``R`` and ``L`` pieces of a cell as IRFs, read lazily from the cache, to be
        evaluated with a photon energy equal to the lower edge of the cell (see
        :meth:`cell_edges_keV`). Only its computed pixels have meaningful values, so
        :meth:`ensure` must have been called for the pixels that will be evaluated.

        Parameters
        ----------
        selections : EnergySelector or tuple thereof, optional
            Measured-energy selection, as for :class:`IRFRelativeHistUnpolarized`.
        """

        path = self._cell_path(level, index)

        return tuple(IRFRelativeHistUnpolarized(self._filename,
                                                aeff=Histogram.open(self._filename, f'{path}/AEFF_{piece}'),
                                                group=f'{path}/{piece}',
                                                selections=selections)
                     for piece in PIECES)
