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

Each piece is an IRF in the same format as :class:`IRFRelativeHistUnpolarized` (contents
per bin in cm2, plus a total effective area), with a single ``Ei`` bin, the cell, and
``Epsilon' = Em / a - 1`` instead of ``Epsilon``. Its ``NuLambda`` pixels are computed
when an evaluation needs them, and optionally kept in an HDF5 file. It is evaluated as
an :class:`IRFRelativeHistUnpolarized` at the photon energy ``a``, which normalizes it
and computes its selected effective area from the same table. The integral of each
piece's differential effective area over the selected data space is then exactly its
selected effective area, so a folding that uses the pieces for both the density and
the expected counts is consistent for any grid: the error of the interpolation of ``g``
only changes the shape of the model, not its normalization.

HDF5 layout of the optional file, under ``/BROADENED`` (the attributes describe how it
was built, and are checked when the cache is opened again)::

    BROADENED/                attrs: energy_range_keV, cells_per_decade, weight_index,
                                     energy_degree (1: hats linear in ln E),
                                     direction_degree (1: linear between pixel centers),
                                     steps_per_cell, max_epsilon_step,
                                     continuum_epsilon_step, quadrature_step,
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
    filename : str or path-like, optional
        HDF5 file to keep the pieces in. It can be the file of the IRF. If it already
        has a cache, it must have been built with the same parameters and IRF. By
        default (None) the pieces are computed when an evaluation needs them and not
        kept: that is about twice as slow as reading them from the file, but a file
        takes ~45 MB per ``NuLambda`` pixel for a grid of 16 cells.
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
    continuum_epsilon_step : float
        Width of the bins in ``Epsilon'`` below the photopeak (see :meth:`epsilon_edges`).
    quadrature_step : float, optional
        Step in ``ln E`` of the midpoint rule of the integral over the photon energy. By
        default, a quarter of the narrowest ``Epsilon`` bin of the IRF (in
        ``ln(1 + Epsilon)``), so that the photopeak is resolved.
    max_level : int
        Finest level allowed.
    """

    def __init__(self,
                 irf: IRFRelativeHistUnpolarized,
                 filename: Union[str, Path, None] = None,
                 energy_range_keV: Tuple[float, float] = (100., 10_000.),
                 cells_per_decade: int = 1,
                 weight_index: float = -2.,
                 steps_per_cell: int = 16,
                 max_epsilon_step: float = np.log(10) / 64,
                 continuum_epsilon_step: float = 0.05,
                 quadrature_step: float = None,
                 max_level: int = 6):

        self._irf = irf
        self._filename = None if filename is None else Path(filename)
        self._memory = {}  # (level, index, pixel) -> rows not read yet, without a file
        self._log10_range = np.log10(np.asarray(energy_range_keV, dtype=float))
        self._cells_per_decade = int(cells_per_decade)
        self._weight_index = float(weight_index)
        self._steps_per_cell = int(steps_per_cell)
        self._max_epsilon_step = float(max_epsilon_step)
        self._continuum_epsilon_step = float(continuum_epsilon_step)
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
                       'continuum_epsilon_step': self._continuum_epsilon_step,
                       'quadrature_step': self._quadrature_step,
                       'irf_id': self._irf_id(irf),
                       'version': VERSION}

        if self._filename is not None:
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
        Edges of the ``Epsilon' = Em / a - 1`` axis of a cell ``[a, b]``:

        - those of the IRF's ``Epsilon`` axis;
        - over the broadened photopeak, a uniform grid in ``ln(1 + Epsilon')``, from one
          cell width below it to the highest reachable value,
          ``ln(b/a) + ln(1 + eps_max)``. Its step is ``ln(b/a) / steps_per_cell``, but
          at most ``max_epsilon_step``: through the photopeak (``Em ~ E``), the kernel
          follows the variation of the IRF with ``Ei``, whatever the width of the cell;
        - below it, over the Compton continuum, a uniform grid in ``Epsilon'`` of step
          ``continuum_epsilon_step``. The coarse ``Epsilon`` bins of the IRF there are
          not enough for the fraction of the kernel inside an energy selection whose
          boundary falls in the continuum (-0.6% in the expected counts of the Crab with
          a 200 keV cut; +0.02% with the default).
        """

        lo, hi = self.cell_edges_keV(level, index)
        width = np.log(hi / lo)
        step = min(width / self._steps_per_cell, self._max_epsilon_step)
        u_max = width + np.log1p(self._epsilon_edges[-1])

        k = np.arange(np.floor(-width / step), np.ceil(u_max / step) + 1)
        continuum = np.arange(self._epsilon_edges[0], np.expm1(-width), self._continuum_epsilon_step)
        edges = np.concatenate([self._epsilon_edges, np.expm1(k * step), [np.expm1(u_max)], continuum])
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

    def _piece_axes(self, level: int, index: int) -> Tuple[Axes, Axes]:
        """Axes, with units, of the pieces of a cell and of their total effective area."""

        lo, hi = self.cell_edges_keV(level, index)
        nulambda = self._irf.axes['NuLambda']
        ei_axis = Axis([lo, hi] * u.keV, label='Ei', scale='log')
        irf_axes = Axes([nulambda, ei_axis] + self._cds_axes(Axis(self.epsilon_edges(level, index), label='Epsilon')))

        return irf_axes, Axes([nulambda, ei_axis])

    def _cell_path(self, level: int, index: int) -> str:
        return f'{ROOT}/e{level}/c{index}/point'

    def _create_cell(self, f, level: int, index: int):
        """Empty (unallocated) datasets of a cell."""

        path = self._cell_path(level, index)
        group = f.require_group(path)
        lo, hi = self.cell_edges_keV(level, index)
        group.attrs['energy_range_keV'] = [lo, hi]

        nulambda = self._irf.axes['NuLambda']
        irf_axes, aeff_axes = self._piece_axes(level, index)

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

    def _require_file(self):
        if self._filename is None:
            raise RuntimeError("This cache has no file")

    def computed(self, level: int, index: int) -> np.ndarray:
        """Whether each ``NuLambda`` pixel of a cell has been computed and stored."""

        self._require_file()

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
        Compute and store the ``pixels`` of each ``(level, index)`` cell that are not in
        the file yet: from its children if they are both stored for those pixels, else
        from the IRF.
        """

        self._require_file()
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
        Compute and store all the cells of ``level`` from the IRF, and every coarser
        level from them, for ``pixels`` (default: all).
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
        energy, de_w, hats = self._quadrature(level, index)
        nq = energy.size

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
        The ``R`` and ``L`` pieces of a cell as IRFs, to be evaluated with a photon
        energy equal to the lower edge of the cell (see :meth:`cell_edges_keV`). Each
        ``NuLambda`` pixel is computed (or read from the file) the first time an
        evaluation needs it.

        Parameters
        ----------
        selections : EnergySelector or tuple thereof, optional
            Measured-energy selection, as for :class:`IRFRelativeHistUnpolarized`.
        """

        areas = self._piece_areas(level, index)

        return tuple(IRFRelativeHistUnpolarized(_PieceRows(self, level, index, piece),
                                                aeff=None if areas is None else areas[piece],
                                                selections=selections)
                     for piece in PIECES)

    def _quadrature(self, level: int, index: int):
        """Midpoint rule in ln E over a cell: energies, dE w(E), and the R and L hats."""

        lo, hi = self.cell_edges_keV(level, index)
        width = np.log(hi / lo)
        nq = max(int(np.ceil(width / self._quadrature_step)), 1)
        t = (np.arange(nq) + 0.5) / nq
        energy = lo * np.exp(t * width)

        return energy, (width / nq) * energy * energy ** self._weight_index, {'R': t, 'L': 1 - t}

    def _piece_areas(self, level: int, index: int):
        """
        If the IRF has a separate total effective area (usually on a finer grid), the
        area of the pieces on that grid: ``int h w Aeff dE`` for each of its pixels. The
        area is linear in its table, so this is exactly the integral over the cell of the
        area that the IRF interpolates, wherever it is evaluated. Otherwise None, and the
        area of each pixel is computed with its contents.
        """

        irf = self._irf
        if irf._aeff_from_irf:
            return None

        aeff = irf._tot_aeff
        energy, de_w, hats = self._quadrature(level, index)
        inside = irf._in_ei_range(aeff, energy)

        ei_bins, ei_weights = (np.asarray(a) for a in aeff.axes['Ei'].interp_weights(energy))
        ei_matrix = np.zeros((energy.size, aeff.axes['Ei'].nbins))
        for b, w in zip(ei_bins, ei_weights):
            np.add.at(ei_matrix, (np.arange(energy.size), b), w)

        lo, hi = self.cell_edges_keV(level, index)
        areas = {}
        for piece in PIECES:
            # New axes for each, since the IRF standardizes them in place
            axes = Axes([aeff.axes['NuLambda'], Axis([lo, hi] * u.keV, label='Ei', scale='log')])
            contents = np.asarray(aeff.contents) @ ((hats[piece] * de_w * inside) @ ei_matrix)
            areas[piece] = Histogram(axes, contents=contents[:, None], unit=u.cm ** 2)

        return areas

    def _read_piece_pixel(self, level: int, index: int, piece: str, pixel: int):
        """Contents and total effective area of one pixel of a piece."""

        if self._filename is not None:
            self.ensure([(level, index)], [pixel])
            rows = self._read_rows(level, index, [pixel])
            return rows[piece][0], rows[f'AEFF_{piece}'][0]

        # Both pieces are computed together, and kept until the other one is read
        key = (level, index, pixel)
        if key not in self._memory:
            self._memory[key] = {name: values[0] for name, values in
                                 self._direct(level, index, np.array([pixel])).items()}
        rows = self._memory[key]
        contents, area = rows.pop(piece), rows.pop(f'AEFF_{piece}')
        if not rows:
            del self._memory[key]

        return contents, area


class _PieceRows:
    """
    Row provider of one piece of a cell, for :class:`IRFRelativeHistUnpolarized`: the
    axes and unit of the piece, and its contents and total effective area per pixel.
    """

    def __init__(self, cache: BroadenedIRFCache, level: int, index: int, piece: str):
        self._cache, self._level, self._index, self._piece = cache, level, index, piece
        self.axes = cache._piece_axes(level, index)[0]
        self.unit = u.cm ** 2

    def read_pixel(self, pixel: int):
        return self._cache._read_piece_pixel(self._level, self._index, self._piece, pixel)
