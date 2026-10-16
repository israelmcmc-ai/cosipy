#!/usr/bin/env python
# coding: UTF-8

"""
Add the broadened IRF ("BROADENED_IRF" and "BROADENED_AEFF" groups) to an HDF5
file built by relative_hist_irf_from_rsp.py or relative_hist_irf_from_nf_response.py,
for UnbinnedThreeMLPointSourceResponseBroadened. Load it with
IRFRelativeHistUnpolarized.from_h5(path, broadened=True).

See cosipy.response.broadened_relative_irf_hist. The whole original IRF is
loaded into memory.
"""

from pathlib import Path

from cosipy.response.broadened_relative_irf_hist import write_broadened_irf


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="HDF5 file with the IRF (and optionally AEFF) groups")
    p.add_argument("--nodes-per-decade", type=int, default=6)
    p.add_argument("--emin", type=float, default=100., help="First node [keV]")
    p.add_argument("--emax", type=float, default=10_000., help="Last node [keV]")
    p.add_argument("--weight-index", type=float, default=-2.)
    p.add_argument("--overwrite", action="store_true", help="Replace existing broadened groups")
    args = p.parse_args()

    write_broadened_irf(args.path, nodes_per_decade=args.nodes_per_decade, energy_range_keV=(args.emin, args.emax),
                        weight_index=args.weight_index, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
