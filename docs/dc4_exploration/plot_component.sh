#!/bin/sh
# Make the psichi matrix plots for one binned component: plot_component.sh comp.npz outdir
# (spacecraft + Earth survey north/south matrices, and the minus-total versions). Needs ORI.
NPZ=${1:?npz}; OUT=${2:?outdir}; ORI=${ORI:?orientation fits}
HERE=$(dirname "$0")
mkdir -p "$OUT"
PYTHONPATH=$HERE python "$HERE/plot_dc4.py" "$NPZ" "$OUT" 4 && \
PYTHONPATH=$HERE python "$HERE/plot_minus_total.py" "$NPZ" "$ORI" "$OUT" 4
