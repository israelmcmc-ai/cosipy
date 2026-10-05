#!/bin/sh
# Plot several binned components one after another: plot_components.sh bindir plotdir name1 name2 ...
# (bindir holds <name>.npz from run_components.sh; figures go to plotdir/<name>/). Needs ORI.
BIN=${1:?bindir}; PLOTS=${2:?plotdir}; shift 2
HERE=$(dirname "$0")
for n in "$@"; do
  "$HERE/plot_component.sh" "$BIN/$n.npz" "$PLOTS/$n" > "$PLOTS/$n.log" 2>&1
  echo "$n plotted (exit $?)"
done
