#!/bin/sh
# Background histograms for the quality-cut study: week 1 of every background component with a filtered .tra (as run_hit_positions.sh).
# Usage: run_bkg_quality.sh outdir
OUT=${1:?outdir}; mkdir -p "$OUT"
HERE=$(dirname "$0")
EP=https://s3.us-west-1.wasabisys.com; B=COSI-SMEX/DC4/Data/Backgrounds/Trafile
T0=1835487300; T1=$((T0 + 604800))
one() {
  name=$1; key=$2
  aws s3 cp "s3://cosi-pipeline-public/$B/$key" - --endpoint-url=$EP --only-show-errors | gunzip -c \
    | awk -v T0=$T0 -v T1=$T1 -f "$HERE/tra_events.awk" | PYTHONPATH=$HERE python "$HERE/bkg_order.py" "$OUT/$name.npz" > "$OUT/$name.log" 2>&1
  echo "$name: $(tail -1 "$OUT/$name.log")"
}
one CosmicPhotons CosmicPhotons.extracted.filtered.tra.gz &
one AlbedoPhotons AlbedoPhotons_WithDetCst.extracted.filtered.tra.gz &
one PrimaryProtons PrimaryProtons_WithDetCst.extracted.filtered.tra.gz &
wait
one GalacticDiffuse GalacticDiffuse.inc1.id1.extracted.filtered.tra.gz &
one PrimaryAlphas PrimaryAlphas_WithDetCst.extracted.filtered.tra.gz &
one SecondaryPositrons SecondaryPositrons.extracted.filtered.tra.gz &
one AlbedoNeutrons AlbedoNeutrons_WithDetCst.extracted.filtered.tra.gz &
wait
one SecondaryElectrons SecondaryElectrons.extracted.filtered.tra.gz
one SecondaryProtons SecondaryProtons_WithDetCst.extracted.filtered.tra.gz
one PrimaryElectrons PrimaryElectrons_WithDetCst.extracted.filtered.tra.gz
one PrimaryPositrons PrimaryPositrons_WithDetCst.extracted.filtered.tra.gz
