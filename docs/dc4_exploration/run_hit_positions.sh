#!/bin/sh
# Hit-position histograms (week 1) for the DC4 background components that have a .tra file.
# Each .tra is a concatenation of increments that each span the whole 3 months, so the whole file is scanned and the events
# with TI inside week 1 are kept. Needs awk, gunzip, awscli with AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, python with numpy.
# Usage: [DMIN=cm] run_hit_positions.sh outdir      (DMIN: keep only events with first-to-second-hit distance > DMIN, default 0)
OUT=${1:?outdir}; mkdir -p "$OUT"
HERE=$(dirname "$0")
EP=https://s3.us-west-1.wasabisys.com; B=COSI-SMEX/DC4/Data/Backgrounds/Trafile
T0=1835487300; T1=$((T0 + 604800))
one() {
  name=$1; key=$2
  aws s3 cp "s3://cosi-pipeline-public/$B/$key" - --endpoint-url=$EP --only-show-errors | gunzip -c \
    | awk -v T0=$T0 -v T1=$T1 '/^TI /{t=$2; keep=(t>=T0&&t<T1)} keep&&/^CH /{print t, $2, $3, $4, $5, $6}' \
    | python "$HERE/hit_positions.py" "$OUT/$name.npz" "${DMIN:-0}" > "$OUT/$name.log" 2>&1
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
