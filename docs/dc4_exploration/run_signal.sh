#!/bin/sh
# Signal: the isotropic far-field response simulations (true direction and energy) in
# COSI-SMEX/DC3/Data/Responses/TrasFiltered -> one .npz of per-event arrays per file (tra_events.awk + resp_signal.py).
# Needs awscli with AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY. Usage: run_signal.sh outdir
OUT=${1:?outdir}; mkdir -p "$OUT"
HERE=$(dirname "$0")
EP=https://s3.us-west-1.wasabisys.com; P=COSI-SMEX/DC3/Data/Responses/TrasFiltered
one() {
  name=$1
  aws s3 cp "s3://cosi-pipeline-public/$P/$name.filtered.tra.gz" - --endpoint-url=$EP --only-show-errors | gunzip -c \
    | awk -v T1=0 -f "$HERE/tra_events.awk" | PYTHONPATH=$HERE python "$HERE/resp_signal.py" "$OUT/$name.npz" > "$OUT/$name.log" 2>&1
  echo "$name: $(tail -1 "$OUT/$name.log")"
}
for g in "ResponseContinuum.inc194047324.id1 ResponseContinuum.inc194081750.id1 ResponseContinuum.inc194094877.id1 ResponseContinuum.inc194172710.id1" \
         "ResponseContinuum.inc194207599.id1 ResponseContinuum.inc194255435.id1 ResponseContinuum.inc194320193.id1" \
         "ResponseContinuum.inc759680486.id1 ResponseContinuum.inc759780249.id1 ResponseContinuum.inc759813507.id1"; do
  for n in $g; do one $n & done
  wait
done
