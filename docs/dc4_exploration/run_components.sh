#!/bin/sh
# Stream each DC4 background component from Wasabi and bin it (psichi histograms, Distance >= MIN_DIST cm).
# Needs AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (public keys in the data-products README), awscli, and
# ORI = path to the 15 s orientation file. Usage: run_components.sh outdir [min_dist_cm]
OUT=${1:?outdir}; MIN=${2:-1}; ORI=${ORI:?path to orientation fits}
EP=https://s3.us-west-1.wasabisys.com; B=COSI-SMEX/DC4/Data/Backgrounds; B3=COSI-SMEX/DC3/Data/Backgrounds/Ge
mkdir -p "$OUT"
while read name key; do
  [ -s "$OUT/$name.npz" ] && continue
  aws s3 cp "s3://cosi-pipeline-public/$key" - --endpoint-url=$EP --only-show-errors \
    | python "$(dirname "$0")/bin_dc4.py" - "$ORI" "$OUT/$name.npz" 32 "$MIN" > "$OUT/$name.log" 2>&1
  echo "$name done: $(tail -1 "$OUT/$name.log")"
done <<EOF
AlbedoNeutrons $B/AlbedoNeutrons_WithDetCstunbinned_data_filtered_with_SAAcut.fits.gz
PrimaryAlphas $B/PrimaryAlphas_WithDetCstunbinned_data_filtered_with_SAAcut.fits.gz
PrimaryElectrons $B/PrimaryElectrons_WithDetCstunbinned_data_filtered_with_SAAcut.fits.gz
PrimaryPositrons $B/PrimaryPositrons_WithDetCstunbinned_data_filtered_with_SAAcut.fits.gz
PrimaryProtons $B/PrimaryProtons_WithDetCstunbinned_data_filtered_with_SAAcut.fits.gz
SAA $B/SAA_3months_unbinned_data_filtered_with_SAAcut_statreduced_akaHEPD01result.fits.gz
SecondaryPositrons $B/SecondaryPositrons_3months_unbinned_data_filtered_with_SAAcut.fits.gz
SecondaryProtons $B/SecondaryProtons_WithDetCstunbinned_data_filtered_with_SAAcut.fits.gz
SecondaryElectrons $B3/SecondaryElectrons_3months_unbinned_data_filtered_with_SAAcut.fits.gz
GalacticDiffuse $B3/GalTotal_SA100_F98_3months_unbinned_data_filtered_with_SAAcut.fits.gz
CosmicPhotons $B3/CosmicPhotons_3months_unbinned_data_filtered_with_SAAcut.fits.gz
EOF
