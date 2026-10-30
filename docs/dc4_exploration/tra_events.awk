# One line per Compton event of a MEGAlib .tra file (reads stdin):
#   TI SQ Eg Ee LA x1 y1 z1 x2 y2 z2 Etrue dx dy dz
# Eg = energy of the scattered gamma (CE), Ee = electron energy (CE), LA = lever arm (= first-to-second-hit distance),
# (x1,y1,z1)/(x2,y2,z2) = first / second hit (CH 0 / CH 1), Etrue and (dx,dy,dz) = initial energy and direction from the OI line
# (simulations only; 0 when absent). Only events with ET CO and both hits are written.
# Options: -v T0=.. -v T1=.. keep only events with T0 <= TI < T1.
function emit() {
  if (type == "CO" && h0 && h1 && (T1 == 0 || (t >= T0 && t < T1)))
    printf "%s %d %s %s %s %s %s %s %s %s %s %s %s %s %s\n", t, sq, eg, ee, la, x1, y1, z1, x2, y2, z2, et, dx, dy, dz
}
/^SE/ { emit(); type = ""; h0 = 0; h1 = 0; sq = 0; et = 0; dx = 0; dy = 0; dz = 0; next }
/^ET / { type = $2; next }
/^TI / { t = $2; next }
/^OI / { dx = $5; dy = $6; dz = $7; et = $11; next }
/^SQ / { sq = $2; next }
/^CE / { eg = $2; ee = $4; next }
/^LA / { la = $2; next }
/^CH 0 / { x1 = $3; y1 = $4; z1 = $5; h0 = 1; next }
/^CH 1 / { x2 = $3; y2 = $4; z2 = $5; h1 = 1; next }
END { emit() }
