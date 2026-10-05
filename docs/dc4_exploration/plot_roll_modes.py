"""Matrix plots (energy x phi) of SC-frame psichi maps per survey mode and 10 deg roll bin (see roll_modes.py).

One file per occupied (mode, roll bin), flat, with suffixes:
  psichi_maps_spacecraft_<Component>_survey_<north|south>_roll_<lo>to<hi>.png
plus roll_distribution_<Component>.png (events vs roll for each mode).

Usage: python plot_roll_modes.py roll.npz outdir Component [smooth_fwhm_deg] [min_events]
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_dc4 as P


def main(npz, outdir, comp, fwhm=4.0, min_events=20000):
    os.makedirs(outdir, exist_ok=True)
    z = np.load(npz)
    H, edges, nside = z['H'], z['roll_edges'], int(z['nside'])
    md = float(z['min_dist'])
    P.CUT = f' [Distance ≥ {md:g} cm]' if md > 0 else ''
    nE, nP = len(P.E_SLICES), len(P.PHI_SLICES)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for row, name in ((0, 'survey north'), (1, 'survey south'), (2, 'slews')):
        ax.step(edges[:-1] + 5, z['roll_all'][row], where='mid', label=name)
    ax.set_yscale('log'); ax.set_xlabel('roll angle [deg] (azimuth χ of the Earth zenith in SC coordinates)')
    ax.set_ylabel('events per 10° bin'); ax.axhline(min_events, color='k', lw=0.5, ls=':')
    ax.legend(); ax.set_title(f'{comp}: events vs roll{P.CUT}')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f'roll_distribution_{comp}.png'), dpi=100); plt.close(fig)

    made = []
    for row, mode in ((0, 'north'), (1, 'south')):
        for ir in range(len(edges) - 1):
            n = int(z['roll_all'][row][ir])
            if n < min_events:
                continue
            full = np.zeros((10, 36, H.shape[-1]))
            for k, ((e0, e1), (p0, p1)) in enumerate([(e, p) for e in P.E_SLICES for p in P.PHI_SLICES]):
                full[e0, p0] = H[row, ir, k]
            lo, hi = int(edges[ir]), int(edges[ir + 1])
            P.plot_maps({'spacecraft': full}, 'spacecraft', z['e_edges'], z['phi_edges'], nside, outdir, fwhm,
                        tag=f'_{comp}_survey_{mode}_roll_{lo:+04d}to{hi:+04d}',
                        label=f'{comp}, survey {mode}, roll {lo:+d}° to {hi:+d}°, N = {n:,} events')
            made.append((mode, lo, hi, n))
    print(len(made), 'matrices')
    return made


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 4.0,
         int(sys.argv[5]) if len(sys.argv) > 5 else 20000)
