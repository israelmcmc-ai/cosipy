"""Overlap between the Distance < 1 cm cut and the other cuts (SQ <= 5, upper two layers, phi < 90 deg, second hit below the first).
Usage: python plot_distance_overlap.py signal_dir bkgorder_dir outdir"""
import sys, os, itertools
import numpy as np
from order_common import *

D1 = np.digitize(1.0, DEDGE)                                                   # first Distance bin with D >= 1 cm
short = (np.arange(ND) < D1).reshape(1, 1, ND, 1, 1, 1, 1)
others = {'SQ ≤ 5': dict(sq=(2, 5)), 'upper two layers': dict(lay=(2, 3)), 'φ < 90°': dict(phimax=90), 'second hit below first': dict(down=1)}
Ssig = Sc_all[..., -1]                                                          # (NE,)+CS signal events
def frac(w, m, sel=None):
    """fraction of the weight in `sel` (default all) that passes mask m, per energy bin and all bins."""
    sel = np.ones(w.shape, bool) if sel is None else sel
    num = (w * (m & sel)).sum(tuple(range(1, w.ndim))); den = (w * sel).sum(tuple(range(1, w.ndim))); return num / den, (w * (m & sel)).sum() / (w * sel).sum()
shortb = np.broadcast_to(short, Nb.shape)
md = []
def table(t, rows, h):
    md.append(f'\n### {t}\n'); md.append('| ' + ' | '.join(h) + ' |'); md.append('|' + '---|' * len(h)); md.extend('| ' + ' | '.join(r) + ' |' for r in rows)
for kind, w in (('background', Nb), ('signal', Ssig)):
    rows = [[f'Distance < 1 cm (removed by the Distance ≥ 1 cm cut)'] + [f'{100 * v:.0f}' for v in frac(w, shortb)[0]] + [f'{100 * frac(w, shortb)[1]:.0f}']]
    for nm, kw in others.items():
        keep = rect(**kw); rem_short = frac(w, ~keep, shortb); rem_long = frac(w, ~keep, ~shortb)
        rows.append([f'{nm}: share of Distance < 1 cm events it removes'] + [f'{100 * v:.0f}' for v in rem_short[0]] + [f'{100 * rem_short[1]:.0f}'])
        rows.append([f'{nm}: share of Distance ≥ 1 cm events it removes'] + [f'{100 * v:.0f}' for v in rem_long[0]] + [f'{100 * rem_long[1]:.0f}'])
    allk = rect(sq=(2, 5), lay=(2, 3), phimax=90, down=1); r = frac(w, ~allk, shortb); rows.append(['all four: share of Distance < 1 cm events removed'] + [f'{100 * v:.0f}' for v in r[0]] + [f'{100 * r[1]:.0f}'])
    table(f'{kind}: overlap of each cut with the Distance < 1 cm events (%)', rows, ['cut'] + LAB + ['all bins'])
# marginal removal of the Distance cut after adding the other cuts one by one (background events, all bins)
rows = []; names = list(others); cur = {}
for k in range(len(names) + 1):
    if k: cur.update(others[names[k - 1]])
    base = rect(**cur); with_d = rect(dmin=1, **cur); before = 1 - (Nb * base).sum() / Nb.sum(); after = 1 - (Nb * with_d).sum() / Nb.sum()
    rows.append(['no other cut' if not k else '+ ' + names[k - 1], f'{100 * before:.0f}', f'{100 * after:.0f}', f'{100 * (after - before):.1f}', f'{100 * (1 - (Nb * with_d).sum() / (Nb * base).sum() if (Nb * base).sum() else 0):.0f}'])
table('Extra background removed by Distance ≥ 1 cm when the other cuts are added cumulatively (percentage points of all background events)', rows, ['cuts already applied', 'removed without the Distance cut [%]', 'removed with it [%]', 'extra [points]', 'share of the remaining events it removes [%]'])
# which pairs overlap most with Distance < 1: pairwise
rows = []
for a, b in itertools.combinations(names, 2):
    keep = rect(**others[a], **{k: v for k, v in others[b].items()}); rows.append([f'{a} + {b}', f'{100 * frac(Nb, ~keep, shortb)[1]:.0f}'])
table('Share of the Distance < 1 cm background events removed by pairs of the other cuts [%]', rows, ['pair', 'removed [%]'])
open(os.path.join(outdir, 'distance_overlap.md'), 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
