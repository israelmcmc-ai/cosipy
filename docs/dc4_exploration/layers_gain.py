import sys, itertools, numpy as np
sys.path.insert(0, '/home/user/cosipy/docs/dc4_exploration')
import sensitivity_quality as Q
from sensitivity_quality import Cut, fom, combined
Hc = Q.cumulative(Q.signal_hist(Q.signal_files(sys.argv[1]))[0]); bN, bS, _ = Q.bkg_hist(sys.argv[2])
base = fom(Cut(), Hc, bN, bS); N0 = base['Bn'].sum()
LY = {'all layers': (0, 1, 2, 3), 'not bottom (layers 2-4)': (1, 2, 3), 'not bottom 2 (layers 3-4)': (2, 3), 'top only': (3,)}
Ds = [0, 0.5, 1, 1.5, 2, 3, 4, 6]; SQ = [(a, b) for a in (2, 3, 4) for b in (3, 4, 5, 7) if b >= a]
res = {k: [] for k in LY}
for (k, l), d, (a, b) in itertools.product(LY.items(), Ds, SQ):
    r = fom(Cut(la_min=d, sq_min=a, sq_max=b, layers1=l), Hc, bN, bS)
    res[k].append((1 - r['Bn'].sum() / N0, 100 * (combined(r['fom']) / combined(base['fom']) - 1), r['fom'] / base['fom'] - 1))
print('best all-bin S/sqrt(B) change [%] at >= removal (Distance x SQ range x layer set)')
print('removal  ' + '  '.join(f'{k:>27}' for k in LY))
for lv in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
    print(f'{100*lv:5.0f}%   ' + '  '.join(f'{max([c for rm, c, _ in v if rm >= lv - 1e-9], default=np.nan):27.1f}' for v in res.values()))
for k in ('not bottom 2 (layers 3-4)',):
    r = fom(Cut(layers1=LY[k]), Hc, bN, bS); print(k, 'alone: bkg removed %.0f%%, signal kept %.0f%%, per bin' % (100*(1-r['Bn'].sum()/N0), 100*(r['Sall']/base['Sall']).mean()), np.round(100*(r['fom']/base['fom']-1),1), 'w', r['w'])
