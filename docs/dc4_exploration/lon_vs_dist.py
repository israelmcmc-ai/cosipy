"""SC-longitude structure near the axes (chi = 0/90/180/270 deg) in bands of the distance between the 1st and 2nd hit.

Histograms the offset of chi from the nearest axis (0.25 deg bins, +-20 deg) for each Distance band of a week of events
and saves lon_vs_dist.npz; the plot (counts / far-field baseline) is figures_dist1cm/sc_longitude_spike_vs_distance.png.
Usage: python lon_vs_dist.py events.fits.gz
"""
import sys, gzip, numpy as np
from sensitivity import stream_rows

BANDS = [(1, 2), (2, 3), (3, 5), (5, 8), (8, 30)]    # cm
edges = np.arange(-20, 20.001, 0.25)
H = np.zeros((len(BANDS), len(edges) - 1))
for c in stream_rows(gzip.GzipFile(sys.argv[1])):
    d = c['dist'].astype(float); E = c['E'].astype(float)
    off = ((np.rad2deg(c['chi'].astype(float)) + 45.) % 90.) - 45.      # offset from the nearest axis
    for i, (a, b) in enumerate(BANDS):
        s = (E >= 100) & (E < 10000) & (d >= a) & (d < b)
        H[i] += np.histogram(off[s], bins=edges)[0]
np.savez('lon_vs_dist.npz', H=H, edges=edges, bands=np.array(BANDS))
