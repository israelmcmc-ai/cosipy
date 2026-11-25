# Build an IRF with single Phi/Theta/Zeta bins (same Ei/Epsilon behavior) and write it to h5.
import sys, numpy as np, h5py, astropy.units as u
from histpy import Histogram, Axes, Axis, HealpixAxis
from cosipy.polarization import PolarizationAxis
S, name, src = sys.argv[1], sys.argv[2], sys.argv[3]
with h5py.File(src) as f:
    axes = Axes.open(f['IRF/axes'])
    aeff = Histogram.open(src, 'AEFF')
P = np.load(f'{S}/out/proj_{name}.npz')
conv = axes['Zeta'].convention
new_axes = [axes['NuLambda'], axes['Ei'], axes['Epsilon'],
            Axis([0, 180] * u.deg, label='Phi'), Axis([-180, 180] * u.deg, label='Theta'),
            PolarizationAxis(Axis([0, 360] * u.deg, label='Zeta'), convention=conv)]
h = Histogram(new_axes, contents=P['p_eps'][..., None, None, None], unit=u.cm ** 2)
h.write(f'{S}/out/reduced_{name}.h5', name='IRF', overwrite=True)
aeff.write(f'{S}/out/reduced_{name}.h5', name='AEFF', overwrite=True)
print('ok')
