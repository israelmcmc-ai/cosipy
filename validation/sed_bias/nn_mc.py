# Monte Carlo Em spectrum of the NN response for the Crab (3 months, injected power law), weighted by its area.
import sys, time, numpy as np, healpy as hp, warnings
from cosipy.response.ml.NFResponse import NFResponse
from cosipy.response.ml.nf_instrument_response_function import UnpolarizedNFFarFieldInstrumentResponseFunction
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame

def main():
    warnings.filterwarnings('ignore')
    S, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(seed)
    T = np.load(f'{S}/out/exposure.npz')['crab3m_n32_occ0']
    pix = rng.choice(len(T), N, p=T / T.sum())
    th, ph = hp.pix2ang(32, pix)
    # jitter within ~pixel
    E = np.exp(rng.uniform(np.log(100.), np.log(10000.), N))
    photons = PhotonListWithDirectionAndEnergyInSCFrame(ph, np.pi / 2 - th, E)
    rsp = NFResponse(f'{S}/data/unpolarized_nfresponse_v1-01.pt', devices=['cpu'], show_progress=False,
                     area_compile_mode=None, density_compile_mode=None, density_batch_size=20_000)
    irf = UnpolarizedNFFarFieldInstrumentResponseFunction(rsp)
    t0 = time.time()
    A = np.asarray(irf._effective_area_cm2(photons)); print('area', time.time() - t0, flush=True)
    ev = irf._random_events(photons); print('sample', time.time() - t0, flush=True)
    Em = np.asarray(ev.energy_keV)
    norm = T.sum() * 4.94717171 * np.log(100.) / E / N
    np.savez(f'{S}/out/nnlog_mc_{seed}.npz', E=E, A=A, Em=Em, norm=norm, phi=np.asarray(ev.scattering_angle_rad))
    print('done', flush=True)


if __name__ == '__main__':
    main()