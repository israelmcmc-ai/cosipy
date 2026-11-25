import sys, os, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wasabi import client, B
c = client()
D = os.path.join(sys.argv[1], 'data')
os.makedirs(D, exist_ok=True)
keys = ['COSI-SMEX/DC3/Data/Sources/crab_powerlaw_3months_unbinned_data_filtered_with_SAAcut.fits.gz',
        'COSI-SMEX/DC3/Data/Sources/GRB_bn090424592_3months_unbinned_data_filtered_with_SAAcut.fits.gz',
        'COSI-SMEX/DC4/Data/Orientation/DC4_final_530km_3_month_with_slew_15sbins_GalacticEarth_SAA.fits',
        'COSI-SMEX/develop/Data/Responses/ResponseContinuum.area.relative.nonsparse_smoothing1p0.h5.zip',
        'COSI-SMEX/develop/Data/Responses/relative_hist_irf_from_nf_response.h5.zip',
        'COSI-SMEX/DC4/Data/Responses/unpolarized_nfresponse_v1-01.pt',
        'COSI-SMEX/DC4/Data/Responses/ResponseContinuum.o3.e100_10000.b10log.s10396905069491.m2284.filtered.nonsparse.binnedimaging.imagingresponse.h5']
for k in keys:
    out = os.path.join(D, 'binned_continuum.h5' if 'm2284' in k else os.path.basename(k))
    final = out[:-4] if out.endswith('.zip') else out
    if os.path.exists(final + '.done'): continue
    for attempt in range(20):
        try:
            c.download_file(B, k, out); break
        except Exception as e:
            print('retry', k, e, flush=True)
    if out.endswith('.zip'):
        with zipfile.ZipFile(out) as z:
            z.extractall(D); print(z.namelist(), flush=True)
        os.remove(out)
    open(final + '.done', 'w').close()
    print('done', k, flush=True)
