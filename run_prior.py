from xid_functions import *
import numpy as np
from astropy.io import fits

print("start", flush = True)

# bands = ["PRIMA_1A_1", "PRIMA_1A_2", "PRIMA_1A_3", "PRIMA_1A_4", "PRIMA_1A_5", "PRIMA_1A_6",
#          "PRIMA_1B_1", "PRIMA_1B_2", "PRIMA_1B_3", "PRIMA_1B_4", "PRIMA_1B_5", "PRIMA_1B_6",
#          "PRIMA_2A", "PRIMA_2B", "PRIMA_2C", "PRIMA_2D"]

# bands = ["PRIMA_1A_1", "PRIMA_1A_2", "PRIMA_1A_3", "PRIMA_1A_4", "PRIMA_1A_5", "PRIMA_1A_6",
#          "PRIMA_1B_1", "PRIMA_1B_2", "PRIMA_1B_3", "PRIMA_1B_4", "PRIMA_1B_5", "PRIMA_1B_6"]

# bands = ["PRIMA_1A_1"]

# bands = [f"{band}_coadd" for band in bands]
# should change this to the input txt file but less flexible



primager = pd.read_csv("/mnt/lustre/users/astro/bp259/sides/inputs/PRIMAgerv2.2_coadd.txt")
bands = primager.band.to_numpy()
fwhms = primager.fwhm_arcsec.to_numpy()
sens = primager.sens_1sigma_Jy.to_numpy() * 1e3 # Convert to mJy TODO: Implement units into XID+?

band_indeces = [0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 10, 11,
                12, 13, 14, 15]

# band_indeces = [0,6,12,13,14,15]
# band_indeces = [11]

# band_indeces = band_indeces[0:6]
# band_indeces = band_indeces[4:6]


bands = list(bands[band_indeces])
fwhms = list(fwhms[band_indeces])
sens = list(sens[band_indeces])

psf_kernels = []
for band in bands:
    # psf_kernels.append(fits.open(f"/mnt/lustre/users/astro/bp259/sides/beams/v2/level2_broadened/{band}.fits")[0].data)
    # psf_kernels.append(fits.open(f"~/prima/sides/SV_beams/1.275m/{band}.fits")[0].data)
    # psf_kernels.append(fits.open(f"~/prima/sides/SV_beams/1.654m/{band}.fits")[0].data)
    # psf_kernels.append(fits.open(f"~/prima/sides/SV_beams/1.171m/{band}.fits")[0].data)

    psf_kernels.append(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data)

    # psf_kernels.append(fits.open(f"~/prima/sides/20260416_MSV2_Beam_Profiles/2.8_micron_RMS/{band}.fits")[0].data)    #MSV2
    # psf_kernels.append(fits.open(f"~/prima/sides/20260416_MSV2_Beam_Profiles/3.7_micron_RMS/{band}.fits")[0].data)    #MSV2
    # psf_kernels.append(fits.open(f"~/prima/sides/MSV2_beams/2.0_micron_RMS/{band}.fits")[0].data)    #MSV2
    # psf_kernels.append(fits.open(f"~/prima/sides/MSV2_beams/3.1_micron_RMS/{band}.fits")[0].data)    #MSV2

    # psf_kernels.append(fits.open(f"../sides/beams/v1/gaussian/{band}.fits")[0].data)
    # psf_kernels.append(fits.open(f"../sides/beams/v2/coadd/{band}.fits")[0].data)
    # psf_kernels.append(fits.open(f"../sides/beams/v2/coadd/{band}.fits")[0].data)
    # psf_kernels.append(fits.open(f"../sides/beams/v2/positionaloffset_broadened/{band}.fits")[0].data)
    # psf_kernels.append(fits.open(f"../sides/beams/v2/positionaloffset_broadened/{band}.fits")[0].data)

    # mf_kernels.append(fits.open(f"../sides/beams/v1/wiener/{band}.fits")[0].data)
    # mf_kernels.append(np.load("../cirrus/cirrus_pipeline/local_bkg_test_2A_new.npy"))

xid_prior(
    prior_name = f"euclid_wide_v2.2_order11_7_fwhmandsens",
    map_choice = "v2.2",
    catalogue_choice = "euclid_wide",
    bands = bands,
    order = 11,
    order_large = 7,
    id_large_tile = 77828,
    psf_kernels = psf_kernels,
    # cirrus_intensity = 2.5
    fwhms = fwhms,
    sigma_sens_list = sens
    )

    