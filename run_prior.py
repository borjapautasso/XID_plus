from xid_functions import *
import numpy as np
from astropy.io import fits


bands = ["PRIMA_1A_1", "PRIMA_1A_2", "PRIMA_1A_3", "PRIMA_1A_4", "PRIMA_1A_5", "PRIMA_1A_6",
         "PRIMA_1B_1", "PRIMA_1B_2", "PRIMA_1B_3", "PRIMA_1B_4", "PRIMA_1B_5", "PRIMA_1B_6",
         "PRIMA_2A", "PRIMA_2B", "PRIMA_2C", "PRIMA_2D"]

bands = ["PRIMA_2A"]
bands = [f"{band}_coadd" for band in bands]
# should change this to the input txt file but less flexible


psf_kernels = []

for band in bands:
    psf_kernels.append(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data)


    # psf_kernels.append(fits.open(f"../sides/beams/v1/gaussian/{band}.fits")[0].data)
    # psf_kernels.append(fits.open(f"../sides/beams/v2/coadd/{band}.fits")[0].data)
    # psf_kernels.append(fits.open(f"../sides/beams/v2/coadd/{band}.fits")[0].data)
    # psf_kernels.append(fits.open(f"../sides/beams/v2/positionaloffset_broadened/{band}.fits")[0].data)
    # mf_kernels.append(fits.open(f"../sides/beams/v1/wiener/{band}.fits")[0].data)
    # mf_kernels.append(np.load("../cirrus/cirrus_pipeline/local_bkg_test_2A_new.npy"))

xid_prior(
    prior_name = f"euclid_wide_v2.2_cirrus_2.5_testing_xid_modelling_larger_tile",
    map_choice = "v2.2",
    catalogue_choice = "euclid_wide",
    bands = bands,
    order = 11,
    order_large = 7,
    id_large_tile = 77828,
    psf_kernels = psf_kernels,
    cirrus_intensity = 2.5)

    