from xid_functions import *
import sys
from time import time

arg1 = int(sys.argv[1])

run_XID_modelling(
    prior_name = "euclid_wide_v2.2_cirrus_2.5_testing_xid_modelling_larger_tile",
    output_name = "euclid_wide_v2.2_cirrus_2.5_testing_xid_modelling_larger_tile",
    job_array_num = arg1,
    order = 11,
    order_large = 7,
    id_large_tile = 77828,
    flux_prior = None,
    output_path = "./new_cirrus_modelling",
    cirrus_present = True,
    cirrus_structure_path = lustre_path / "cirrus" / "cirrus_pipeline" / "cirrus_2.3_v2.2.npy")