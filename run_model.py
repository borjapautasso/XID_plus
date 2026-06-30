from xid_functions import *
import sys

params = sys.argv[1:]

prior_name = params[0]
output_name = params[1]
job_array_num = int(params[2])
order = int(params[3])
order_large = int(params[4])
id_large_tile = int(params[5])
flux_prior = None if params[6] == "None" else float(params[6])
flux_stepwise = int(params[7])
output_path = None if params[8] == "None" else Path(params[8])
cirrus_structure_path = None if params[9] == "None" else Path(params[9])
num_samples = int(params[10])
num_warmup = int(params[11])
num_chains = int(params[12])
chain_method = params[13]
output = True if params[14] == "True" else False
expand_fwhm = float(params[15])

# Is this a bit overcooked? Helps with the slurm scheduling though
run_XID_modelling(prior_name, output_name, job_array_num, order, order_large,
                  id_large_tile, flux_prior, flux_stepwise, output_path,
                  cirrus_structure_path, num_samples, num_warmup, num_chains,
                  chain_method, output, expand_fwhm)

