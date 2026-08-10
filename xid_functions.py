"""
Combines bp_xid_prior and bp_xid_functions into one folder,
contains functions for both making priors, and running XID+
"""

import os
import pickle
from pathlib import Path
from time import time
from typing import Optional, Literal

import numpy as np
import pandas as pd

from tqdm import tqdm

import astropy.convolution as conv
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, join

import jax
import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, uniform_filter

import xidplus
from xidplus import moc_routines, HPC
from xidplus.numpyro_fit.misc import sp_matmul

lustre_path = Path("/mnt/lustre/users/astro/bp259/")
lustre_path_prima = lustre_path / "prima_data"
lustre_path_xid = lustre_path_prima / "xid_plus"

research_path = Path("/research/astro/fir/")

### Main XID+ functions

def xid_prior(
    prior_name: str,
    map_choice: str,
    catalogue_choice: str,
    bands: list[str],
    order: int,
    order_large:int,
    id_large_tile: int,
    psf_kernels: list[np.ndarray],
    cirrus_intensity: None|float = None,
    subtraction_type: None|str = None,
    fwhms: None|list[float] = None,
    sigma_sens_list: None|list[float] = None,
    ):
    """
    Creates XID+ prior.

    For each band creates three pickle files:
     - `Master_prior`: contains the prior object alongside a list of all the 
        small tiles;
     - `Tiles`: contains a list of all the small and large tiles;
     - `Tile_X`: contains the prior object of the large tile, selected via 
        `id_large_tile`.

    Args:
        prior_name (str):
            Run name, determines name of output file.
        map_choice (str):
            Choice of map to use.
        catalogue_choice (str):
            Choice of catalogue to use as positional/flux priors.
            The catalogue must include the bands which are being modelled.
        bands (list of str):
            List of bands being modelled.
        order (int):
            HEALPix order of small tile.
        order_large (int):
            HEALPix order of large tile.
        id_large_tile (int):
            HEALPix ID of large tile.
        psf_kernels (list of np.ndarray):
            List of 2D arrays that define the beam for each band.
        cirrus_intensity (None or float):
            Cirrus intensity (I100) in MJy/sr.
        subtraction_type (None or str):
            Type of background subtraction to apply to the maps.
    """
    tstart = time()

    print(f"Creating {prior_name = } with {map_choice = } map and {catalogue_choice = } cat.", flush = True)

    ### Select prior catalogue
    cat: Table = get_catalogue(catalogue_choice)

    print(f"Prior catalogue contains {len(cat)} sources.", flush = True)

    # For sources with no true flux (e.g. quiescent), change from 0 to 1e-12 Jy.
    for band in bands:
        cat[f"S{band}"] = np.where(cat[f"S{band}"] == 0, 1e-12, cat[f"S{band}"])

    inid = np.array(cat["ID"])
    inra = np.array(cat["ra"])
    indec = np.array(cat["dec"])
    ### Select map
    map_path, npps = get_map(map_choice, bands)

    ### Create list of priors
    priors = []
    print("Starting prior creation...", flush = True)
    for i, band in enumerate(bands):
        # Get image and error maps
        hdul = fits.open(map_path[i])
        im = hdul[0].data
        header = hdul[0].header
        hdul.close()

        # Add cirrus
        if cirrus_intensity != 0 and cirrus_intensity != None:
            print(f"Adding cirrus. Intensity = {cirrus_intensity}", flush = True)
            try:
                cirrus_im = np.load(lustre_path / "cirrus" / "cirrus_pipeline" / f"{band}_v2.npy")
                im += cirrus_im * cirrus_intensity
            except IOError:
                raise IOError("Cirrus map at given band does not exist.")

        # Convert to mJy
        im *= 1e3
        im -= np.mean(im)
        error_im = np.full_like(im, npps[i] * 1e3)
        
        # Beam
        psf = psf_kernels[i]

        if fwhms is not None:
            fwhm = fwhms[i]
        else:
            fwhm = None

        if sigma_sens_list is not None:
            sigma_sens = sigma_sens_list[i]
        else:
            sigma_sens = None


        # Perform background subtraction
        if subtraction_type is not None:
            im = bkg_subtraction(im, subtraction_type, band)

        # Initialise prior
        influx_mu = np.array(cat[f"S{band}"]) * 1e3
        influx_sigma = np.array(cat[f"S{band}"]) * 1e3

        prior = xidplus.prior(im, error_im, header, header)
        # prior.stepwise_prima_prior_cat(inra, indec, cat, flux_mu = influx_mu, flux_sigma = influx_sigma)

        # Remove cat, duplicating it at every step??
        prior.stepwise_prima_prior_cat(inra, indec, None,
                                       flux_mu = influx_mu, flux_sigma = influx_sigma,
                                       ID = inid, fwhm = fwhm, sigma_sens = sigma_sens)
        prior.prior_bkg(0., 5.)

        pind = np.arange(0, psf.shape[0], 1)
    
        prior.set_prf(psf, pind, pind) # requires PRF as 2d grid, and x and y bins for grid (in pixel scale)
        priors.append(prior)
        
        print(f"Finished {band} prior.", flush = True)
    
    ### Create pickle files..
    tiles = moc_routines.get_HEALPix_pixels(order, inra, indec, unique = True)
    tiles_large = moc_routines.get_HEALPix_pixels(order_large, inra, indec, unique = True)

    # +1 since hierarchical_tile needs it, dont like it but easier this way.
    index_large_tile = np.where(tiles_large == id_large_tile)[0][0] + 1
    
    print(f"----- There are {len(tiles)} tiles required for input catalogue and {len(tiles_large)} large tiles.", flush = True)

    for i, band in enumerate(bands):

        output_folder = lustre_path_xid / "prior_processing_output" / band

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # Master prior
        outfile_master = output_folder / f"{prior_name}_Master_prior.pkl"
        with open(outfile_master, "wb") as f:
            pickle.dump({"priors": priors[i], "tiles": tiles, "order": order, "version": xidplus.io.git_version()}, f)

        # Tiles
        outfile_tiles = output_folder / f"{prior_name}_Tiles.pkl"
        with open(outfile_tiles, "wb") as f:
            pickle.dump({"tiles": tiles, "order": order, "tiles_large": tiles_large, "order_large": order_large, "version": xidplus.io.git_version()}, f)

        # Individual large tile
        HPC.hierarchical_tile_single(outfile_master, outfile_tiles, output_folder, prior_name, index_large_tile)
        print(f"Created {band} files.", flush = True)

    tstop = time()
    print(f"Prior created in {tstop - tstart:.3f} s", flush = True)

def run_XID_modelling(
    prior_name: str,
    output_name: str,
    job_array_num: int,
    order: int,
    order_large: int,
    id_large_tile: int,
    flux_prior: None|float,
    flux_stepwise: bool = False,
    output_path = None,
    cirrus_structure_path = None,
    num_samples = 500,
    num_warmup = 500,
    num_chains = 4,
    chain_method = "parallel",
    output: bool = True,
    expand_fwhm = False,
    stepwise_prior_index = None,
    stepwise_prior_name = None,
    flux_sampling_space = "linear",
    ):
    """
    Runs XID+ through the HPC via array jobs. Alongside a general rewrite of
    ``run_xid.ipynb`` from James.
    
    Args:
        prior_name (str):
            Name of the prior to be used. There needs to be prior created
            with this name.
        output_name (str):
            Name of output file.
        job_array_num (int):
            SLURM array job number. Encodes both band and small tile index.
        order (int):
            HEALPix order of small tile.
        order_large (int):
            HEALPix order of large tile. Must be less or equal to `order`.
        id_large_tile (int):
            HEALPix ID of large tile.
        flux_prior (None or float):
            Flux knowledge to use in modelling. If `None` uses a flat prior.
            Otherwise sets the scale factor applied to the mean and width of
            the Gaussian flux prior.
        flux_stepwise (int):
            Whether to use stepwise methodology in the modelling.
            If 0, no stepwise
            If 1, stepwise for >1a1
            if 2, stepwise for >=1a1
        output_path (Path or str or None):
            Path to output directory. Results are saved in `output_path / output_name`.
            Defaults to `lustre_path_xid / "xid_outputs" / output_name`.
        cirrus_structure_path (Path or str or None):
            Path to input cirrus structure array. Defaults to inside lustre.
            If `None` or `False`, does not perform cirrus modelling.
        num_samples (int):
            Number of samples to perform in the MCMC modelling.
        num_warmup (int):
            Number of warmup steps to perform in the MCMC modelling.
        num_chains (int):
            Number of chains to run in the MCMC modelling. 
        chain_method (str):
            Paralisation method in numpyro.
        output (bool):
            Whether to output the modelling results.
    """
    from time import time
    
    if output_path is None:
        outfolder = lustre_path_xid / "xid_outputs" / output_name
    else:
        if not isinstance(output_path, Path):
            outfolder = Path(output_path) / output_name
        else:
            outfolder = output_path / output_name

    # happened one too many time :/
    if chain_method == "vectorised":
        chain_method = "vectorized"

    print(f"Using {prior_name} prior.", flush = True)
    print(f"Saving as {output_name}.")

    bands = ["PRIMA_1A_1", "PRIMA_1A_2", "PRIMA_1A_3", "PRIMA_1A_4", "PRIMA_1A_5", "PRIMA_1A_6",
            "PRIMA_1B_1", "PRIMA_1B_2", "PRIMA_1B_3", "PRIMA_1B_4", "PRIMA_1B_5", "PRIMA_1B_6",
            "PRIMA_2A", "PRIMA_2B", "PRIMA_2C", "PRIMA_2D"]
    bands = [f"{band}_coadd" for band in bands]
    
    # Defines how many small tiles there are in a large tile
    # Used alongside job_array_num to get band and small tile
    small_tiles_per_large = int(4**(order - order_large))

    index_band = int(job_array_num/small_tiles_per_large)
    index_tile = job_array_num%small_tiles_per_large

    band = bands[index_band]
    print(f"Band: {band}; Tile: {index_tile}.")

    ### Prepare small tile prior
    tstart = time()
    # prior.get_pointing_matrix()
    # 
    # pointing_time = tstop - tstart
    # print(f"Pointing matrix calculated in {pointing_time:.3f} s", flush = True)
    input_folder = lustre_path_xid / "prior_processing_output" / band

    # Get the list of all small tile IDs
    infile = input_folder / f"{prior_name}_Tiles.pkl"
    with open(infile, "rb") as f:
        obj = pickle.load(f)
    all_tiles = obj["tiles"]

    # Mask list to get small tile IDs within the large tile
    tile_mask = np.full_like(all_tiles, False, dtype=bool)
    for itile, tile in enumerate(all_tiles):
        itile_large = moc_routines.tile_in_tile(order, tile,order_large)
        if itile_large == id_large_tile:
            tile_mask[itile] = True
    tiles = all_tiles[tile_mask]
    
    id_small_tile = tiles[index_tile]

    # Load Prior object for large tile
    infile = input_folder / f"{prior_name}_Tile_{id_large_tile}_{order_large}.pkl"
    with open(infile, "rb") as f:
        obj = pickle.load(f)
    prior = obj["priors"]
    
    # Trim map to small tile area + ring
    if expand_fwhm:
        moc = moc_routines.get_fitting_region(order, id_small_tile, -1)
        prior.moc = moc
        prior.cut_down_prior(expand_fwhm)
        print(f"Expanding by {expand_fwhm} * FWHM.")
    else:
        moc = moc_routines.get_fitting_region(order, id_small_tile, order+2 if order >= 9 else 11)
        prior.moc = moc
        prior.cut_down_prior() # DOING THIS WAY ALSO DOESNT CAUSE ISSUE WITH PRIORS TAHT WERENT GENERATED WITH THE FWHM AND EXPAND PARAMS
        print(f"Expanding by a HEALPIX order {order+2 if order >= 9 else 11} tile.")
    tstop = time()
    prior_time = tstop - tstart
    print(f"Prior loaded in {prior_time:.3f} s", flush = True)
    # # Trim map to small tile area + ring
    # moc = moc_routines.get_fitting_region(order, id_small_tile, order+2)
    # prior.moc = moc
    # prior.cut_down_map()

    # # Trim catalogue to small tile area + ring
    # moc = moc_routines.get_fitting_region(order, id_small_tile, order+2)
    # prior.moc = moc
    # prior.cut_down_cat()

    print(f"{prior.nsrc} sources in prior")
    print(f"{prior.snpix} pixels in prior", flush = True)

    if cirrus_structure_path is None or cirrus_structure_path == False:
        cirrus_map = None
    else:
        beam = fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data
        cirrus_full = np.load(cirrus_structure_path)
        cirrus_full = conv.convolve_fft(cirrus_full, beam, boundary = "wrap", normalize_kernel = True, allow_huge = True)
        cirrus_tile = cirrus_full[prior.sy_pix, prior.sx_pix]
        cirrus_map = cirrus_tile

    print("\nStarting pointing matrix", flush = True)
    tstart = time()
    prior.get_pointing_matrix()
    tstop = time()
    pointing_time = tstop - tstart
    print(f"Pointing matrix calculated in {pointing_time:.3f} s", flush = True)

    print("\nStarting upper limits", flush = True)
    tstart = time()
    prior.upper_lim_map()
    tstop = time()
    upper_time = tstop - tstart
    print(f"Upper limits calculated in {upper_time:.3f} s", flush = True)

    print(f"Prior object ready for small tile {id_small_tile}, HEALpix order = {order} (Large Tile {id_large_tile}, HEALpix order = {order_large})")
    
    prior.prior_true_flux = prior.prior_flux_mu.copy()
    
    # Set 1A_1 prior to 0 (i.e. uniform) when doing stepwise for band > 1A_1 (with uniform 25um)
    if index_band == 0 and flux_stepwise == 1:
        flux_prior = 0 
        print(f"\nRunning {band} with uniform flux prior")

    elif index_band > 0 and flux_stepwise != 0:
        tstart = time()
        print("\nLoading previous posterior...")
        prev_band = bands[index_band - 1]

        if index_band == stepwise_prior_index and stepwise_prior_name is not None:
            prev_run_name = stepwise_prior_name
            print(f"Loading {prev_band} outputs from {prev_run_name}")
        else:
            prev_run_name = output_name



        # prev_posterior_file = outfolder / "posterior" / f"xid_{output_name}_{prev_band}_tile{id_small_tile}_order{order}_large{order_large}_posterior.pkl"
        # with open(prev_posterior_file, "rb") as f:
        #     data = pickle.load(f)
        #     prev_posterior = data["posterior"]
        # prev_median = np.percentile(prev_posterior.samples['src_f'][:,0,:], 50.0, axis=0)

        # No longer doing it on a tile by tile, what with the different expansions for different bands
        merged_file = lustre_path_xid / "xid_outputs" / prev_run_name / "summary" / f"merged_xid_{prev_run_name}_{prev_band}_{id_large_tile}.csv"
        merged = pd.read_csv(merged_file).set_index("ID")

        # This breaks if theres a missing source (from tiles in the edge of the o7, as you progress through the band the prior includes more and more flux)
        # prev_median = merged.loc[prior.ID, "f_xid"].values

        # TODO: Explicitly check its sources outside of the o7, although I think that is the only way it would happen


        prev_median = merged.reindex(prior.ID)["f_xid"].values.copy()
        prev_sigma = prev_median.copy()

        missing = np.isnan(prev_median)
        if np.any(missing):
            print(f"WARNING: {np.sum(missing)} sources outside o7 tile")
            prev_median[missing] = 1.
            prev_sigma[missing] = 100.

        if index_band < 6 or index_band == 15:
            correction_factor = 1.0
        elif index_band < 12:
            correction_factor = 1.3
        else:
            correction_factor = 2.0

        prior.prior_flux_mu = prev_median * correction_factor
        prior.prior_flux_sigma = prev_sigma * correction_factor
        print("Stepwise ready")
        tstop = time()
        stepwise_time = tstop - tstart
        print(f"Stepwise prepped in {stepwise_time:.3f} s", flush = True)

        # print(f"flux_mu min: {np.min(prior.prior_flux_mu)}, contains NaN: {np.any(np.isnan(prior.prior_flux_mu))}, contains zero/neg: {np.any(prior.prior_flux_mu <= 0)}")



    # # Make prior.prior_flux_lower != 0 (for log modelling, makes no difference otherwise)
    # prior.prior_flux_lower = np.full((prior.sra.shape), 1e-9)


    if flux_sampling_space == "log":
        prior.prior_flux_lower = np.full((prior.sra.shape), prior.sigma_sens/10)

        # Not sure i love this tbh ugh

        prior.prior_flux_mu = np.maximum(prior.prior_flux_mu, prior.prior_flux_lower)
        prior.prior_flux_sigma = np.maximum(prior.prior_flux_sigma, prior.prior_flux_lower)
        
    ### Runs numpyro fitting on the small tile
    print("\nStarting modelling", flush = True)
    tstart = time()
    fit = single_band([prior], flux_prior, flux_sampling_space, cirrus_map, num_samples, num_warmup, num_chains, chain_method)

    samples = fit.get_samples() # get samples from the fit (not sure if this is needed when using the numpyro method. I think left over)
    posterior = xidplus.posterior.posterior_numpyro(fit, [prior])

    tstop = time()
    model_time = tstop - tstart
    print(f"\nModelling completed in: {model_time:.3f} s", flush = True)

    # Diagnostics to check mcmc run correctly, should be 0 for all three
    # TODO: Should probably check that none of the kept sources have bad diagnostics? this is being a bit conservative right now
    print(np.sum(np.array(posterior.divergences)))
    print(np.sum(posterior.Rhat['src_f'] >= 1.1))
    print(np.sum(posterior.n_eff['src_f']/posterior.samples['src_f'].shape[0] < 0.001))


    if output:
        print("Saving results...")


        ### Save outputs

        outfolder_prior = outfolder / "prior"
        outfolder_posterior = outfolder / "posterior"
        outfolder_summary = outfolder / "summary"
        
        if not os.path.isdir(outfolder_prior):
            os.makedirs(outfolder_prior)

        if not os.path.isdir(outfolder_posterior):
            os.makedirs(outfolder_posterior)

        if not os.path.isdir(outfolder_summary):
            os.makedirs(outfolder_summary)

        outfile_prior = outfolder_prior / f"xid_{output_name}_{band}_tile{id_small_tile}_order{order}_large{order_large}_prior.pkl"
        outfile_posterior = outfolder_posterior / f"xid_{output_name}_{band}_tile{id_small_tile}_order{order}_large{order_large}_posterior.pkl"
        outfile_summary = outfolder_summary / f"xid_{output_name}_{band}_tile{id_small_tile}_order{order}_large{order_large}_summary.csv"

        # Mask out sources which were in the modelling due to the tile
        # expansion but not  within the small tile.
        central_sources = moc_routines.sources_in_tile([id_small_tile],order,prior.sra,prior.sdec)

        src_id = prior.ID
        ra = prior.sra
        dec = prior.sdec

        f_true = prior.prior_true_flux

        f_xid = np.percentile(posterior.samples['src_f'][:,0,:], 50.0, axis=0)
        f_xid_l = np.percentile(posterior.samples['src_f'][:,0,:], 15.9, axis=0)
        f_xid_u = np.percentile(posterior.samples['src_f'][:,0,:], 84.1, axis=0)

        ### Output results to pkl
        with open(outfile_prior, "wb") as f:
            pickle.dump({"prior": prior}, f)
            
        with open(outfile_posterior, "wb") as f:
            pickle.dump({"posterior": posterior,
                        "kept_sources": central_sources}, f)


        df = pd.DataFrame(data = {"ID": src_id,
                                  "f_true": f_true,
                                  "f_xid": f_xid,
                                  "f_xid_l": f_xid_l,
                                  "f_xid_u": f_xid_u,
                                  "ra": ra,
                                  "dec": dec,
                                  "is_central": central_sources
                                })
            
        df.to_csv(outfile_summary, index = False)

    print("Completed")

### Modelling helper functions

def single_model(
    priors,
    flux_prior: float|None = None,
    flux_sampling_space: str|None = None,
    cirrus_map: np.ndarray|None = None
    ):

    pointing_matrices = [([p.amat_row, p.amat_col], p.amat_data) for p in priors]
    flux_lower = np.asarray([p.prior_flux_lower for p in priors]).T
    flux_upper = np.asarray([p.prior_flux_upper for p in priors]).T
    bkg_mu= np.asarray([p.bkg[0] for p in priors]).T
    bkg_sig = np.asarray([p.bkg[1] for p in priors]).T
    flux_mu = np.asarray([p.prior_flux_mu for p in priors]).T
    flux_sigma = np.asarray([p.prior_flux_sigma for p in priors]).T

    log_flux_lower = np.log(flux_lower)
    log_flux_upper = np.log(flux_upper)
    log_flux_mu = np.log(flux_mu)
    log_flux_sigma = np.sqrt(np.log(1 + (flux_sigma / flux_mu)**2))

    with numpyro.plate('bands', len(priors)):
        
        sigma_conf = numpyro.sample('sigma_conf', dist.HalfCauchy(scale = 0.5))

        bkg = numpyro.sample('bkg', dist.Normal(bkg_mu, bkg_sig))

        if cirrus_map is not None:
            cirrus_scale = numpyro.sample('cirrus_scale', dist.Uniform(0, 100))

        with numpyro.plate('nsrc', priors[0].nsrc):
            if flux_sampling_space == "linear":
                if flux_prior is not None and flux_prior != 0:
                    src_f = numpyro.sample('src_f', dist.TruncatedNormal(flux_mu, flux_sigma * flux_prior, low = flux_lower, high = flux_upper))
                else:
                    src_f = numpyro.sample('src_f', dist.Uniform(flux_lower, flux_upper))

            elif flux_sampling_space == "log":
                if flux_prior is not None and flux_prior != 0:
                    log_src_f = numpyro.sample('log_src_f', dist.TruncatedNormal(log_flux_mu, log_flux_sigma * flux_prior, low=log_flux_lower, high=log_flux_upper))
                else:
                    log_src_f = numpyro.sample('log_src_f', dist.Uniform(log_flux_lower, log_flux_upper)) # This is log uniform now
                # Convert back to linear space for the model
                src_f = numpyro.deterministic('src_f', jnp.exp(log_src_f))
            else:
                raise ValueError(f"flux_sampling_space must be either 'linear' or 'log' = {flux_sampling_space}")


    # # Modelled map = convolved sources + bkg (+ cirrus)
    modelled_map = sp_matmul(pointing_matrices[0], src_f[:, 0][:, None], priors[0].snpix).reshape(-1) + bkg[0]

    if cirrus_map is not None:
        cirrus_component = cirrus_scale[0] * cirrus_map.reshape(-1)
        modelled_map += cirrus_component

    # Total noise = sqrt(inst^2 + conf^2)
    sigma_tot = jnp.sqrt(jnp.power(priors[0].snim, 2) + jnp.power(sigma_conf[0], 2))

    with numpyro.plate('map_pixels', priors[0].snim.size):
        numpyro.sample("modelled_map", dist.Normal(modelled_map, sigma_tot), obs = priors[0].sim)

def single_band(
    priors,
    flux_prior: float|None = None,
    flux_sampling_space: str|None = None,
    cirrus_map: np.ndarray|None = None,
    num_samples = 500,
    num_warmup = 500,
    num_chains = 4,
    chain_method = "parallel"
    ):
    if jax.default_backend() == "gpu":
        print("GPU detected, running with GPU.")
        print(f"{jax.device_count()} GPU(s) detected.")
    elif chain_method == "parallel":
        print("GPU not detected, running with CPU in parallel mode.")
        print(f"Setting # devices to # chains ({num_chains}).")
        numpyro.set_host_device_count(num_chains)
    else:
        print(f"GPU not detected, running with CPU in {chain_method} mode.")


    nuts_kernel = NUTS(single_model, init_strategy = numpyro.infer.init_to_median(num_samples = 100))
    rng_key = random.PRNGKey(0)

    print("\nMODELLING PARAMETERS:")
    print(f"{flux_prior = }")
    print(f"{flux_sampling_space = }")
    print(f"{type(cirrus_map) = }")
    print(f"{num_samples = }")
    print(f"{num_warmup = }")
    print(f"{num_chains = }")
    print(f"{chain_method = }")

    mcmc = MCMC(nuts_kernel, num_samples = num_samples, num_warmup = num_warmup, num_chains = num_chains, chain_method = chain_method)
    mcmc.run(rng_key, priors, flux_prior, flux_sampling_space, cirrus_map, extra_fields = ('potential_energy', 'energy',))

    return mcmc

### Prior helper functions

def get_catalogue(catalogue_choice):
    # For now just copy over from before, eventually clean it up and make it better

    # TODO:
    # Allow this dir to be passed as an argument, would have to change the logic though
    # Should allow it to just take the output from sides directly though,
    # instead of having a copy in a different dir for this
    prior_dir = lustre_path / "XID_plus" / "prima_cats"


    if catalogue_choice == "blind_v2.2_wiener":
        prior_cat = "blind_merged_v2.2_wiener_p95.csv"
        cat = pd.read_csv(prior_dir / prior_cat) 
        cat = Table.from_pandas(cat)
        
        # THIS ID WILL NOT MATCH THE OTHERS. IT SHOULD BE ADDED AT THE CATALOGUE LEVEL, SOMETHING FOR THE NEXT CATALOGUE VERSION TO FIX
        cat.add_column(np.arange(len(cat)), name="ID", index=0)


    elif catalogue_choice == "euclid_wide":
        prior_cat = "PRIMAv2.2_coadd.fits"

        # Move away from astropy.Table? Very slow for massive catalogues
        # cat = fits.open(prior_dir / prior_cat)[1].data
        # cat = euclid_mass_cut(cat, "wide")

        cat = Table.read(prior_dir / prior_cat) 
        cat.add_column(np.arange(len(cat)), name="ID", index=0)
        cat = euclid_mass_cut(cat, "wide")
    elif catalogue_choice == "euclid_deep":
        prior_cat = "PRIMAv2.2_coadd.fits"
        cat = Table.read(prior_dir / prior_cat) 
        cat.add_column(np.arange(len(cat)), name="ID", index=0)
        cat = euclid_mass_cut(cat, "deep")

    elif catalogue_choice == "euclid_wide_missing":
        prior_cat = "PRIMAv2.2_coadd.fits"
        cat = Table.read(prior_dir / prior_cat)
        cat = euclid_mass_cut(cat, "wide")

        # Mask 10% of sources with 1A1 flux above 0.1 mJy


        rng = np.random.default_rng(0)

        idx = np.where(cat["SPRIMA_1A_1_coadd"] > 0.1e-3)[0]
        n_mask = int(0.25 * len(idx))
        mask_idx = rng.choice(idx, size=n_mask, replace=False)

        final_mask = np.ones(len(cat), dtype=bool)
        final_mask[mask_idx] = False

        table_masked = cat[final_mask]



        cat = table_masked
    
    elif catalogue_choice == "euclid_wide_shark":
        prior_cat = "shark_2sqdeg.txt"
        cat = pd.read_csv(lustre_path / "sides" / "shark" / prior_cat) 
        cat = Table.from_pandas(cat)
        cat = euclid_mass_cut(cat, "wide_shark")

    elif catalogue_choice == "Herschel_10sqdeg_bright250":
        cat = Table.read(lustre_path_prima / "sides/outputs/cat/Herschel_10sqdeg_bright250.fits")
    elif catalogue_choice == "Herschel_10sqdeg_brightish250":
        cat = Table.read(lustre_path_prima / "sides/outputs/cat/Herschel_10sqdeg_brightish250.fits")
    else:
        raise ValueError("prior_choice not recognised.")

    return cat

def get_map(map_choice, bands):
    # For now just copy over from before, eventually clean it up
    # and make it better but fuck me is this messy and horrible

    

    if map_choice == "v1":
        noisy_maps = [lustre_path / f"sides/outputs/maps/v1/noisy/pySIDES_PRIMAv1_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv1.txt").query("band in @bands").npp_Jy.tolist()
    elif map_choice == "v2.1":
        noisy_maps = [lustre_path / f"sides/outputs/maps/v2/2.1/coadd_noisy/pySIDES_PRIMAv2.1_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.1_coadd.txt").query("band in @bands").npp_Jy.tolist()
    elif map_choice == "v2.2":
        noisy_maps = [lustre_path / f"sides/outputs/maps/v2/2.2/coadd_noisy/pySIDES_PRIMAv2.2_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()
    elif map_choice == "v2.2_modelled":
        noisy_maps = [lustre_path / f"sides/outputs/maps/v2/2.2/modelled/noisy/pySIDES_PRIMAv2.2_modelled_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()

        # What an absolutely horrible way of doing this :/
        old_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data**2)) for band in bands]
        new_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/modelled/default/{band}.fits")[0].data**2)) for band in bands]
        npps = [npp*new_area/old_area for npp, old_area, new_area in zip(npps, old_areas, new_areas)]
    # elif map_choice == "v2.2_modelledfwhm10":
    #     noisy_maps = [f"../sides/outputs/maps/v2/2.2/modelledfwhm10/noisy/pySIDES_PRIMAv2.2_modelledfwhm10_{band}_noisy_Jy_beam.fits" for band in bands]
    #     npps = pd.read_csv("../sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()
    elif map_choice == "v2.2_level2_broadened":
        noisy_maps = [lustre_path / f"sides/outputs/maps/v2/2.2/level2_broadened/noisy/pySIDES_PRIMAv2.2_level2_broadened_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()

        # What an absolutely horrible way of doing this :/
        old_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data**2)) for band in bands]
        new_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/level2_broadened/{band}.fits")[0].data**2)) for band in bands]
        npps = [npp*new_area/old_area for npp, old_area, new_area in zip(npps, old_areas, new_areas)]
    elif map_choice == "v2.2_positionaloffset_broadened":
        noisy_maps = [lustre_path / f"sides/outputs/maps/v2/2.2/positionaloffset_broadened/noisy/pySIDES_PRIMAv2.2_positionaloffset_broadened_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()

        # What an absolutely horrible way of doing this :/
        old_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data**2)) for band in bands]
        new_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/positionaloffset_broadened/{band}.fits")[0].data**2)) for band in bands]
        npps = [npp*new_area/old_area for npp, old_area, new_area in zip(npps, old_areas, new_areas)]
    elif map_choice == "v2.2_convolvednoise":
        noisy_maps = [research_path / f"PRIMA_v2p2_coadd_smoothed_noise_maps/{band}_v2p2_smoothed_noise_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()

        # # What an absolutely horrible way of doing this :/
        # old_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data**2)) for band in bands]
        # new_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/positionaloffset_broadened/{band}.fits")[0].data**2)) for band in bands]
        # npps = [npp*new_area/old_area for npp, old_area, new_area in zip(npps, old_areas, new_areas)]
    elif map_choice == "v2.2_shark":
        noisy_maps = [lustre_path / f"sides/outputs/maps/v2/2.2/shark/noisy/pySIDES_PRIMAv2.2_shark_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()

    elif map_choice == "v2.2_MSV2_2.8":
        noisy_maps = [lustre_path / f"sides/outputs/maps/v2/2.2/MSV2_2.8RMS/coadd_noisy/pySIDES_MSV2_2.8RMS_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()

        # What an absolutely horrible way of doing this :/
        old_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data**2)) for band in bands]
        new_areas = [np.sqrt(np.sum(fits.open(f"~/prima/sides/20260416_MSV2_Beam_Profiles/2.8_micron_RMS/{band}.fits")[0].data**2)) for band in bands]
        npps = [npp*new_area/old_area for npp, old_area, new_area in zip(npps, old_areas, new_areas)]

    elif map_choice == "v2.2_MSV2_3.7":
        noisy_maps = [lustre_path / f"sides/outputs/maps/v2/2.2/MSV2_3.7RMS/coadd_noisy/pySIDES_MSV2_3.7RMS_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()

        # What an absolutely horrible way of doing this :/
        old_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data**2)) for band in bands]
        new_areas = [np.sqrt(np.sum(fits.open(f"~/prima/sides/20260416_MSV2_Beam_Profiles/3.7_micron_RMS/{band}.fits")[0].data**2)) for band in bands]
        npps = [npp*new_area/old_area for npp, old_area, new_area in zip(npps, old_areas, new_areas)]


    elif map_choice == "v2.2_MSV2_2.0":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/MSV2_2.0umRMS/coadd_noisy/pySIDES_MSV2_2.0umRMS_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()

        # What an absolutely horrible way of doing this :/
        old_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data**2)) for band in bands]
        new_areas = [np.sqrt(np.sum(fits.open(f"~/prima/sides/MSV2_beams/2.0_micron_RMS/{band}.fits")[0].data**2)) for band in bands]
        npps = [npp*new_area/old_area for npp, old_area, new_area in zip(npps, old_areas, new_areas)]

    elif map_choice == "v2.2_MSV2_3.1":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/MSV2_3.1umRMS/coadd_noisy/pySIDES_MSV2_3.1umRMS_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()

        # What an absolutely horrible way of doing this :/
        old_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data**2)) for band in bands]
        new_areas = [np.sqrt(np.sum(fits.open(f"~/prima/sides/MSV2_beams/3.1_micron_RMS/{band}.fits")[0].data**2)) for band in bands]
        npps = [npp*new_area/old_area for npp, old_area, new_area in zip(npps, old_areas, new_areas)]

    elif map_choice == "v2.2_SV_Airy1.275m":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/SV_Airy1.275m/noisy/pySIDES_SV_Airy1.275m_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()

        # What an absolutely horrible way of doing this :/
        old_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data**2)) for band in bands]
        new_areas = [np.sqrt(np.sum(fits.open(f"~/prima/sides/SV_beams/1.275m/{band}.fits")[0].data**2)) for band in bands]
        npps = [npp*new_area/old_area for npp, old_area, new_area in zip(npps, old_areas, new_areas)]
    elif map_choice == "v2.2_SV_NominalPixel1.029":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/SV_NominalPixel1.029/noisy/pySIDES_SV_NominalPixel1.029_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()

        # # What an absolutely horrible way of doing this :/
        # old_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data**2)) for band in bands]
        # new_areas = [np.sqrt(np.sum(fits.open(f"~/prima/sides/SV_beams/1.275m/{band}.fits")[0].data**2)) for band in bands]
        # npps = [npp*new_area/old_area for npp, old_area, new_area in zip(npps, old_areas, new_areas)]
    elif map_choice == "v2.2_SV_Airy1.654m":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/SV_Airy1.654m/noisy/pySIDES_SV_Airy1.654m_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()

        # What an absolutely horrible way of doing this :/
        old_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data**2)) for band in bands]
        new_areas = [np.sqrt(np.sum(fits.open(f"~/prima/sides/SV_beams/1.654m/{band}.fits")[0].data**2)) for band in bands]
        npps = [npp*new_area/old_area for npp, old_area, new_area in zip(npps, old_areas, new_areas)]
    elif map_choice == "v2.2_SV_NominalPixel1.092":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/SV_NominalPixel1.092/noisy/pySIDES_SV_NominalPixel1.092_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()
    elif map_choice == "v2.2_SV_NominalPixel1.131": 
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/SV_NominalPixel1.131/noisy/pySIDES_SV_NominalPixel1.131_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()
    elif map_choice == "v2.2_SV_NominalPixel0.897":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/SV_NominalPixel0.897/noisy/pySIDES_SV_NominalPixel0.897_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()
    elif map_choice == "v2.2_SV_Airy1.171m":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/SV_Airy1.171m/noisy/pySIDES_SV_Airy1.171m_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()

        # What an absolutely horrible way of doing this :/
        old_areas = [np.sqrt(np.sum(fits.open(lustre_path / f"sides/beams/v2/coadd/{band}.fits")[0].data**2)) for band in bands]
        new_areas = [np.sqrt(np.sum(fits.open(f"~/prima/sides/SV_beams/1.171m/{band}.fits")[0].data**2)) for band in bands]
        npps = [npp*new_area/old_area for npp, old_area, new_area in zip(npps, old_areas, new_areas)]
    elif map_choice == "v2.2_SV_NominalPixel1.248":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/SV_NominalPixel1.248/noisy/pySIDES_SV_NominalPixel1.248_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()
    elif map_choice == "v2.2_SV_NominalPixel1.248Noisy":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/SV_NominalPixel1.248Noisy/noisy/pySIDES_SV_NominalPixel1.248Noisy_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()
        npps = [npp*2.43189737392 for npp in npps]
        print("npp * 2.43189737392")
    elif map_choice == "v2.2_WideDepth":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/PRIMAv2.2_WideDepth/noisy/pySIDES_PRIMAv2.2_WideDepth_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()
        npps = [npp*np.sqrt(10) for npp in npps]
        print("npp * np.sqrt(10)")
    elif map_choice == "v2.2_SV_CBEPixel1.029":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/SV_CBEPixel1.029/noisy/pySIDES_SV_CBEPixel1.029_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()
    elif map_choice == "v2.2_SV_CBEPixel1.248":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/SV_CBEPixel1.248/noisy/pySIDES_SV_CBEPixel1.248_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = pd.read_csv(lustre_path / "sides/inputs/PRIMAgerv2.2_coadd.txt").query("band in @bands").npp_Jy.tolist()
    elif map_choice == "HerschelDeep":
        noisy_maps = [lustre_path / f"prima_data/sides/outputs/maps/HerschelDeep/noisy/pySIDES_HerschelDeep_{band}_noisy_Jy_beam.fits" for band in bands]
        npps = [0.006996113504272006, 0.005696499857606891, 0.008307884786323006]

    else:
        raise ValueError("map_choice not recognised.")

    return noisy_maps, npps

def bkg_subtraction(im, subtraction_type, band):

    # Should be impossible since it wouldn't be called but idk
    if subtraction_type is None:
        return im

    if subtraction_type == "nebuliser":
        print("Beginning Nebuliser-like subtraction.", flush = True)

        bkg = estimate_background(im, 75)
        return im - bkg

    if subtraction_type == "photutils":
        print("Beginning photutils subtraction.", flush = True)

        from astropy.stats import SigmaClip
        from photutils.background import Background2D, MedianBackground

        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()

        if "1B" in band:
            mean_shape = (15,15)
        else:
            mean_shape = (25,25)

        bkg = Background2D(im, mean_shape, filter_size=(3, 3),
                        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

        return im - bkg.background
    
    if subtraction_type == "annulus":
        print("Beginning annulus subtraction.", flush = True)

        if "2A" in band:
            fwhm = int(10.82/2.3)
        elif "2B" in band:
            fwhm = int(14.79/2.3)
        elif "2C" in band:
            fwhm = int(21.43/2.3)
        elif "2D" in band:
            fwhm = int(27.5/2.3)
        else:
            raise ValueError("Annulus is not set up for PHI bands yet.")

        return annular_median_map(im, 2*fwhm, 6*fwhm)
    
    raise ValueError(f"Subtraction method '{subtraction_type}' not recognised.")

def euclid_mass_cut(cat, survey_type):
    """
    Perform the Euclid 95% completeness mass cut. Deep is taken to be 0.8 dex 
    smaller than Wide
    """
    from scipy.optimize import curve_fit
    from scipy.interpolate import interp1d

    if survey_type not in ["wide", "deep", "wide_shark"]:
        raise ValueError("survey_type must be either 'wide', 'deep', or 'wide_shark'")

    

    euclid_df = pd.read_csv(lustre_path / "XID_plus" / "euclid_wide_cut.csv",
                            header = 0, names = ["z", "logM"], dtype=float)
    
    if survey_type == "deep":
        euclid_df["logM"] -= 0.8

    # Example model: log(Mlim) = a*log(1+z) + b
    def model(z, a, b):
        return a * np.log10(1 + z) + b

    # for z < 3
    low_z_func = interp1d(euclid_df.z, euclid_df.logM)
    # extrapolating
    high_z_popt, _ = curve_fit(model, euclid_df.z[euclid_df.z > 1.], euclid_df.logM[euclid_df.z > 1.])
    def high_z_func(z, a = high_z_popt[0], b = high_z_popt[1]):
        return a * np.log10(1 + z) + b
    
    if "shark" not in survey_type:
        low_z_cut = np.empty_like(cat["redshift"], dtype = bool)
        low_z_cut[cat["redshift"] > np.max(euclid_df.z)] = False
        low_z_cut[cat["redshift"] <= np.max(euclid_df.z)] = np.log10(cat["Mstar"][cat["redshift"] <= np.max(euclid_df.z)]) >= low_z_func(cat["redshift"][cat["redshift"] <= np.max(euclid_df.z)])


        high_z_cut = np.empty_like(cat["redshift"], dtype = bool)
        high_z_cut[cat["redshift"] <= np.max(euclid_df.z)] = False
        high_z_cut[cat["redshift"] > np.max(euclid_df.z)] = np.log10(cat["Mstar"][cat["redshift"] > np.max(euclid_df.z)]) >= high_z_func(cat["redshift"][cat["redshift"] > np.max(euclid_df.z)])

    else: # Horrible way of doing this but if it works it works
        low_z_cut = np.empty_like(cat["redshift[cos]"], dtype = bool)
        low_z_cut[cat["redshift[cos]"] > np.max(euclid_df.z)] = False
        low_z_cut[cat["redshift[cos]"] <= np.max(euclid_df.z)] = cat["log10(mstar)"][cat["redshift[cos]"] <= np.max(euclid_df.z)] >= low_z_func(cat["redshift[cos]"][cat["redshift[cos]"] <= np.max(euclid_df.z)])


        high_z_cut = np.empty_like(cat["redshift[cos]"], dtype = bool)
        high_z_cut[cat["redshift[cos]"] <= np.max(euclid_df.z)] = False
        high_z_cut[cat["redshift[cos]"] > np.max(euclid_df.z)] =cat["log10(mstar)"][cat["redshift[cos]"] > np.max(euclid_df.z)] >= high_z_func(cat["redshift[cos]"][cat["redshift[cos]"] > np.max(euclid_df.z)])

    mass_cut = low_z_cut | high_z_cut

    # cat[mass_cut].write(f"euclid_{survey_type}.fits", format="fits")
    # print("saved file")
    return cat[mass_cut]

def annular_median_map(map_data, r_in, r_out, sigma_clip=3):
    # This is just a quick test with the help of chatgpt. if it seems to work look more into it
    ny, nx = map_data.shape
    output_map = np.zeros_like(map_data)
    
    # Create relative indices for annulus
    y_offsets, x_offsets = np.indices((2*r_out+1, 2*r_out+1)) - r_out
    r_grid = np.sqrt(x_offsets**2 + y_offsets**2)
    annulus_mask = (r_grid >= r_in) & (r_grid <= r_out)
    
    for y0 in range(ny):
        # tstart = time()
        for x0 in range(nx):
            # Define local patch
            y_min = max(y0-r_out, 0)
            y_max = min(y0+r_out+1, ny)
            x_min = max(x0-r_out, 0)
            x_max = min(x0+r_out+1, nx)
            
            patch = map_data[y_min:y_max, x_min:x_max]
            
            # Adjust mask for edge patches
            mask = annulus_mask[
                (y_min-(y0-r_out)):(y_max-(y0-r_out)),
                (x_min-(x0-r_out)):(x_max-(x0-r_out))
            ]
            
            annulus_pixels = patch[mask]
            
            # Sigma-clipped median
            _, median, _ = sigma_clipped_stats(annulus_pixels, sigma=sigma_clip)
            
            # Subtract median
            output_map[y0, x0] = map_data[y0, x0] - median
        # tstop = time()
        # print(tstop-tstart)
    return output_map

def estimate_background(image, N=30, lower_sigma=-10, upper_sigma=3, iterations=3):
    """
    Estimate the background of an image using median filtering and boxcar smoothing with sigma clipping.

    Parameters:
    -----------
    image : 2D numpy array
        Input intensity map.
    N : int
        Size of the median filter window (N x N).
    lower_sigma : float
        Lower threshold in sigma units for pixel rejection.
    upper_sigma : float
        Upper threshold in sigma units for pixel rejection.
    iterations : int
        Number of iterations for sigma clipping.

    Returns:
    --------
    background : 2D numpy array
        Estimated background map.
    """
    # Initialize the mask (True = use pixel, False = ignore pixel)
    mask = np.ones_like(image, dtype=bool)
    image_work = image.copy()

    for _ in range(iterations):
        # Compute local median using N x N window
        median_map = median_filter(image_work, size=N, mode='reflect')

        # Compute local std (sigma) using median absolute deviation as robust estimator
        diff = image_work - median_map
        sigma = np.std(diff[mask])

        # Sigma clipping: update mask
        mask = (diff > lower_sigma * sigma) & (diff < upper_sigma * sigma)

        # Replace rejected pixels with median for next iteration
        image_work[~mask] = median_map[~mask]

    # Final median map after iterations
    median_map = median_filter(image_work, size=N, mode='reflect')

    # Smooth median map using boxcar (uniform) filter with size N/2
    box_size = max(1, N // 2)
    background = uniform_filter(median_map, size=box_size, mode='reflect')

    return background


### Posterior/modelling analysis functions

def limiting_flux(f_xid, f_true, nbins=50):
    f_ratio = f_xid/f_true

    # bmin = min(np.log10(f_true))
    # bmax = max(np.log10(f_true))

    bmin = -2
    bmax = 2

    bstep = (bmax - bmin)/nbins
    bins = np.arange(bmin,bmax+bstep,bstep)

    bin_width = (bins[1] - bins[0])
    bin_centers = bins[1:] - bin_width/2
    counts = np.zeros(len(bin_centers))
    mad = np.zeros(len(bin_centers))
    rms = np.zeros(len(bin_centers))

    for ii, i in enumerate(bins):
        if i == bins[-1]:
            pass
        else:
            ybin = f_ratio[(f_true >= np.power(10,bins[ii])) & (f_true < np.power(10,bins[ii+1]))]
            xbin = f_true[(f_true >= np.power(10,bins[ii])) & (f_true < np.power(10,bins[ii+1]))]
            counts[ii] = len(ybin)
            if len(ybin) > 0:
                ybin_mad = 1.4826 * np.percentile(np.abs(ybin - np.percentile(ybin,50)),50) # no need to do ybin-1, they cancel out

                mad[ii] = ybin_mad
                ybin_rms = np.sqrt(np.mean((ybin - 1)**2.0))
                rms[ii] = ybin_rms
    return mad, rms, counts, bin_centers

def get_lim(yarr, xarr, counts, lim, min_count = 10):
    if max(yarr) < lim:
        return None
    else:
        yarr = yarr[counts > min_count] #mad
        xarr = xarr[counts > min_count] #flux
        gt_lim_1st = yarr[yarr >= lim][-1]
        ind_gt_lim_1st = np.where(yarr == gt_lim_1st)[0][0]

        lim_line = yarr[ind_gt_lim_1st:ind_gt_lim_1st+2]
        lim_bins = xarr[ind_gt_lim_1st:ind_gt_lim_1st+2]
        interp = interp1d(lim_line,lim_bins)

        try:
            lim_flux = interp(lim)
            return lim_flux
        except ValueError:
            return None

def get_lim_log(yarr, xarr, counts, lim, min_count = 10):
    if max(yarr) < lim:
        return min(xarr)
    else:

        yarr = yarr[counts > min_count] #mad
        xarr = xarr[counts > min_count] #flux

        xarr, yarr, lim = np.log10(xarr), np.log10(yarr), np.log10(lim)

        gt_lim_1st = yarr[yarr >= lim][-1]
        ind_gt_lim_1st = np.where(yarr == gt_lim_1st)[0][0]
        lim_line = yarr[ind_gt_lim_1st:ind_gt_lim_1st+2]
        lim_bins = xarr[ind_gt_lim_1st:ind_gt_lim_1st+2]
        interp = interp1d(lim_line,lim_bins)
        return 10**interp(lim)        

def relative_error(f_xid, f_true, nbins=50):
    rel_error = (f_xid-f_true)/f_true
    
    log_true = np.log10(f_true)

    bmin = -2
    bmax = 2

    bstep = (bmax - bmin)/nbins
    bins = np.arange(bmin,bmax+bstep,bstep)

    bin_width = (bins[1] - bins[0])
    bin_centers = bins[1:] - bin_width/2
    counts = np.zeros(len(bin_centers))

    medians = np.zeros(len(bin_centers))
    p5 = np.zeros(len(bin_centers))
    p16 = np.zeros(len(bin_centers))
    p84 = np.zeros(len(bin_centers))
    p95 = np.zeros(len(bin_centers))

    # loop over bins
    for i in range(len(bin_centers)):
        mask = (log_true >= bins[i]) & (log_true < bins[i+1])

        if np.any(mask):
            vals = rel_error[mask]

            counts[i] = len(vals)
            medians[i] = np.median(vals)
            p5[i]  = np.percentile(vals, 5)
            p16[i] = np.percentile(vals, 16)
            p84[i] = np.percentile(vals, 84)
            p95[i] = np.percentile(vals, 95)

    return bin_centers, counts, medians, p5, p16, p84, p95

def non_relative_error(f_xid, f_true, nbins=50):
    diff = f_xid-f_true
    
    log_true = np.log10(f_true)

    bmin = -2
    bmax = 2

    bstep = (bmax - bmin)/nbins
    bins = np.arange(bmin,bmax+bstep,bstep)

    bin_width = (bins[1] - bins[0])
    bin_centers = bins[1:] - bin_width/2
    counts = np.zeros(len(bin_centers))

    medians = np.zeros(len(bin_centers))
    p5 = np.zeros(len(bin_centers))
    p16 = np.zeros(len(bin_centers))
    p84 = np.zeros(len(bin_centers))
    p95 = np.zeros(len(bin_centers))

    # loop over bins
    for i in range(len(bin_centers)):
        mask = (log_true >= bins[i]) & (log_true < bins[i+1])

        if np.any(mask):
            vals = diff[mask]

            counts[i] = len(vals)
            medians[i] = np.median(vals)
            p5[i]  = np.percentile(vals, 5)
            p16[i] = np.percentile(vals, 16)
            p84[i] = np.percentile(vals, 84)
            p95[i] = np.percentile(vals, 95)

    return bin_centers, counts, medians, p5, p16, p84, p95

def bootstrap_limiting_flux(f_xid, f_true, lim_value = 0.2, nboot = 1000, nbins = 50):
    boot_lims = []

    N = len(f_true)

    for _ in range(nboot):
        idx = np.random.choice(N, size=N, replace=True)

        f_xid_boot = f_xid[idx]
        f_true_boot = f_true[idx]

        mad, rms, counts, bins = limiting_flux(f_xid_boot, f_true_boot, nbins=nbins)

        lim_flux = get_lim(mad, np.power(10,bins), counts, lim_value)
        if lim_flux is not None:
            boot_lims.append(lim_flux)

    return np.array(boot_lims)

def continuous_limiting_flux(f_xid, f_true, dlogflux=0.1, npoints=200):

    f_ratio = f_xid / f_true
    logf = np.log10(f_true)

    log_grid = np.linspace(-2, 2, npoints)

    mad = np.zeros_like(log_grid)
    rms = np.zeros_like(log_grid)
    counts = np.zeros_like(log_grid)

    for i, lg in enumerate(log_grid):

        mask = np.abs(logf - lg) <= dlogflux

        y = f_ratio[mask]

        counts[i] = len(y)

        if len(y) > 0:
            mad[i] = 1.4826 * np.percentile(
                np.abs(y - np.percentile(y,50)), 
                50
            )

            rms[i] = np.sqrt(np.mean((y-1)**2))

        else:
            mad[i] = np.nan
            rms[i] = np.nan

    return mad, rms, counts, log_grid

def bootstrap_continuous(f_xid, f_true, nboot=1000):

    all_mads = []
    all_lims = []

    N = len(f_true)

    for _ in tqdm(range(nboot)):
    # for _ in range(nboot):

        idx = np.random.choice(N, N, replace=True)

        mad, rms, counts, loggrid = continuous_limiting_flux(f_xid[idx], f_true[idx])

        all_mads.append(mad)

        lim = get_lim(mad, 10**loggrid, counts, 0.2)

        if lim is not None:
            all_lims.append(lim)

    return np.array(all_mads), np.array(all_lims)



def get_merged_data(run_name, band, ids_large,
                    order = 11,
                    order_large = 7,
                    output_dir = None
                    ):

    small_tiles_per_large = int(4**(order - order_large))


    merged_f_true = np.empty(0)
    merged_f_xid = np.empty(0)
    merged_f_xid_l = np.empty(0)
    merged_f_xid_u = np.empty(0)
    merged_ra = np.empty(0)
    merged_dec = np.empty(0)

    for large_tile in ids_large:
        if len(ids_large) > 1:
            print(f"Starting large tile {large_tile}")

        large_f_true = np.empty(0)
        large_f_xid = np.empty(0)
        large_f_xid_l = np.empty(0)
        large_f_xid_u = np.empty(0)
        large_ra = np.empty(0)
        large_dec = np.empty(0)

        merged_file = f"merged_xid_{run_name}_{band}_{large_tile}.csv"
        # merged_file = f"merged_xid_{run_name}_{band}.csv" # OLD VERSION
        merged_path = output_dir / merged_file

        # If merged file exists, load and append to merged arrays without looking for small tiles.
        if os.path.isfile(merged_path):
            print("Loading merged file...")
            df = pd.read_csv(merged_path)

            if "is_central" in df.columns:
                df = df[df.is_central]

            merged_f_true = np.concatenate((merged_f_true, df.f_true))
            merged_f_xid = np.concatenate((merged_f_xid, df.f_xid))
            merged_f_xid_l = np.concatenate((merged_f_xid_l, df.f_xid_l))
            merged_f_xid_u = np.concatenate((merged_f_xid_u, df.f_xid_u))
            merged_ra = np.concatenate((merged_ra, df.ra))
            merged_dec = np.concatenate((merged_dec, df.dec))

            continue
        
        num_missing_small = 0 # number of small tiles missing

        for small_tile in tqdm(range(small_tiles_per_large)):
            small_tile_id = large_tile*small_tiles_per_large + small_tile

            small_file = f"xid_{run_name}_{band}_tile{small_tile_id}_order{order}_large{order_large}_summary.csv"
            small_path = output_dir / small_file

            # If small tile file exists, load it and append to the merged arrays
            if os.path.isfile(small_path):
                df = pd.read_csv(small_path)

                if "is_central" in df.columns:
                    df = df[df.is_central]

                large_f_true = np.concatenate((large_f_true, df.f_true))
                large_f_xid = np.concatenate((large_f_xid, df.f_xid))
                large_f_xid_l = np.concatenate((large_f_xid_l, df.f_xid_l))
                large_f_xid_u = np.concatenate((large_f_xid_u, df.f_xid_u))
                large_ra = np.concatenate((large_ra, df.ra))
                large_dec = np.concatenate((large_dec, df.dec))

                continue
            
            # If small tile file doesn't exist, add to missing num
            num_missing_small += 1

        # If no small files loaded, say so
        if num_missing_small == small_tiles_per_large:
            print(f"No data for {large_tile}")
            continue

        # If all small files loaded, create merged file
        if num_missing_small == 0:
            print("Creating merged file...")
            df = pd.DataFrame(data = {"f_true": large_f_true,
                                    "f_xid": large_f_xid,
                                    "f_xid_l": large_f_xid_l,
                                    "f_xid_u": large_f_xid_u,
                                    "ra": large_ra,
                                    "dec": large_dec
                                    })
            df.to_csv(merged_path, index = False)
        else:
            print(f"Missing {num_missing_small}/{small_tiles_per_large}")

        merged_f_true = np.concatenate((merged_f_true, large_f_true))
        merged_f_xid = np.concatenate((merged_f_xid, large_f_xid))
        merged_f_xid_l = np.concatenate((merged_f_xid_l, large_f_xid_l))
        merged_f_xid_u = np.concatenate((merged_f_xid_u, large_f_xid_u))
        merged_ra = np.concatenate((merged_ra, large_ra))
        merged_dec = np.concatenate((merged_dec, large_dec))

    if len(merged_f_true) == 0:
        print(f"No data for {band}\n")
        return None

    print(f"{len(merged_f_true)} sources")

    return merged_f_true, merged_f_xid, merged_f_xid_l, merged_f_xid_u, merged_ra, merged_dec



