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

import astropy.convolution as conv
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

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
    subtraction_type: None|str = None):
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
        col = cat[f"S{band}"]
        cat[f"S{band}"] = np.where(col == 0, 1e-12, col)

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
        
        # Perform background subtraction
        if subtraction_type is not None:
            im = bkg_subtraction(im, subtraction_type, band)

        # Initialise prior
        influx_mu = np.array(cat[f"S{band}"]) * 1e3
        influx_sigma = np.array(cat[f"S{band}"]) * 1e3

        prior = xidplus.prior(im, error_im, header, header)
        prior.stepwise_prima_prior_cat(inra, indec, cat, flux_mu = influx_mu, flux_sigma = influx_sigma)
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
    output: bool = True
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
        flux_stepwise (bool):
            Whether to use stepwise methodology in the modelling. If `True`,
            there needs to be a prior at the previous band already run, except
            for the first band.
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
            outfolder = outfolder / output_name

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
    
    # Trim prior to small tile area
    moc = moc_routines.get_fitting_region(order, id_small_tile)
    prior.moc = moc
    prior.cut_down_prior()

    print(f"{prior.nsrc} sources in prior")
    print(f"{prior.snpix} pixels in prior")

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
    

    if index_band > 0 and flux_stepwise:
        print("\nLoading previous posterior...")
        prev_band = bands[index_band - 1]

        prev_posterior_file = outfolder / "posterior" / f"xid_{output_name}_{prev_band}_tile{id_small_tile}_order{order}_large{order_large}_posterior.pkl"
        
        with open(prev_posterior_file, "rb") as f:
            data = pickle.load(f)

            prev_posterior = data["posterior"]

        if index_band < 6:
            correction_factor = 1.0
        elif index_band < 12:
            correction_factor = 1.3
        else:
            correction_factor = 2.0

        prior.prior_flux_mu = np.percentile(prev_posterior.samples['src_f'][:,0,:], 50.0, axis=0) * correction_factor
        prior.prior_flux_sigma = np.percentile(prev_posterior.samples['src_f'][:,0,:], 50.0, axis=0) * correction_factor
        print("Stepwise ready")

    # Make prior.prior_flux_lower != 0 
    prior.prior_flux_lower = np.full((prior.sra.shape), 1e-9)

    ### Runs numpyro fitting on the small tile
    print("\nStarting modelling", flush = True)
    tstart = time()
    fit = single_band([prior], flux_prior, cirrus_map, num_samples, num_warmup, num_chains, chain_method)

    samples = fit.get_samples() # get samples from the fit (not sure if this is needed when using the numpyro method. I think left over)
    posterior = xidplus.posterior.posterior_numpyro(fit, [prior])

    tstop = time()
    model_time = tstop - tstart
    print(f"\nModelling completed in: {model_time:.3f} s", flush = True)

    # Diagnostics to check mcmc run correctly, should be 0 for all three
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
        kept_sources = moc_routines.sources_in_tile([id_small_tile],order,prior.sra,prior.sdec)

        ra = prior.sra[kept_sources]
        dec = prior.sdec[kept_sources]
        f_true = prior.prior_flux_mu[kept_sources]

        f_xid = np.percentile(posterior.samples['src_f'][:,0,:], 50.0, axis=0)[kept_sources]
        f_xid_l = np.percentile(posterior.samples['src_f'][:,0,:], 15.9, axis=0)[kept_sources]
        f_xid_u = np.percentile(posterior.samples['src_f'][:,0,:], 84.1, axis=0)[kept_sources]

        ### Output results to pkl
        with open(outfile_prior, "wb") as f:
            pickle.dump({"prior": prior}, f)
            
        with open(outfile_posterior, "wb") as f:
            pickle.dump({"posterior": posterior,
                        "kept_sources": kept_sources}, f)


        df = pd.DataFrame(data = {"f_true": f_true,
                                "f_xid": f_xid,
                                "f_xid_l": f_xid_l,
                                "f_xid_u": f_xid_u,
                                "ra": ra,
                                "dec": dec
                                })
            
        df.to_csv(outfile_summary, index = False)

    print("Completed")

### Modelling helper functions

def single_model(
    priors,
    flux_prior: float|None = None,
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

        # probably should just check if not none, no poin for the bool
        if cirrus_map is not None:
            cirrus_scale = numpyro.sample('cirrus_scale', dist.Uniform(0, 100))

        with numpyro.plate('nsrc', priors[0].nsrc):

            # if flux_prior is not None and flux_prior != 0:
            #     src_f = numpyro.sample('src_f', dist.TruncatedNormal(flux_mu, flux_sigma * flux_prior, low = flux_lower, high = flux_upper))
            # else:
            #     src_f = numpyro.sample('src_f', dist.Uniform(flux_lower, flux_upper))


            # Transform priors to log-space
            if flux_prior is not None and flux_prior != 0:
                log_src_f = numpyro.sample('log_src_f', dist.TruncatedNormal(log_flux_mu, log_flux_sigma * flux_prior, low=log_flux_lower, high=log_flux_upper))
            else:
                log_src_f = numpyro.sample('log_src_f', dist.Uniform(log_flux_lower, log_flux_upper)) # This is log uniform now

            # Convert back to linear space for the model
            src_f = numpyro.deterministic('src_f', jnp.exp(log_src_f))



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
    cirrus_map: np.ndarray|None = None,
    num_samples = 500,
    num_warmup = 500,
    num_chains = 4,
    chain_method = "parallel"
    ):
    if jax.default_backend() == "gpu":
        print("GPU detected, running with GPU.")
        print(f"{jax.device_count()} GPU(s) detected.")
    else:
        print("GPU not detected, running with CPU.")
        numpyro.set_host_device_count(num_chains)

    nuts_kernel = NUTS(single_model, init_strategy = numpyro.infer.init_to_median())
    rng_key = random.PRNGKey(0)

    print("\nMODELLING PARAMETERS:")
    print(f"{flux_prior = }")
    print(f"{type(cirrus_map) = }")
    print(f"{num_samples = }")
    print(f"{num_warmup = }")
    print(f"{num_chains = }")
    print(f"{chain_method = }")

    mcmc = MCMC(nuts_kernel, num_samples = num_samples, num_warmup = num_warmup, num_chains = num_chains, chain_method = chain_method)
    mcmc.run(rng_key, priors, flux_prior, cirrus_map, extra_fields = ('potential_energy', 'energy',))

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
        fcat = pd.read_csv(prior_dir / prior_cat) 
        cat = Table.from_pandas(fcat)
    elif catalogue_choice == "euclid_wide":
        prior_cat = "PRIMAv2.2_coadd.fits"
        fcat = Table.read(prior_dir / prior_cat) 
        fcat = euclid_mass_cut(fcat, "wide")
        cat = fcat
    elif catalogue_choice == "euclid_deep":
        prior_cat = "PRIMAv2.2_coadd.fits"
        fcat = Table.read(prior_dir / prior_cat) 
        fcat = euclid_mass_cut(fcat, "deep")
        cat = fcat

    elif catalogue_choice == "euclid_wide_missing":
        prior_cat = "PRIMAv2.2_coadd.fits"
        fcat = Table.read(prior_dir / prior_cat) 
        fcat = euclid_mass_cut(fcat, "wide")

        # Mask 10% of sources with 1A1 flux above 0.1 mJy


        rng = np.random.default_rng(0)

        idx = np.where(fcat["SPRIMA_1A_1_coadd"] > 0.1e-3)[0]
        n_mask = int(0.25 * len(idx))
        mask_idx = rng.choice(idx, size=n_mask, replace=False)

        final_mask = np.ones(len(fcat), dtype=bool)
        final_mask[mask_idx] = False

        table_masked = fcat[final_mask]



        cat = table_masked
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

    if survey_type not in ["wide", "deep"]:
        raise ValueError("survey_type must be either 'wide' or 'deep'")

    

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

    low_z_cut = np.empty_like(cat["redshift"], dtype = bool)
    low_z_cut[cat["redshift"] > np.max(euclid_df.z)] = False
    low_z_cut[cat["redshift"] <= np.max(euclid_df.z)] = np.log10(cat["Mstar"][cat["redshift"] <= np.max(euclid_df.z)]) >= low_z_func(cat["redshift"][cat["redshift"] <= np.max(euclid_df.z)])


    high_z_cut = np.empty_like(cat["redshift"], dtype = bool)
    high_z_cut[cat["redshift"] <= np.max(euclid_df.z)] = False
    high_z_cut[cat["redshift"] > np.max(euclid_df.z)] = np.log10(cat["Mstar"][cat["redshift"] > np.max(euclid_df.z)]) >= high_z_func(cat["redshift"][cat["redshift"] > np.max(euclid_df.z)])

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

    bmin = min(np.log10(f_true))
    bmax = max(np.log10(f_true))

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
        return min(xarr)
    else:
        yarr = yarr[counts > min_count] #mad
        xarr = xarr[counts > min_count] #flux
        gt_lim_1st = yarr[yarr >= lim][-1]
        ind_gt_lim_1st = np.where(yarr == gt_lim_1st)[0][0]
        lim_line = yarr[ind_gt_lim_1st:ind_gt_lim_1st+2]
        lim_bins = xarr[ind_gt_lim_1st:ind_gt_lim_1st+2]
        interp = interp1d(lim_line,lim_bins)
        return interp(lim)