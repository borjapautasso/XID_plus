import numpy as np
from astropy import wcs
from xidplus import moc_routines
import jax


class prior(object):
    def __init__(self, im, nim, imphdu, imhdu, moc=None):

        # ---for any bad pixels set map pixel to zero and uncertianty to 1----
        """Initiate prior class

        :param im: image map from fits file
        :param nim: noise map from fits file
        :param imphdu: Primary header associated with fits file
        :param imhdu: header associated with image map
        :param moc: (default=None) Multi-Order Coverage map of area being fit
        """
        bad = np.logical_or(np.logical_or
                            (np.invert(np.isfinite(im)),
                             np.invert(np.isfinite(nim))), (nim == 0))
        if (bad.sum() > 0):
            im[bad] = 0.
            nim[bad] = 1.
        self.imhdu = imhdu
        wcs_temp = wcs.WCS(self.imhdu)
        self.imphdu = imphdu
        self.imhdu = imhdu

        x_pix, y_pix = np.meshgrid(np.arange(0, wcs_temp.pixel_shape[0]), np.arange(0, wcs_temp.pixel_shape[1]))
        self.sx_pix = x_pix.flatten()
        self.sy_pix = y_pix.flatten()
        self.snim = nim.flatten()
        self.sim = im.flatten()
        self.snpix = self.sim.size
        if moc is not None:
            self.moc = moc
            self.cut_down_map()

    def prior_cat(self, ra, dec, prior_cat_file, flux_lower=None, flux_upper=None, ID=None, moc=None,z_median=None,z_sig=None):
        """Input info for prior catalogue

        :param ra: Right ascension (JD2000) of sources
        :param dec: Declination (JD2000) of sources
        :param prior_cat_file: filename of catalogue
        :param flux_lower: lower limit of flux for each source
        :param flux_upper: upper limit of flux for each source
        :param ID: HELP_ID for each source
        :param moc: Multi-Order Coverage map
        :param z_median: median of redshift pdf
        :param z_sig: sigma of redshift pdf
        """
        # get positions of sources in terms of pixels
        wcs_temp = wcs.WCS(self.imhdu)
        sx, sy = wcs_temp.wcs_world2pix(ra, dec, 0)
        if moc is None:
            cat_moc = moc_routines.create_MOC_from_cat(ra, dec)
        else:
            cat_moc = moc


        # Redefine prior list so it only contains sources in the map
        self.sx = sx
        self.sy = sy
        self.sra = ra
        self.sdec = dec
        self.nsrc = self.sra.size
        self.prior_cat_file = prior_cat_file
        if flux_lower is None:
            flux_lower = np.full((ra.size), 0.00)
            flux_upper = np.full((ra.size), 1000.0)
        self.prior_flux_lower = flux_lower
        self.prior_flux_upper = flux_upper
        if z_median is not None:
            self.z_median=z_median
            self.z_sig=z_sig

        if ID is None:
            ID = np.arange(1, ra.size + 1, dtype='int64')
        self.ID = ID

        self.stack = np.full(self.nsrc, False)
        try:
            self.moc = self.moc.intersection(cat_moc)
        except AttributeError as e:
            self.moc=cat_moc

        self.cut_down_prior()

    def prior_cat_stack(self, ra, dec, prior_cat, flux_lower=None, flux_upper=None,ID=None):
        """Input info for prior catalogue of sources being stacked

        :param ra: Right ascension (JD2000) of sources
        :param dec: Declination (JD2000) of sources
        :param prior_cat: filename of catalogue
        :param ID: HELP_ID for each source
        """
        wcs_temp = wcs.WCS(self.imhdu)
        sx, sy = wcs_temp.wcs_world2pix(ra, dec, 0)


        # Redefine prior list so it only contains sources in the map

        # Redefine prior list so it only contains sources in the map
        self.sx = np.append(self.sx, sx)
        self.sy = np.append(self.sy, sy)
        self.sra = np.append(self.sra, ra)
        self.sdec = np.append(self.sdec, dec)
        self.nstack = ra.size
        self.nsrc = self.sra.size
        self.stack = np.append(self.stack, np.full((self.nstack), True))
        if ID is None:
            ID = np.arange(1, ra.size + 1, dtype='int64')
        self.ID = np.append(self.ID, ID)
        if flux_lower is None:
            flux_lower = np.full((ra.size), 0.00)
            flux_upper = np.full((ra.size), 1000.0)
            self.prior_flux_lower = np.append(self.prior_flux_lower,flux_lower)
            self.prior_flux_upper = np.append(self.prior_flux_upper,flux_upper)

        self.cut_down_prior()

    def stepwise_prima_prior_cat(self, ra, dec, prior_cat_file, flux_mu=None, flux_sigma=None, ID=None, moc=None,z_median=None,z_sig=None, prior_mstar=None):
        """Input info for prior catalogue

        :param ra: Right ascension (JD2000) of sources
        :param dec: Declination (JD2000) of sources
        :param prior_cat_file: filename of catalogue
        :param flux_lower: lower limit of flux for each source
        :param flux_upper: upper limit of flux for each source
        :param ID: HELP_ID for each source
        :param moc: Multi-Order Coverage map
        :param z_median: median of redshift pdf
        :param z_sig: sigma of redshift pdf
        """
        # get positions of sources in terms of pixels
        wcs_temp = wcs.WCS(self.imhdu)
        sx, sy = wcs_temp.wcs_world2pix(ra, dec, 0)
        if moc is None:
            cat_moc = moc_routines.create_MOC_from_cat(ra, dec)
        else:
            cat_moc = moc


        # Redefine prior list so it only contains sources in the map
        self.sx = sx
        self.sy = sy
        self.sra = ra
        self.sdec = dec
        self.nsrc = self.sra.size
        self.prior_cat_file = prior_cat_file
        self.prior_flux_lower = np.full((ra.size), 0.00)
        self.prior_flux_upper = np.full((ra.size), 1000.0)
        self.prior_flux_mu = flux_mu
        self.prior_flux_sigma = flux_sigma
        # self.prior_z_mu = z_mu
        # self.prior_z_sigma = z_sigma
        if z_median is not None:
            self.z_median=z_median
            self.z_sig=z_sig
        if prior_mstar is not None:
            self.prior_mstar = prior_mstar

        if ID is None:
            ID = np.arange(1, ra.size + 1, dtype='int64')
        self.ID = ID

        self.stack = np.full(self.nsrc, False)
        try:
            self.moc = self.moc.intersection(cat_moc)
        except AttributeError as e:
            self.moc=cat_moc

        self.cut_down_prior()

    def set_moc(self, moc):
        self.moc = moc
        
    def set_prf(self, prf, pindx, pindy):
        """Add prf array and corresponding x and y scales (in terms of pixels in map)

        :param prf: n x n array, where n is an odd number, and the centre of the prf is at the centre of the array
        :param pindx: n array, pixel scale of prf array
        :param pindy: n array, pixel scale of prf array
        """

        self.prf = prf
        self.pindx = pindx
        self.pindy = pindy

    def set_tile(self, moc):
        """ Update prior with new MOC and update appropriate variables
        :param moc: Multi-order Coverage map from pymoc
        """
        self.moc = self.moc.intersection(moc)
        self.cut_down_prior()

    def cut_down_map(self):
        """Cuts down prior class variables associated with the map data to the MOC assigned to the prior class: self.moc
        """
        wcs_temp = wcs.WCS(self.imhdu)
        ra, dec = wcs_temp.wcs_pix2world(self.sx_pix, self.sy_pix, 0)
        ind_map = np.array(moc_routines.check_in_moc(ra, dec, self.moc))
        # now cut down and flatten maps (default is to use all pixels, running segment will change the values below to pixels within segment)
        self.sx_pix = self.sx_pix[ind_map]
        self.sy_pix = self.sy_pix[ind_map]
        self.snim = self.snim[ind_map]
        self.sim = self.sim[ind_map]
        self.snpix = sum(ind_map)

    def cut_down_cat(self):
        """Cuts down prior class variables associated with the catalogue data to the MOC assigned to the prior class: self.moc
        """
        sgood = np.array(moc_routines.check_in_moc(self.sra, self.sdec, self.moc))

        self.sx = self.sx[sgood]
        self.sy = self.sy[sgood]
        self.sra = self.sra[sgood]
        self.sdec = self.sdec[sgood]
        self.nsrc = sum(sgood)
        self.ID = self.ID[sgood]
        if hasattr(self, 'nstack'):
            self.stack = self.stack[sgood]
            self.nstack = sum(self.stack)
        if hasattr(self, 'prior_flux_upper'):
            self.prior_flux_upper = self.prior_flux_upper[sgood]
        if hasattr(self, 'prior_flux_lower'):
            self.prior_flux_lower = self.prior_flux_lower[sgood]
        if hasattr(self, 'prior_flux_mu'):
            self.prior_flux_mu = self.prior_flux_mu[sgood]
        if hasattr(self, 'prior_flux_sigma'):
            self.prior_flux_sigma = self.prior_flux_sigma[sgood]
        if hasattr(self,'z_median'):
            self.z_median=self.z_median[sgood]
            self.z_sig=self.z_sig[sgood]

    def cut_down_prior(self):

        """
        Cuts down prior class variables to the MOC assigned to the prior class
        """
        self.cut_down_map()
        self.cut_down_cat()

    def prior_bkg(self, mu, sigma):
        r"""Add background prior. Assumes :math:`B \sim \mathcal{N}(\mu,\sigma^2)`

        :param mu: mean
        :param sigma: standard deviation
        """

        self.bkg = (mu, sigma)

    def upper_lim_map(self):
        """
        Update flux upper limit to abs(bkg)+2*sigma_bkg+max(D)
        where max(D) is maximum value of pixels the source contributes to.

        Vectorised using np.maximum.at, which is O(amat_col).
        """

        self.prior_flux_upper = np.full(self.nsrc, -np.inf)

        bkg_term = np.abs(self.bkg[0]) + 2 * self.bkg[1]

        values = self.sim[self.amat_row]

        # grouped max
        np.maximum.at(self.prior_flux_upper, self.amat_col, values)

        # remove -inf for empty sources
        empty = ~np.isfinite(self.prior_flux_upper)
        self.prior_flux_upper[empty] = 1000.0

        self.prior_flux_upper[~empty] += bkg_term

    def get_pointing_matrix(self):
        """
        Detects whether GPU is present, and utilises it if so.
        Otherwise uses CPU.
        """

        # Im not sure gpu is faster if tile is relatively small.
        # Maybe should check order and take into accoutn, but need testing.

        # Either way if tile is small doubt itll make much difference
        
        if jax.default_backend() == "gpu":
            print("GPU detected, running with GPU.")
            self.get_pointing_matrix_GPU()
        else:
            print("GPU not detected, running with CPU.")
            self.get_pointing_matrix_CPU()

    def get_pointing_matrix_CPU(self):
        """
        Calculate pointing matrix.

        O(nsrc x snpix)? Since both scale equally with tile size O(n^2)?
        Either way RGI is far far faster than the triangulation previously done.

        This could be parallelised since almost always will have multiple cores.
        """
        from scipy.interpolate import RegularGridInterpolator

        paxis1, paxis2 = self.prf.shape

        # List + append rather than arrays + np.append(), much faster
        # Even if the interpolation is easily the most time consuming part
        amat_row = []
        amat_col = []
        amat_data = []

        # ------Deal with PRF array----------
        centre1 = np.rint((paxis1 - 1.) / 2).astype(int)
        centre2 = np.rint((paxis2 - 1.) / 2).astype(int)

        # create pointing array

        # TQDM does not work very well with SLURM output, might have to rethink
        # for s in tqdm(range(self.nsrc), desc = "Calculating pointing matrix", miniters = self.nsrc//20):
        for s in range(self.nsrc):

            # diff from centre of beam for each pixel in x and y
            dx = -np.rint(self.sx[s]).astype(int) + self.pindx[centre1] + self.sx_pix
            dy = -np.rint(self.sy[s]).astype(int) + self.pindy[centre2] + self.sy_pix

            # # diff from each pixel in prf
            # pindx = self.pindx + self.sx[s] - np.rint(self.sx[s]).astype(int)
            # pindy = self.pindy + self.sy[s] - np.rint(self.sy[s]).astype(int)
            # ipx2, ipy2 = np.meshgrid(pindx, pindy)

            # Since SIDES places sources in centre of pixel, use this instead
            pindx = self.pindx
            pindy = self.pindy

            good = (dx >= 0) & (dx <= self.pindx[paxis1 - 1]) & (dy >= 0) & (dy <= self.pindy[paxis2 - 1])

            # Switch to RegularGridInterpolator rather than griddata which uses triangulation.
            # Much much faster, debatable which is 'better' if any.
            # BTW it expects (y,x)
            rgi = RegularGridInterpolator(
                (pindy, pindx),
                self.prf,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )

            atemp = rgi(np.column_stack([dy[good], dx[good]]))

            if atemp.size > 0:
                keep=atemp > np.max(atemp)/1.0E3
                amat_data.append(atemp[keep])
                amat_row.append(np.arange(0, self.snpix, dtype=int)[good][keep])
                amat_col.append(np.full(keep.sum(), s))

        self.amat_data = np.concatenate(amat_data)
        self.amat_row = np.concatenate(amat_row)
        self.amat_col = np.concatenate(amat_col)

    def get_pointing_matrix_GPU(self, batch_size=64, device='cuda'):
        """
        AI NEED TO CHECK THIS.
        I mean it gives the right results and its fast af?

        GPU-accelerated version of get_pointing_matrix using PyTorch.

        Produces amat_row, amat_col, amat_data identical to CPU version.
        """
        
        import torch

        # Move pixel coordinates to GPU
        sx_pix = torch.tensor(self.sx_pix, device=device, dtype=torch.float32)
        sy_pix = torch.tensor(self.sy_pix, device=device, dtype=torch.float32)
        snpix = sx_pix.shape[0]

        # PRF and its pixel scales
        prf = torch.tensor(self.prf, device=device, dtype=torch.float32)
        pindx = torch.tensor(self.pindx, device=device, dtype=torch.float32)
        pindy = torch.tensor(self.pindy, device=device, dtype=torch.float32)

        paxis1, paxis2 = prf.shape
        centre1 = int(round((paxis1 - 1) / 2))
        centre2 = int(round((paxis2 - 1) / 2))

        amat_row = []
        amat_col = []
        amat_data = []

        # Move source positions to GPU
        sx = torch.tensor(self.sx, device=device, dtype=torch.float32)
        sy = torch.tensor(self.sy, device=device, dtype=torch.float32)


        # TQDM does not work very well with SLURM output, might have to rethink
        # for i in tqdm(range(0, self.nsrc, batch_size), desc="GPU pointing matrix", miniters = self.nsrc//20):
        for i in range(0, self.nsrc, batch_size):
            batch_end = min(i + batch_size, self.nsrc)
            bs = batch_end - i

            # Source positions in batch
            sx_batch = sx[i:batch_end][:, None]  # bs x 1
            sy_batch = sy[i:batch_end][:, None]  # bs x 1

            # Compute dx/dy for all map pixels (broadcasted)
            dx = -torch.round(sx_batch).int() + pindx[centre1] + sx_pix[None, :]  # bs x snpix
            dy = -torch.round(sy_batch).int() + pindy[centre2] + sy_pix[None, :]  # bs x snpix

            # Mask pixels outside PRF bounds
            good = (dx >= 0) & (dx <= pindx[-1]) & (dy >= 0) & (dy <= pindy[-1])

            for b in range(bs):
                if good[b].any():
                    # Select valid pixels
                    dx_valid = dx[b][good[b]].float()
                    dy_valid = dy[b][good[b]].float()

                    # Normalize for grid_sample: [-1,1]
                    dx_norm = 2 * dx_valid / (paxis2 - 1) - 1
                    dy_norm = 2 * dy_valid / (paxis1 - 1) - 1

                    # Prepare coordinates: N x 1 x 1 x 2
                    coords = torch.stack([dx_norm, dy_norm], dim=-1)[None, :, None, :]

                    # PRF has shape 1 x 1 x H x W
                    prf_val = torch.nn.functional.grid_sample(
                        prf[None, None, :, :], coords, mode='bilinear', padding_mode='zeros', align_corners=True
                    )

                    prf_val = prf_val.flatten()

                    # Threshold tiny values as CPU version does
                    keep = prf_val > (prf_val.max() / 1e3)

                    amat_data.append(prf_val[keep].cpu().numpy())
                    amat_row.append(torch.arange(snpix, device=device)[good[b]][keep].cpu().numpy())
                    amat_col.append(np.full(keep.sum().item(), i+b, dtype=int))

        # Concatenate all batches
        self.amat_data = np.concatenate(amat_data)
        self.amat_row = np.concatenate(amat_row)
        self.amat_col = np.concatenate(amat_col)


class hier_prior(object):
    def __init__(self,phys_prior_table, emulator,emulator_file,hier_params):
        """Initiate SED prior class

        :param phys_prior_table and astropy table with prior parameters
        :param emulator emulator neural net structure
        :param emulator_path path to saved emulator file
        :param dictionary of hierarchical parameter"""


        from jax import random
        # load parameters saved in numpy file
        x = np.load(emulator_file, allow_pickle=True)
        # initiate passed emulator
        net_init, net_apply = emulator()
        #get input shape
        in_shape = (-1, x['arr_0'][0][0].shape[0],)
        key = random.PRNGKey(1)
        _, init_params = net_init(key, input_shape=in_shape)
        # check input emulator model and loaded parameters match
        for a, b in zip(x['arr_0'], init_params):
            if len(a) != len(b):
                raise ValueError('neural net emulator structure does not match parameter file')
                for c, d in zip(a, b):
                    if len(c) != len(d):
                        raise ValueError('neural net emulator structure does not match parameter file')

        self.emulator = {'net_init':net_init,'net_apply':net_apply,'params':x['arr_0'].tolist()}
        self.phys_prior_table=phys_prior_table
        self.hier_params=hier_params