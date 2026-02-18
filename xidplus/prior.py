import numpy as np

from astropy import wcs
from xidplus import moc_routines


# from scipy import interpolate
# from joblib import Parallel, delayed
# import jax
# import jax.numpy as jnp
# from jax.scipy.interpolate import RegularGridInterpolator
# from jax.experimental import sparse




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

    def _upper_lim_map(self):
        """Update flux upper limit to abs(bkg)+2*sigma_bkg+max(D)
         where max(D) is maximum value of pixels the source contributes to"""

        self.prior_flux_upper = np.full((self.nsrc), 1000.0)
        for i in range(0, self.nsrc):
            ind = self.amat_col == i
            if ind.sum() > 0:
                self.prior_flux_upper[i] = np.max(self.sim[self.amat_row[ind]]) + (np.abs(self.bkg[0]) + 2 * self.bkg[1])


    def upper_lim_map(self):
        """Update flux upper limit to abs(bkg)+2*sigma_bkg+max(D)
         where max(D) is maximum value of pixels the source contributes to"""

        self.prior_flux_upper = np.full(self.nsrc, 1000.0)

        # Calculate the background term once outside the loop
        bkg_term = np.abs(self.bkg[0]) + 2 * self.bkg[1]

        for i in range(self.nsrc):
            ind = self.amat_col == i
            if ind.any(): # Check .any() instead of boolean sum. .any() stops at first True
                self.prior_flux_upper[i] = np.max(self.sim[self.amat_row[ind]]) + bkg_term



    def _get_pointing_matrix(self, bkg=True):
        """Calculate pointing matrix. If bkg = True, bkg is fitted to all pixels. If False, bkg only fitted to where prior sources contribute
        """
        from scipy import interpolate
        paxis1, paxis2 = self.prf.shape

        amat_row = np.array([], dtype=int)
        amat_col = np.array([], dtype=int)
        amat_data = np.array([])

        # ------Deal with PRF array----------
        centre1 = np.rint((paxis1 - 1.) / 2).astype(int)
        centre2 = np.rint((paxis2 - 1.) / 2).astype(int)
        # create pointing array
        for s in range(0, self.nsrc):

            # diff from centre of beam for each pixel in x
            dx = -np.rint(self.sx[s]).astype(int) + self.pindx[centre1] + self.sx_pix
            # diff from centre of beam for each pixel in y
            dy = -np.rint(self.sy[s]).astype(int) + self.pindy[centre2] + self.sy_pix

            # # diff from each pixel in prf
            # Not using this right now, but isnt hte use of rint here a bit weird?
            # Like it works since its consistent with dx and dy but its weird

            # pindx = self.pindx + self.sx[s] - np.rint(self.sx[s]).astype(int)
            # pindy = self.pindy + self.sy[s] - np.rint(self.sy[s]).astype(int)
            # ipx2, ipy2 = np.meshgrid(pindx, pindy)

            # Since SIDES places sources in centre of pixel, use this instead
            ipx2, ipy2 = np.meshgrid(self.pindx, self.pindy)

            good = (dx >= 0) & (dx < self.pindx[paxis1 - 1]) & (dy >= 0) & (dy < self.pindy[paxis2 - 1])
            ngood = good.sum()
            bad = np.asarray(good) == False
            nbad = bad.sum()
            
            atemp = interpolate.griddata((ipx2.ravel(), ipy2.ravel()), self.prf.ravel(), (dx[good], dy[good]), method='linear')

            if atemp.size > 0:
                keep=atemp > np.max(atemp)/1.0E3
                amat_data = np.append(amat_data, atemp[keep])
                amat_row = np.append(amat_row,np.arange(0, self.snpix, dtype=int)[good][keep])  # what pixels the source contributes to
                amat_col = np.append(amat_col, np.full(keep.sum(), s))  # what source we are on


        self.amat_data = amat_data
        self.amat_row = amat_row
        self.amat_col = amat_col


    def get_pointing_matrix(self):
        """
        Calculate pointing matrix.
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


    # Other functions used. For e.g. parallelism, or running with gpu
    # Need to look into them though. Test + verify

    # They were also built on the old version with e.g. wrong pointing matrix for SIDES, so have to update those
    # def upper_lim_map_per_source(self, i):
    #     """Calculates the flux upper limit for an individual source."""

    #     ind = self.amat_col == i
    #     if ind.sum() > 0:
    #         # self.prior_flux_upper[i] = np.max(self.sim[self.amat_row[ind]]) + (np.abs(self.bkg[0]) + 2 * self.bkg[1])
    #         return i,  np.max(self.sim[self.amat_row[ind]]) + (np.abs(self.bkg[0]) + 2 * self.bkg[1])
        
    # def upper_lim_per_batch(self, batch_sources):
    #     """Calculates the upper limit of a batch of sources."""
    #     indices = np.array([])
    #     values = np.array([])
    #     for s in batch_sources:
    #         index, value = self.upper_lim_map_per_source(s)
    #         indices = np.append(indices, index)
    #         values = np.append(values, value)
    #     return indices, values

    # def upper_lim_map_parallel(self):
    #     """Updates the flux upper limit of all sources in parallel. See upper_lim_map."""
    #     self.prior_flux_upper = np.full((self.nsrc), 1000.0)

    #     # For low numbers of sources, it's not worth the overheads of parallelising 
    #     batches = list(self.batcher(range(self.nsrc), 10000))

    #     results = Parallel(n_jobs=-1)(delayed(self.upper_lim_per_batch)(batch) for batch in batches)
        

    #     indices, values = zip(*results)

    #     indices = np.concatenate(indices).ravel().astype(np.int64)
    #     values = np.concatenate(values).ravel().astype(np.float64)

    #     for index, value in zip(indices, values):
    #         self.prior_flux_upper[index] = value

    # # ---------------
    
    # # --- Build interpolator once outside ---
    # @staticmethod
    # def build_prf_interpolator(prf):
    #     paxis1, paxis2 = prf.shape
    #     px = jnp.arange(paxis1)
    #     py = jnp.arange(paxis2)
    #     interp = RegularGridInterpolator(
    #         (px, py),
    #         jnp.array(prf),
    #         method='linear',
    #         bounds_error=False,
    #         fill_value=0.0
    #     )
    #     return interp

    # # --- Top-level function to process a single source ---
    # @staticmethod
    # def process_single_source(interp, sx_val, sy_val, sx_pix, sy_pix, snpix, source_index):
    #     paxis1, paxis2 = interp.grid[0].shape[0], interp.grid[1].shape[0]
    #     dx = -jnp.rint(sx_val) + (paxis1-1)/2 + sx_pix
    #     dy = -jnp.rint(sy_val) + (paxis2-1)/2 + sy_pix

    #     coords = jnp.stack([dy, dx], axis=-1)
    #     atemp = interp(coords)  # out-of-bounds points are 0

    #     rows = jnp.arange(snpix)
    #     cols = jnp.full(snpix, source_index)
    #     data = atemp

    #     return rows, cols, data


    # def get_pointing_matrix_jax(self, bkg=True, batch_size=15):
    #     """
    #     Build pointing matrix fully on GPU without pre-allocating huge arrays.
    #     Handles 44k sources with large snpix.
    #     """
    #     interp = self.build_prf_interpolator(self.prf)
    #     nsrc = self.sx.shape[0]
    #     snpix = self.snpix

    #     all_rows, all_cols, all_data = [], [], []

    #     for i in range(0, nsrc, batch_size):
    #         if i % 300 == 0:
    #             print(i)
    #         end = min(i + batch_size, nsrc)
    #         batch_idx = jnp.arange(i, end)
    #         sx_batch = self.sx[i:end]
    #         sy_batch = self.sy[i:end]

    #         # Compute pointing matrix for this batch
    #         rows_b, cols_b, data_b = jax.vmap(
    #             lambda sx_val, sy_val, src_idx: self.process_single_source(
    #                 interp, sx_val, sy_val, self.sx_pix, self.sy_pix, snpix, src_idx
    #             )
    #         )(sx_batch, sy_batch, batch_idx)

    #         # Flatten batch
    #         rows_b = rows_b.ravel()
    #         cols_b = cols_b.ravel()
    #         data_b = data_b.ravel()

    #         # Keep only non-zero entries (on GPU)
    #         mask = data_b > 1e-3 * jnp.max(data_b)
    #         rows_b = rows_b[mask]
    #         cols_b = cols_b[mask]
    #         data_b = data_b[mask]

    #         # Append to lists (still on GPU)
    #         all_rows.append(rows_b)
    #         all_cols.append(cols_b)
    #         all_data.append(data_b)

    #     # Concatenate everything at the end (on GPU) then convert to CPU
    #     amat_row = jnp.concatenate(all_rows)
    #     amat_col = jnp.concatenate(all_cols)
    #     amat_data = jnp.concatenate(all_data)

    #     self.amat_row = np.array(amat_row)
    #     self.amat_col = np.array(amat_col)
    #     self.amat_data = np.array(amat_data)

    # def get_pointing_matrix_per_source(self, s):
    #     from scipy import interpolate

    #     """Calculate the pointing matrix for an individual source."""
    #     paxis1, paxis2 = self.prf.shape

    #     centre_x = self.pindx[np.rint((paxis1 - 1.) / 2).astype(np.long)]
    #     centre_y = self.pindy[np.rint((paxis2 - 1.) / 2).astype(np.long)]

    #     # diff from centre of beam for each pixel in x
    #     dx = -np.rint(self.sx[s]).astype(np.long) + centre_x + self.sx_pix

    #     # diff from centre of beam for each pixel in y
    #     dy = -np.rint(self.sy[s]).astype(np.long) + centre_y + self.sy_pix

    #     # diff from each pixel in prf
    #     # pindx = self.pindx + self.sx[s] - np.rint(self.sx[s]).astype(np.long)
    #     # pindy = self.pindy + self.sy[s] - np.rint(self.sy[s]).astype(np.long)
    #     pindx = self.pindx # for SIDES, use this, for real data use above
    #     pindy = self.pindy

    #     good = (dx >= 0) & (dx < self.pindx[paxis1 - 1]) & (dy >= 0) & (dy < self.pindy[paxis2 - 1])
    #     # ngood = good.sum()

    #     if good.sum() == 0: # This should never happen? but if it does it saves the interpolation.
    #         return np.array([], dtype=float), np.array([], dtype=int), np.array([], dtype=int)

    #     # bad = np.asarray(good) == False
    #     # nbad = bad.sum()

    #     ipx2, ipy2 = np.meshgrid(pindx, pindy)

    #     # method should be linear or cubic, nearest causes some issues 
    #     atemp = interpolate.griddata((ipx2.ravel(), ipy2.ravel()), self.prf.ravel(), (dx[good], dy[good]),
    #                                         method='linear')
        
    #     if atemp.size == 0:
    #         return np.array([], dtype=float), np.array([], dtype=int), np.array([], dtype=int)
        

    #     keep=atemp > np.max(atemp)/1.0E3
    #     amat_data = atemp[keep]
    #     amat_row = np.arange(0, self.snpix, dtype=int)[good][keep]  # what pixels the source contributes to
    #     amat_col = np.full(keep.sum(), s)  # what source we are on
 
    #     return amat_data, amat_row, amat_col

    # def get_pointing_matrix_per_batch(self, batch_sources):
    #     """Calculate pointing matrix for a batch of sources."""

    #     data = np.array([])
    #     row = np.array([])
    #     col = np.array([])
    #     for s in batch_sources:
    #         d, r, c = self.get_pointing_matrix_per_source(s)
    #         data = np.append(data, d)
    #         row = np.append(row, r)
    #         col = np.append(col, c)
    #     return data, row, col

    # def batcher(self, num_sources, batch_size):
    #     """Split sources into batches"""
    #     # could this not be [range(i:i+n) for i in range(0, )]. and remove this function as a whole
    #     for i in range(0, len(num_sources), batch_size):
    #         yield num_sources[i:i+batch_size]
    #     return

    # def get_pointing_matrix_parallel(self):
    #     """Calculate pointing matrix for all sources in parallel. See get_pointing_matrix."""

    #     # If one interpolation takes ~ 0.5 s, 120 sources per batch should take ~1 min per batch.
    #     batches = list(self.batcher(range(self.nsrc), 120))

    #     results = Parallel(n_jobs=-1)(delayed(self.get_pointing_matrix_per_batch)(batch) for batch in batches)

    #     amat_data, amat_row, amat_col = zip(*results)
        
    #     self.amat_data = np.concatenate(amat_data).ravel().astype(np.int64)
    #     self.amat_row = np.concatenate(amat_row).ravel().astype(np.int64)
    #     self.amat_col = np.concatenate(amat_col).ravel().astype(np.int64)

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