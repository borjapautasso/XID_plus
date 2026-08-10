import numpy as np
from astropy import wcs
from xidplus import moc_routines
import jax
from astropy.coordinates import SkyCoord
import astropy.units as u

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

    def stepwise_prima_prior_cat(self, ra, dec, prior_cat_file, flux_mu=None, flux_sigma=None, ID=None, moc=None,z_median=None,z_sig=None, prior_mstar=None, fwhm=None, sigma_sens=None):
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
        self.fwhm = fwhm
        self.sigma_sens = sigma_sens
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

    def cut_down_map(self, expand_fwhm):
        """Cuts down prior class variables associated with the map data to the MOC assigned to the prior class: self.moc
        """
        wcs_temp = wcs.WCS(self.imhdu)
        ra, dec = wcs_temp.wcs_pix2world(self.sx_pix, self.sy_pix, 0)
        ind_map = np.array(moc_routines.check_in_moc(ra, dec, self.moc))


        if expand_fwhm:
            print(f"Expanding map by {expand_fwhm}FWHM")
            ### Expand map by HWHM 
            all_pixels = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            in_moc_pixels = SkyCoord(ra=ra[ind_map]*u.deg, dec=dec[ind_map]*u.deg)
            
            # Find all pixels within radius of ANY good pixel
            fwhm = self.fwhm*u.arcsec

            idx_all, idx_in_moc, ang_sep, _ = in_moc_pixels.search_around_sky(all_pixels, fwhm * expand_fwhm)

            near_mask = np.zeros(len(self.sx_pix), dtype=bool)
            near_mask[idx_all] = True

            ind_map = ind_map | near_mask

        self.sx_pix = self.sx_pix[ind_map]
        self.sy_pix = self.sy_pix[ind_map]
        self.snim = self.snim[ind_map]
        self.sim = self.sim[ind_map]
        self.snpix = sum(ind_map)

    def cut_down_cat(self, expand_fwhm):
        """Cuts down prior class variables associated with the catalogue data to the MOC assigned to the prior class: self.moc
        """
        sgood = np.array(moc_routines.check_in_moc(self.sra, self.sdec, self.moc))

        if expand_fwhm:
            print(f"Expanding cat by {expand_fwhm}FWHM")
            ### Expand catalogue by HWHM (a further HWHM than the map already woudl've been) 
            all_sources = SkyCoord(ra=self.sra*u.deg, dec=self.sdec*u.deg)

            wcs_temp = wcs.WCS(self.imhdu)
            map_ra, map_dec = wcs_temp.wcs_pix2world(self.sx_pix, self.sy_pix, 0)
            map_pixels = SkyCoord(ra=map_ra*u.deg, dec=map_dec*u.deg)

            fwhm = self.fwhm*u.arcsec

            idx_all, idx_map, ang_sep, _ = map_pixels.search_around_sky(all_sources, fwhm * expand_fwhm)

            near_mask = np.zeros(len(self.sra), dtype=bool)
            near_mask[idx_all] = True

            sgood = sgood | near_mask

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

    def cut_down_prior(self, expand_fwhm = False):

        """
        Cuts down prior class variables to the MOC assigned to the prior class

        Args:
            expand_fwhm (bool):
                Whether to expand the map by a FWHM, and the cat by a further FWHM.
        """
        self.cut_down_map(expand_fwhm)
        self.cut_down_cat(expand_fwhm)

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

    def get_pointing_matrix(self, batch_size=1024, pad=2):
        """
        Calculate pointing matrix using JAX, restricted to a small window
        around each source (sized to the PRF footprint) instead of every
        pixel in the tile. O(nsrc x window) instead of O(nsrc x snpix).
        """
        import jax.numpy as jnp
        from jax import lax
        from jax.scipy.ndimage import map_coordinates

        paxis1, paxis2 = self.prf.shape
        centre1 = int(round((paxis1 - 1) / 2))
        centre2 = int(round((paxis2 - 1) / 2))

        pindx_np = np.asarray(self.pindx, dtype=np.float64)
        pindy_np = np.asarray(self.pindy, dtype=np.float64)
        x0, dxs = float(pindx_np[0]), float(pindx_np[1] - pindx_np[0])
        y0, dys = float(pindy_np[0]), float(pindy_np[1] - pindy_np[0])
        x_max, y_max = float(pindx_np[-1]), float(pindy_np[-1])

        # fixed window size (image pixels) that comfortably covers the PRF footprint
        nwx = int(np.ceil(x_max - x0)) + 1 + 2 * pad
        nwy = int(np.ceil(y_max - y0)) + 1 + 2 * pad

        # --- build a padded lookup grid: image pixel (y, x) -> row index into
        # sx_pix/sy_pix/sim, or -1 if that pixel isn't in the cut-down tile ---
        sx_pix_int = np.rint(self.sx_pix).astype(np.int64)
        sy_pix_int = np.rint(self.sy_pix).astype(np.int64)
        x_min, x_max_img = sx_pix_int.min(), sx_pix_int.max()
        y_min, y_max_img = sy_pix_int.min(), sy_pix_int.max()

        half_x = nwx // 2 + 1
        half_y = nwy // 2 + 1

        grid_w = (x_max_img - x_min) + 1 + 2 * half_x
        grid_h = (y_max_img - y_min) + 1 + 2 * half_y

        index_grid_np = np.full((grid_h, grid_w), -1, dtype=np.int32)
        gx = sx_pix_int - x_min + half_x
        gy = sy_pix_int - y_min + half_y
        index_grid_np[gy, gx] = np.arange(self.snpix, dtype=np.int32)

        index_grid = jnp.asarray(index_grid_np)
        prf = jnp.asarray(self.prf, dtype=jnp.float32)
        pindx = jnp.asarray(self.pindx, dtype=jnp.float32)
        pindy = jnp.asarray(self.pindy, dtype=jnp.float32)
        sx = jnp.asarray(self.sx, dtype=jnp.float32)
        sy = jnp.asarray(self.sy, dtype=jnp.float32)

        def single_source(sx_s, sy_s):
            x_img_start = jnp.rint(sx_s).astype(jnp.int32) - nwx // 2
            y_img_start = jnp.rint(sy_s).astype(jnp.int32) - nwy // 2

            grid_x_start = x_img_start - x_min + half_x
            grid_y_start = y_img_start - y_min + half_y

            # NB: dynamic_slice clips the start so the slice stays in bounds;
            # the padding (half_x/half_y) is sized so that never happens for
            # sources actually inside the cut-down tile.
            window = lax.dynamic_slice(index_grid, (grid_y_start, grid_x_start), (nwy, nwx))

            local_x = x_img_start + jnp.arange(nwx)
            local_y = y_img_start + jnp.arange(nwy)
            xx, yy = jnp.meshgrid(local_x, local_y)

            dx = -jnp.rint(sx_s) + pindx[centre1] + xx
            dy = -jnp.rint(sy_s) + pindy[centre2] + yy

            good = (dx >= 0) & (dx <= x_max) & (dy >= 0) & (dy <= y_max) & (window >= 0)

            idx_x = (dx - x0) / dxs
            idx_y = (dy - y0) / dys
            vals = map_coordinates(prf, [idx_y, idx_x], order=1, mode='constant', cval=0.0)

            return vals, good, window

        batched_fn = jax.jit(jax.vmap(single_source))

        amat_row = []
        amat_col = []
        amat_data = []

        for i in range(0, self.nsrc, batch_size):
            j = min(i + batch_size, self.nsrc)
            vals, good, window = batched_fn(sx[i:j], sy[i:j])
            vals = np.asarray(vals).reshape(j - i, -1)
            good = np.asarray(good).reshape(j - i, -1)
            window = np.asarray(window).reshape(j - i, -1)

            vals_masked = np.where(good, vals, -np.inf)
            row_max = vals_masked.max(axis=1)
            thresh = row_max / 1.0e3
            keep = good & (vals > thresh[:, None])

            src_idx, win_idx = np.nonzero(keep)
            amat_data.append(vals[src_idx, win_idx])
            amat_row.append(window[src_idx, win_idx])
            amat_col.append(src_idx + i)

        self.amat_data = np.concatenate(amat_data)
        self.amat_row = np.concatenate(amat_row).astype(np.int64)
        self.amat_col = np.concatenate(amat_col).astype(np.int64)



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