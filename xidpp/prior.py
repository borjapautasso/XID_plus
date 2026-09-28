from astropy.coordinates import SkyCoord, search_around_sky
from astropy.units import arcsec
from astropy.wcs import WCS
from astropy.io import fits
import jax
from scipy.ndimage import maximum_filter
import warnings
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from xidpp import moc_routines
from numpy.typing import ArrayLike, NDArray
from pymoc import MOC
"""
Only the prior object.

Methods
init
create catalogue
prf
cuts (map, cat, both)
pointing

Tracking changes from the main branch (i.e. not the HELP version).

General name changes to make it more readable, alongisde adding docs and comments.
'Bad' pixels are now masked properly.
Removed set_moc and set_tile: Redundant, never used, and very simple to replace if/when needed.
Merged catalogue methods.

Three verbs for the methods.
Set: User gives something, and it sets the variables/objects.
        - set_catalogue
        - set_prf
        - set_bkg

        (since a map is always the starting point, there isn't strictly a set_map, although it's basically the innit)

Cut: Trim catalogue/map/both for a given MOC.
        - cut_map
        - cut_catalogue
        - cut_prior

Compute: Compute something from the current state of the object. (gen if I want three letters but probs not:/)
        - compute_pointing_matrix
        - compute_upper_lims

        
Potentially simplify the two main functions by adding a few private methods?

i.e. the innit is these tasks:
- map (including the masking and flattening)
- moc (should probably stardadise it throughout, i.e. reintroduce set_moc?)
- fwhm+survey_sens stays as is.

And the catalogue.
- Set coords
- Set properties (loops over all the optionals, lots of asserts too hidden away ^^)
- Set moc (as above :/)
"""

class Prior():
    """
    Contains the image and prior source information used by XID+.

    The prior is initialised with the image and noise maps and corresponding WCS information.
    Source catalogue information is added separately via :meth:`set_catalogue`.

    Arguments
    ---------
    
    """

    def __init__(
        self,
        image: NDArray,
        noise: NDArray,
        header: fits.Header,
        moc: MOC|None = None,
        fwhm: float|None = None,
        survey_sens: float|None = None,
        ) -> None:
        """
        Initialise Prior object.

        Sets up the ``Prior`` object with the map data. Takes both an image (i.e. flux) map, and a
        noise/error map.        
        Any pixel with a non-finite value (i.e. nan or inf) in either map, or with exactly zero
        noise is treated as unusable, and are masked out.
        
        Parameters
        ----------
        image : array_like
            Image map.
        noise : array_like
            Noise map.
        header : fits.Header
            FITS Header object associated with image, used to make WCS object.
        moc : pymoc.MOC, optional
            MOC object which defines the area being kept.
        fwhm : float, optional
            FWHM of the map beam, in arcseconds.
        survey_sens : float, optional
            1-sigma survey sensitivity, in map intensity units. 
        """

        self.header = header

        wcs = WCS(self.header)

        # There seems to be some hatred towards using the wcs to get the map dimensions.
        # Although by construction (for a sensible input map), our wcs.pixel_shape is never None.
        x_pix, y_pix = np.meshgrid(np.arange(wcs.pixel_shape[0]), np.arange(wcs.pixel_shape[1]))

        self.map_x = x_pix.flatten()
        self.map_y = y_pix.flatten()

        self.map_flux = image.flatten()
        self.map_noise = noise.flatten()

        bad_mask = ~np.isfinite(self.map_flux) | ~np.isfinite(self.map_noise) | (self.map_noise == 0)

        if bad_mask.any():
            self.map_x = self.map_x[~bad_mask]
            self.map_y = self.map_y[~bad_mask]
            self.map_flux = self.map_flux[~bad_mask]
            self.map_noise = self.map_noise[~bad_mask]

        self.npix = self.map_flux.size

        if fwhm is not None:
            self.fwhm = fwhm

        if survey_sens is not None:
            self.survey_sens = survey_sens

        if moc is not None:
            self.moc = moc
            self.cut_map()

    def set_catalogue(
            self,
            ra: NDArray,
            dec: NDArray,
            cat_name: str|None = None,
            src_id: NDArray|None = None,
            flux_lower: NDArray|None = None,
            flux_upper: NDArray|None = None,
            flux_mu: NDArray|None = None,
            flux_sigma : NDArray|None= None,
            z_mu: NDArray|None = None,
            z_sigma: NDArray|None = None,
            moc: MOC|None = None
        ):
        """
        Create prior source catalogue, containing the properties of the sources that XID+ will fit.

        Parameters
        ----------

        
        Args:
            ra:
                Right ascension (ICRF) of sources, in degrees.
            dec:
                Declination (ICRF) of sources, in degrees.
            cat_name:
                Name of the input catalogue.
            ID:
                Unique ID of each source.
            flux_lower:
                Lower flux limit of each source. Defaults to 0 mJy.
            flux_upper:
                Upper flux limit of each source. Defaults to 1000 mJy.
            flux_mu:
                Mean flux of each source. Used when modelling with non-uniform flux priors.
            flux_sigma:
                Standard deviation on the flux of each source. Used when modelling with non-uniform
                flux priors.
            z_mu:
                Median redshift of each source. Currently unused? SED?
            z_sigma:
                Standard deviation on the redshift of each source.
            moc:
                Pymoc MOC object covering the catalogue.
        """
        # NOTE
        # Potential thigns to assert?
        # len ra = len dec
        # if ID, that they are unique, and that they have same length as ra
        # all others (bar cat_name and moc), that they are same length.
        # if either mu is given, respective sigma must be too.

        wcs = WCS(self.header)

        src_x, src_y = wcs.wcs_world2pix(ra, dec, 0)

        if moc is None:
            cat_moc = moc_routines.create_moc_from_cat(ra, dec)
        else:
            cat_moc = moc

        self.src_x = src_x
        self.src_y = src_y
        self.src_ra = ra
        self.src_dec = dec
        self.nsrc = self.src_ra.size
        self.cat_name = cat_name

        if src_id is None:
            src_id = np.arange(1, self.src_ra.size + 1)
        self.src_id = src_id

        if flux_lower is None:
            flux_lower = np.full((self.src_ra.size), 0.)

        if flux_upper is None:
            flux_upper = np.full((self.src_ra.size), 1000.)

        self.flux_lower = flux_lower
        self.flux_upper = flux_upper

        if flux_mu is not None:
            self.flux_mu = flux_mu
            self.flux_sigma = flux_sigma

        if z_mu is not None:
            self.z_mu = z_mu
            self.z_sigma = z_sigma

        try:
            self.moc = self.moc.intersection(cat_moc)
        except AttributeError:
            self.moc = cat_moc

        self.cut_prior()

    def set_prf(self, prf, prf_x, prf_y):
        """
        Sets the point response function (PRF) array and coordinate axes.

        The peak of the PRF must be in its central pixel, and it must be uniformly positioned (i.e. constant spacial distance between pixels)
        ``prf`` may be given as an oversampled PSF (i.e. with a different pixel scale than the map), though ``prf_x``/``prf_y`` must account for this.

        If given at map pixel scale, an n x n ``prf`` would have both ``prf_x`` and ``prf_y`` equivalent to ``np.arange(0, n, 1)``.
        More generally: ``np.arange(0, n, (prf_pix_scale/map_pix_scale))``.
        """
        # NOTE: Pointing matrix assumes the PSF array is a uniform square in map pixel units
        # i.e. np.diff(prf_{x,y}) == -1 everywhere, and prf.shape[0] == prf.shape[1]
        # Also assumes the peak of the PSF is in it's centre.

        # At some point it'll be worth fixing the PSF when that is not the case: padding the axes
        # or changing those assumptions.
        # For now just raise not implemented.

        # Equal axes length.
        if prf.shape[0] != prf.shape[1]:
            raise NotImplementedError("PSF must have equal axis lengths.")
        
        # Check array is odd.
        if prf.shape[0] % 2 == 0:
            raise NotImplementedError("PSF array length but have odd length (to have a central pixel)")

        # Peak is in centre
        if np.nanmax(prf) != prf[prf.shape[0]//2, prf.shape[0]//2]:
            raise NotImplementedError("PSF peak must be in its centre.")
        
        # In map pixels
        if np.any(np.abs(np.diff(prf_x)) != 1) or np.any(np.abs(np.diff(prf_y)) != 1):
            raise NotImplementedError("PSF array must be in map pixels.")


        # NOTE: it would be possible to create the x and y axes ourselves, and just ask for the prf and the prf_pixscale.
        self.prf = prf
        self.prf_x = prf_x
        self.prf_y = prf_y


        # TODO: Pointing matrix assumes the prf is stored in map pixel coordinates.
        # We need to enforce this logic here, downsampling if needed.
        # If the PRF is downsampled compared to the map, probably just raise a chunky error cos wtf

    def set_bkg(self, mu, sigma):
        """
        Set a Gaussian prior on the background (B).
        
        XID+ assumes the background is normally distributed, i.e. ``B ~ N(mu, sigma^2)``.

        Args:
            mu:
                Mean
            sigma:
                Standard deviation
        
        """
        # NOTE:
        # For consistency either this should be split into lower and upper equivalent (i.e. _mu, _sigma), or the flux bounds should also be set as a tuple?
        # self.bkg = (mu, sigma)

        self.bkg_mu = mu
        self.bkg_sigma = sigma

    def cut_map(self, expand_fwhm = False):
        """
        Cut down prior map to within MOC.

        If ``expand_fwhm`` is True, adds a layer of padding of 1 FWHM around the MOC.
        """
        wcs = WCS(self.header)

        # Get mask of pixels with centres within the MOC
        ra, dec = wcs.wcs_pix2world(self.map_x, self.map_y, 0)
        keep_mask = np.array(moc_routines.check_in_moc(ra, dec, self.moc))

        if expand_fwhm:
            all_pixel_coords = SkyCoord(ra = ra, dec = dec, unit = "deg")
            in_moc_pixel_coords = SkyCoord(ra = ra[keep_mask], dec = dec[keep_mask], unit = "deg")

            # NOTE currently just errors if fwhm undefined 
            # First output of function is those indeces of the all_pixels_coords
            # that have an in_moc_pixel_coord within a FWHM
            idx_all, *_ = search_around_sky(all_pixel_coords, in_moc_pixel_coords, self.fwhm*arcsec)

            fwhm_mask = np.full(self.map_x.size, False)
            fwhm_mask[idx_all] = True

            # Union of those pixels in the moc, and those within a fwhm of an in_moc pix
            keep_mask = keep_mask | fwhm_mask
        
        self.map_x = self.map_x[keep_mask]
        self.map_y = self.map_y[keep_mask]
        self.map_flux = self.map_flux[keep_mask]
        self.map_noise = self.map_noise[keep_mask]        
        self.npix = self.map_flux.size

    def cut_catalogue(self, expand_fwhm = False):
        """
        Cut down prior catalogue to within MOC.
        
        If ``expand_fwhm`` is True, adds a layer of padding of 1 FWHM around every map pixel.
        """
        # Get mask of sources within the MOC
        keep_mask = np.array(moc_routines.check_in_moc(self.src_ra, self.src_dec, self.moc))

        if expand_fwhm:
            all_source_coords = SkyCoord(self.src_ra, self.src_dec, unit = "deg")

            wcs = WCS(self.header)
            map_ra, map_dec = wcs.wcs_pix2world(self.map_x, self.map_y, 0)
            map_pixel_coords = SkyCoord(map_ra, map_dec, unit = "deg")

            idx_all, *_ = search_around_sky(all_source_coords, map_pixel_coords, self.fwhm*arcsec)

            fwhm_mask = np.full(self.src_ra.size, False)
            fwhm_mask[idx_all] = True

            keep_mask = keep_mask | fwhm_mask

        self.src_x = self.src_x[keep_mask]
        self.src_y = self.src_y[keep_mask]
        self.src_ra = self.src_ra[keep_mask]
        self.src_dec = self.src_dec[keep_mask]

        self.nsrc = self.src_ra.size

        self.src_id = self.src_id[keep_mask]

        self.flux_lower = self.flux_lower[keep_mask]
        self.flux_upper = self.flux_upper[keep_mask]

        try:
            self.flux_mu = self.flux_mu[keep_mask]
            self.flux_sigma = self.flux_sigma[keep_mask]
        except AttributeError:
            pass

        try:
            self.z_mu = self.z_mu[keep_mask]
            self.z_sigma = self.z_sigma[keep_mask]
        except AttributeError:
            pass

    def cut_prior(self, expand_fwhm = False):
        """
        Cuts down prior map and catalogue to the prior MOC.

        If ``expand_fwhm`` is True, adds a layer of padding of 1 FWHM to the map, and an additional
        one (i.e. a FWHM from the padded map) to the catalogue.
        """
        self.cut_map(expand_fwhm)
        self.cut_catalogue(expand_fwhm)

    def _build_pix_grid(self, values, fill, pad = 0):
        """
        Build pixel grid, where grid[y,x] = fill.

        Used both in :meth:`compute_upper_lims`, where it recreates the flux map, and in
        :meth:`compute_pointing_matrix`, where it links a pixel to its pointing matrix row.
        """
        x_origin = np.min(self.map_x) - pad
        y_origin = np.min(self.map_y) - pad

        shape = (np.max(self.map_y) - y_origin + pad + 1,
                 np.max(self.map_x) - x_origin + pad + 1)

        grid = np.full(shape, fill)

        grid[self.map_y - y_origin, self.map_x - x_origin] = values

        return grid, x_origin, y_origin

    def compute_upper_lims(self, window_size = 5):
        """
        Update flux upper limits (``self.flux_upper``) of each source to an estimate given the map
        (with some margin via the background parameter).

        For sources with a non-missing map pixel at their position, sets the upper limit to
        ``max(D) + |bkg.mu| + 2 * bkg.sigma``, where D is the set of map pixel values in a 5 x 5
        window centred on the source pixel. Sources whose central pixel is missing (e.g. masked, or
        outside the tile) keep their exisiting ``flux_upper``.

        In the original method, ``D`` was every pixel value the source contributed to, calculated
        via the pointing matrix.It also assumed a source's own pixel could never be missing from
        the map, which is no longer the case.
        """
        # bkg_term = np.abs(self.bkg[0]) + 2 * self.bkg[1]
        bkg_term = np.abs(self.bkg_mu) + 2 * self.bkg_sigma

        # Build image grid, where value_grid[y,x] is the value of that pixel
        value_grid, x_origin, y_origin = self._build_pix_grid(self.map_flux, fill = -np.inf)
        grid_height, grid_width = value_grid.shape

        # Get source pixel position (integer, not sub-pix), relative to grid origin
        src_x = (np.rint(self.src_x) - x_origin).astype(int)
        src_y = (np.rint(self.src_y) - y_origin).astype(int)

        # Check that the source's pixel exists within the bounds of the map, and that it is not missing 
        within_grid = (src_x >= 0) & (src_x < grid_width) & (src_y >= 0) & (src_y < grid_height)

        has_centre_pix = np.full(self.nsrc, False)
        has_centre_pix[within_grid] = np.isfinite(value_grid[src_y[within_grid], src_x[within_grid]])

        # For every pixel, find the maximum in a 5x5 window centred on it
        max_grid = maximum_filter(value_grid, size = window_size)

        max_vals = np.full(self.nsrc, -np.inf)
        max_vals[within_grid] = max_grid[src_y[within_grid], src_x[within_grid]]

        # For all sources who have a flux value at their pixel, update their upper flux limits
        self.flux_upper[has_centre_pix] = max_vals[has_centre_pix] + bkg_term
        
    def compute_pointing_matrix(self, pad = 2, chunk = 2000, subpix = True):
        """
        Compute sparse pointing matrix.

        For each source, samples the PRF over a fixed window of the map around it, keeping pixels
        where the beam exceeds 1e-3 (``keep_thresh``) times its peak.
        Results are stored in ``self.amat_data``/``self.amat_row``/``self.amat_col``, where source
        ``amat_col[i]`` with flux F contributes ``amat_data[i] * F`` to map pixel ``amat_row[i]``.
        ``amat_col`` and ``amat_row`` index based on the catalogue and map pixel lists respectively.

        The interpolation is done manually for improved performance in both compute time and memory
        efficiency due to the separation of each independent axis.
        The latest publicly available method used ``scipy.interpolate``'s ``griddata`` with
        ``method = nearest``, i.e. did not interpolate.
        Private build used ``RegularGridInterpolator`` with ``method = 'linear'`` which proved far
        faster than ``griddata``, but is still slower than the manual method as per above.

        The PRF is trimmed to pixels above 1e-4 (``prf_thresh``) of its peak before interpolation.
        This reduces the number of pixels to look into, but also allows sufficient padding for
        the pixel trim via ``keep_thresh`` later.

        ``subpix`` takes into account the subpixel position of sources, rather than assuming it is
        in the centre of its pixel. This allows usage with both simple (i.e. SIDES) and more
        sophistated/real map makers.

        ``pad`` allows for sources slightly outside the kept map grid (i.e. those introduced from
        the ``expand_fwhm`` padding) to be computed correctly.

        Changes:
         - Trimmed PRF to above ``prf_thresh`` of the peak.
         - Manual bilinear interpolation.
         - Consistently keep only those pixels above ``keep_thresh`` times the peak of the PRF.
           Original method trimmed based on the maximum pixel within the map. For sources outside
           the tile (i.e. with no corresponding map pixel), the brightest pixel in the map isn't
           the beam peak, resulting in the cut keeping more of the wings.
        
        Args:
            pad:
                How many pixels on either side of the window to be added. Provides margin on the
                PRF cut, alongside the subpixel calculation.
            chunk:
                How many sources are processed per batch. For sufficiently large values (>1,000)
                runtime is independent of it. Too low or too high values may cause performance loss
                from Python overheads and memory issues respectively.
            subpix:
                Whether to treat a source's position to be defined at subpixel accuracy.
                If false, sources are treated as being in the exact centre of their pixel.
        """
        # NOTE 
        # More than likely these two should just be constants but idk
        # Value at which to trim the PRF. Needs to be larger than `keep_thresh`
        prf_thresh = 1e-4
        # A source is counted as contributing towards a pixel if its prf at that pixel is > 1e-3 its peak.
        keep_thresh = 1e-3

        # NOTE THIS ASSUMES THE PIXEL SCALE OF THE PSF IS THE SAME AS THE PIXEL SCALE OF THE MAP.
        # SHOULD MAKE A THING IN SET_PRF OF THAT LOGIC, AND MANUALLY CONVERT IF NOT?
        prf = self.prf

        # Radii (in map pixels) at which the beam drops below each threshold.
        # Calculates the 'Chebyshev' distance, to give enough area for the window later on.
        peak_iy, peak_ix = prf.shape[0] // 2, prf.shape[1] // 2
        peak = prf[peak_iy, peak_ix]

        cell_iy, cell_ix = np.indices(prf.shape)
        cell_radius = np.maximum(np.abs(cell_iy - peak_iy), np.abs(cell_ix - peak_ix))

        trim_radius = int(cell_radius[prf > peak * prf_thresh].max())
        keep_radius = int(cell_radius[prf > peak * keep_thresh].max())

        # Trim PRF so it looks over a smaller set of pixels
        trim_x = slice(peak_ix - trim_radius, peak_ix + trim_radius + 1)
        trim_y = slice(peak_iy - trim_radius, peak_iy + trim_radius + 1)

        # Stores the PRF contiguously in memory. No idea how much faster tbh but meh
        prf = np.ascontiguousarray(prf[trim_y, trim_x])
        prf_ny, prf_nx = prf.shape

        # The radius (in map pixels) at which the PRF falls below the `keep` limit, with some padding just in case.
        window_radius = keep_radius + pad
        window_nx = window_ny = 2 * window_radius + 1

        # Look up table.
        # Map pixel [y, x] -> row index into the cut-down pixel list. If pixel is not in the tile, set to -1.
        # The padding means sources slightly outside of the map (e.g. from expand_fwhm) are still accounted for.

        # Pad by a full window, needed for sliding_window_view
        grid_pad = window_nx

        index_grid, x_origin, y_origin = self._build_pix_grid(np.arange(self.npix), -1, grid_pad)
        grid_height, grid_width = index_grid.shape

        # This is basically some fancy slicing, which is far quicker than normaly Python slicing
        # As mentioned above, need to pad by a whole window width because the window view onnly looks at pixels where there is a full window to view.
        #
        # 3x3 grid:
        # [1,2,3]
        # [4,5,6]
        # [7,8,9].
        # Assuing window is a 2x2
        # window[0,0] is:
        # [1,2]
        # [4,5]
        # window[0,1]:
        # [2,3]
        # [5,6]
        # etc. The window index specifies the index of the top left of the window 
        # However window[2,2], or window[0,2], or window[2,0] would not exist, since there is not a full window for them.
        windows = sliding_window_view(index_grid, (window_ny, window_nx))

        # BASE PRF INDECES (?)
        # Accounts for the origin of the window being funky cos of the padding.
        base_x = np.arange(window_nx) - window_radius + trim_radius
        base_y = np.arange(window_ny) - window_radius + trim_radius

        # Per source position.
        # If `subpix`, accounts for subpixel position in source coordinates
        src_xi = np.rint(self.src_x).astype(int)
        src_yi = np.rint(self.src_y).astype(int)

        if subpix:
            sub_x = self.src_x - src_xi
            sub_y = self.src_y - src_yi
        else:
            sub_x = np.zeros(self.nsrc)
            sub_y = np.zeros(self.nsrc)

        # Transform from central position to corner position (to match how the windows are indexed)
        corner_x = src_xi - window_radius - x_origin
        corner_y = src_yi - window_radius - y_origin

        # Account for sources that may fall outside of the window entirely.
        # By default numpy would wrap around. No bueno.
        on_grid = (corner_x >= 0) & (corner_x + window_nx < grid_width) & (corner_y >= 0) & (corner_y + window_ny < grid_height)

        if np.any(~on_grid):
            warnings.warn(
                f"{np.sum(~on_grid)} sources dropped from the pointing matrix. "
                "Their windows fall outside the padded grid. Consider increasing the pad size."
                )

        keep_cut = peak * keep_thresh

        src_idx = np.flatnonzero(on_grid)

        if src_idx.size == 0:
            raise ValueError("There are no sources whose window falls within the padded grid.")

        amat_row = []
        amat_col = []
        amat_data = []

        for lo in range(0, src_idx.size, chunk):
            # Batched bilinear interpolation of the PRF, calculated manually.
            # It is much faster (if far less readable :/) to not use RegularGridInterpolator.
            # Effectively our grid can be separated onto its x and its y components, whereas RGI
            # always stores it as 2D.

            batch = src_idx[lo:lo + chunk]

            # Sample coordinates for this batch, accounting for window padding and subpixel position
            sample_x = base_x[None, :] - sub_x[batch][:, None]
            sample_y = base_y[None, :] - sub_y[batch][:, None]

            # Check which samples have valid PRF data. For bilinear interpolation, it requires the
            # four surrounding pixels to exist, hence the [0,nx-1] logic
            in_prf_x = (sample_x >= 0) & (sample_x <= prf_nx - 1)
            in_prf_y = (sample_y >= 0) & (sample_y <= prf_ny - 1)

            # Find which cell a source falls in, and the subpixel distance
            cell_x0 = np.floor(sample_x).astype(np.int64)
            cell_y0 = np.floor(sample_y).astype(np.int64)

            frac_x = sample_x - cell_x0
            frac_y = sample_y - cell_y0

            # Account for edge sources, without masking which would distort the shape.
            # Allows interpolation to not fail for edge sources, they will later be masked via in_prf
            np.clip(cell_x0, 0, prf_nx - 2, out=cell_x0)
            np.clip(cell_y0, 0, prf_ny - 2, out=cell_y0)

            # Add axes so they can be broadcast. More memory efficient
            col = cell_x0[:, None, :]
            row = cell_y0[:, :, None]

            wgt_x1 = frac_x[:, None, :]
            wgt_y1 = frac_y[:, :, None]

            wgt_x0 = 1.0 - wgt_x1
            wgt_y0 = 1.0 - wgt_y1

            # Manual bilinear interplation. First in x, then in y
            lower = prf[row, col] * wgt_x0
            lower += prf[row, col + 1] * wgt_x1
            upper = prf[row + 1, col] * wgt_x0
            upper += prf[row + 1, col + 1] * wgt_x1

            footprint = lower * wgt_y0
            footprint += upper * wgt_y1
            del lower, upper

            # Mask out the points that failed the edge case (i.e. the ones that were clipped)
            footprint *= in_prf_y[:, :, None] & in_prf_x[:, None, :]

            # Row index of the map pixels for each window. Masked or missing values have -1
            window = windows[corner_y[batch], corner_x[batch]]

            # Keep those pixels where the beam strength is above the amplitude cut and the
            # map pixel exists
            keep = (footprint > keep_cut) & (window >= 0)

            # Number of surviving pixels per source
            counts = keep.sum(axis=(1, 2))

            # Boolean indexing flattens in C order (source-major; source, y, x).
            # All of one source's entries are contiguous, hence np.repeat
            amat_data.append(footprint[keep])           # PRF strength
            amat_row.append(window[keep])               # Pixel index
            amat_col.append(np.repeat(batch, counts))   # Source index

            # Source amat_col[i] contributes amat_data[i]*src_flux to map pixel amat_row[i]
            # col indexes the source lists, row indexes the pixel list

        self.amat_data = np.concatenate(amat_data)
        self.amat_row = np.concatenate(amat_row)
        self.amat_col = np.concatenate(amat_col)

    def _compute_pointing_matrix_original(self):
        from scipy import interpolate
        paxis1, paxis2 = self.prf.shape

        amat_row = np.array([], dtype=int)
        amat_col = np.array([], dtype=int)
        amat_data = np.array([])

        centre = paxis1 // 2

        for s in range(self.nsrc):
            dx = -np.rint(self.src_x[s]).astype(int) + self.prf_x[paxis1//2] + self.map_x
            dy = -np.rint(self.src_y[s]).astype(int) + self.prf_y[paxis2//2] + self.map_y

            prf_x = self.prf_x
            prf_y = self.prf_y

            good = (dx >= 0) & (dx < self.prf_x[paxis1 - 1]) & (dy >= 0) & (dy < self.prf_y[paxis2 - 1])
            ngood = good.sum()
            bad = np.asarray(good) == False
            nbad = bad.sum()
            ipx2, ipy2 = np.meshgrid(prf_x, prf_y)
            atemp = interpolate.griddata((ipx2.ravel(), ipy2.ravel()),
                                         self.prf.ravel(),
                                         (dx[good], dy[good]),
                                         method = "linear")

            if atemp.size > 0:
                keep = atemp > np.max(atemp)/1e3

                amat_data = np.append(amat_data, atemp[keep])
                amat_row = np.append(amat_row, np.arange(self.npix, dtype = int)[good][keep])
                amat_col = np.append(amat_col, np.full(keep.sum(), s)) 

        self.amat_row = amat_row
        self.amat_col = amat_col
        self.amat_data = amat_data

    def _compute_pointing_matrix_v2(self):
        from scipy.interpolate import RegularGridInterpolator

        paxis1, paxis2 = self.prf.shape

        amat_row = []
        amat_col = []
        amat_data = []

        centre1 = paxis1 // 2
        centre2 = paxis2 // 2

        for s in range(self.nsrc):
            dx = -np.rint(self.src_x[s]).astype(int) + self.prf_x[paxis1//2] + self.map_x
            dy = -np.rint(self.src_y[s]).astype(int) + self.prf_y[paxis2//2] + self.map_y

            prf_x = self.prf_x
            prf_y = self.prf_y

            good = (dx >= 0) & (dx < self.prf_x[paxis1 - 1]) & (dy >= 0) & (dy < self.prf_y[paxis2 - 1])
            
            rgi = RegularGridInterpolator(
                    (prf_y, prf_x),
                    self.prf,
                    method = "linear",
                    bounds_error = False,
                    fill_value = 0.
                    )
            
            atemp = rgi(np.column_stack([dy[good], dx[good]]))

            if atemp.size > 0:
                keep = atemp > np.max(atemp)/1e3

                amat_data.append(atemp[keep])
                amat_row.append(np.arange(self.npix, dtype = int)[good][keep])
                amat_col.append(np.full(keep.sum(), s)) #

        self.amat_row = amat_row
        self.amat_col = amat_col
        self.amat_data = amat_data

    def _compute_pointing_matrix_v3(self, pad=2):
        """
        Compute the sparse pointing matrix.

        Equivalent to ``compute_pointing_matrix_v2`` but restricted to a fixed window around
        each source rather than scanning the whole map, and with the PRF interpolation hoisted
        out of the source loop entirely.

        Because source positions are quantised to the pixel grid via ``np.rint``, the PRF is
        sampled at identical offsets for every source, so the interpolation is done once for
        the whole catalogue. The per-source work is then a strided slice into a lookup grid.

        Results are stored in ``self.amat_data`` / ``self.amat_row`` / ``self.amat_col``, where
        rows index into the cut-down map pixel list (``self.map_x`` etc., length ``self.npix``)
        and columns index into the catalogue (``self.src_x`` etc., length ``self.nsrc``).

        Args:
            pad:
                Extra margin, in map pixels, added to each side of the window. Guards against
                off-by-one effects from the sub-pixel offset of the PRF centre. The default of
                2 is comfortable; raising it only costs a little wasted work.
        """
        from scipy.interpolate import RegularGridInterpolator

        paxis1, paxis2 = self.prf.shape
        centre1 = paxis1 // 2
        centre2 = paxis2 // 2

        prf_x = np.asarray(self.prf_x, dtype=float)
        prf_y = np.asarray(self.prf_y, dtype=float)

        # ------------------------------------------------------------------
        # Window size, in map pixels, guaranteed to cover the PRF footprint.
        # ------------------------------------------------------------------
        nwx = int(np.ceil(prf_x[-1] - prf_x[0])) + 1 + 2 * pad
        nwy = int(np.ceil(prf_y[-1] - prf_y[0])) + 1 + 2 * pad

        # ------------------------------------------------------------------
        # Lookup grid: map pixel (y, x) -> row index into the cut-down pixel
        # list, or -1 where that pixel is not in the tile (masked / outside
        # the MOC). Padded by a full window on every side so that no slice
        # taken below can run off the edge.
        # ------------------------------------------------------------------
        map_xi = np.rint(self.map_x).astype(np.int64)
        map_yi = np.rint(self.map_y).astype(np.int64)

        x_min, y_min = map_xi.min(), map_yi.min()

        grid_w = (map_xi.max() - x_min) + 1 + 2 * nwx
        grid_h = (map_yi.max() - y_min) + 1 + 2 * nwy

        index_grid = np.full((grid_h, grid_w), -1, dtype=np.int64)
        index_grid[map_yi - y_min + nwy, map_xi - x_min + nwx] = np.arange(self.npix)

        # ------------------------------------------------------------------
        # Source-independent PRF footprint.
        #
        # For a source at integer pixel position sxi, the window starts at
        # map pixel  x_start = sxi - floor(prf_x[centre1]) - pad,  so
        #
        #   dx = -sxi + prf_x[centre1] + (x_start + arange(nwx))
        #      = arange(nwx) - floor(prf_x[centre1]) + prf_x[centre1] - pad
        #
        # i.e. sxi cancels and dx is the same for every source.
        # ------------------------------------------------------------------
        cx, cy = prf_x[centre1], prf_y[centre2]
        fx, fy = int(np.floor(cx)), int(np.floor(cy))

        dx = np.arange(nwx) - fx + cx - pad
        dy = np.arange(nwy) - fy + cy - pad
        dxx, dyy = np.meshgrid(dx, dy)

        # Same bounds test as v2, but evaluated once rather than per source.
        in_prf = (dxx >= 0) & (dxx < prf_x[-1]) & (dyy >= 0) & (dyy < prf_y[-1])

        rgi = RegularGridInterpolator(
            (prf_y, prf_x),
            self.prf,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        footprint = np.zeros((nwy, nwx), dtype=float)
        footprint[in_prf] = rgi(np.column_stack([dyy[in_prf], dxx[in_prf]]))

        # ------------------------------------------------------------------
        # Per-source window origins in grid coordinates.
        # ------------------------------------------------------------------
        src_xi = np.rint(self.src_x).astype(np.int64)
        src_yi = np.rint(self.src_y).astype(np.int64)

        gx0 = src_xi - fx - pad - x_min + nwx
        gy0 = src_yi - fy - pad - y_min + nwy

        # Sources whose window would fall outside the padded grid entirely.
        # With a catalogue cut to the map's MOC this should never fire, but a
        # negative start would silently wrap under numpy indexing, so guard it.
        on_grid = (gx0 >= 0) & (gx0 + nwx <= grid_w) & (gy0 >= 0) & (gy0 + nwy <= grid_h)

        amat_row = []
        amat_col = []
        amat_data = []

        for s in np.flatnonzero(on_grid):
            window = index_grid[gy0[s]:gy0[s] + nwy, gx0[s]:gx0[s] + nwx]

            # In the PRF's support, and a pixel that actually exists in the tile.
            good = in_prf & (window >= 0)
            if not good.any():
                continue

            atemp = footprint[good]

            # Threshold is per source, over the pixels that source actually has,
            # matching v2. An edge source missing its brightest pixel therefore
            # gets a correspondingly lower threshold.
            keep = atemp > atemp.max() / 1e3
            if not keep.any():
                continue

            amat_data.append(atemp[keep])
            amat_row.append(window[good][keep])
            amat_col.append(np.full(keep.sum(), s, dtype=np.int64))

        if amat_data:
            self.amat_data = np.concatenate(amat_data)
            self.amat_row = np.concatenate(amat_row).astype(np.int64)
            self.amat_col = np.concatenate(amat_col).astype(np.int64)
        else:
            self.amat_data = np.array([], dtype=float)
            self.amat_row = np.array([], dtype=np.int64)
            self.amat_col = np.array([], dtype=np.int64)

    def _compute_pointing_matrix_jax(self, batch_size = 1024, pad = 2):
        import jax.numpy as jnp
        from jax import lax
        from jax.scipy.ndimage import map_coordinates

        paxis1, paxis2 = self.prf.shape
        centre1 = int(round((paxis1 - 1) / 2))
        centre2 = int(round((paxis2 - 1) / 2))

        pindx_np = np.asarray(self.prf_x, dtype=np.float64)
        pindy_np = np.asarray(self.prf_y, dtype=np.float64)
        x0, dxs = float(pindx_np[0]), float(pindx_np[1] - pindx_np[0])
        y0, dys = float(pindy_np[0]), float(pindy_np[1] - pindy_np[0])
        x_max, y_max = float(pindx_np[-1]), float(pindy_np[-1])

        # fixed window size (image pixels) that comfortably covers the PRF footprint
        nwx = int(np.ceil(x_max - x0)) + 1 + 2 * pad
        nwy = int(np.ceil(y_max - y0)) + 1 + 2 * pad

        # --- build a padded lookup grid: image pixel (y, x) -> row index into
        # sx_pix/sy_pix/sim, or -1 if that pixel isn't in the cut-down tile ---
        sx_pix_int = np.rint(self.map_x).astype(np.int64)
        sy_pix_int = np.rint(self.map_y).astype(np.int64)
        x_min, x_max_img = sx_pix_int.min(), sx_pix_int.max()
        y_min, y_max_img = sy_pix_int.min(), sy_pix_int.max()

        half_x = nwx // 2 + 1
        half_y = nwy // 2 + 1

        grid_w = (x_max_img - x_min) + 1 + 2 * half_x
        grid_h = (y_max_img - y_min) + 1 + 2 * half_y

        index_grid_np = np.full((grid_h, grid_w), -1, dtype=np.int32)
        gx = sx_pix_int - x_min + half_x
        gy = sy_pix_int - y_min + half_y
        index_grid_np[gy, gx] = np.arange(self.npix, dtype=np.int32)

        index_grid = jnp.asarray(index_grid_np)
        prf = jnp.asarray(self.prf, dtype=jnp.float32)
        pindx = jnp.asarray(self.prf_x, dtype=jnp.float32)
        pindy = jnp.asarray(self.prf_y, dtype=jnp.float32)
        sx = jnp.asarray(self.src_x, dtype=jnp.float32)
        sy = jnp.asarray(self.src_y, dtype=jnp.float32)

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

    def _compute_upper_lims_original(self):
        """
        Original version of upper limti calculation.
        Requires pointing matrix to be computed
        """
        self.prior_flux_upper = np.full((self.nsrc), 1000.0)
        for i in range(0, self.nsrc):
            ind = self.amat_col == i
            if np.sum(ind) > 0:
                # self.prior_flux_upper[i] = np.max(self.map_flux[self.amat_row[ind]]) + (np.abs(self.bkg[0]) + 2 * self.bkg[1])
                self.prior_flux_upper[i] = np.max(self.map_flux[self.amat_row[ind]]) + (np.abs(self.bkg_mu) + 2 * self.bkg_sigma)

    def _compute_upper_lims_v2(self):
        """
        Updated version of upper limit calculation.

        Much faster than original, but still looks over the entire area a source contributes to.
        Still needs 
        """
        self.prior_flux_upper = np.full(self.nsrc, -np.inf)

        # bkg_term = np.abs(self.bkg[0]) + 2 * self.bkg[1]
        bkg_term = np.abs(self.bkg_mu) + 2 * self.bkg_sigma

        values = self.map_flux[self.amat_row]

        # grouped max
        np.maximum.at(self.prior_flux_upper, self.amat_col, values)

        # remove -inf for empty sources
        empty = ~np.isfinite(self.prior_flux_upper)
        self.prior_flux_upper[empty] = 1000.0

        self.prior_flux_upper[~empty] += bkg_term