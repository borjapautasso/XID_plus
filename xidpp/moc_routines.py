from pymoc import MOC
from healpy import pixelfunc
import numpy as np
from numpy.typing import ArrayLike, NDArray


def get_healpix_pixels(order: int,
                       ra: ArrayLike,
                       dec: ArrayLike,
                       unique: bool = True
                       ) -> NDArray[np.integer]:
    """
    Work out what HEALPix pixel sources are in. 

    Parameters
    ----------
    order : int
        HEALPix order.
    ra : array_like
        Right Ascension of sources, in degrees.
    dec : array_like
        Declination of sources, in degrees.
    unique : bool, optional
        Whether to return a unique set of HEALPix indeces, or one per source.
    
    Returns
    -------
    np.ndarray of int
        HEALPix pixel index (or indeces), in the nested scheme.
    """
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)

    theta = np.pi/2 - np.radians(dec)
    phi = np.radians(ra)

    ipix = pixelfunc.ang2pix(2**order, theta, phi, nest = True)

    if unique:
        return np.unique(ipix)

    return ipix


def create_moc_from_cat(ra: ArrayLike, dec: ArrayLike, order:int = 11) -> MOC:
    """
    Generate MOC from catalogue.

    Parameters
    ----------
    ra : array_like
        Right Ascension of sources, in degrees.
    dec : array_like
        Declination of sources, in degrees.
    order: int
        HEALPix order of the MOC.
    Returns
    -------
    pymoc.MOC
        MOC covering the catalogue's source positions.
    """
    pixels = get_healpix_pixels(order, ra, dec)

    moc = MOC()
    moc.add(order, pixels)

    return moc

def check_in_moc(ra: ArrayLike, dec: ArrayLike, moc: MOC) -> NDArray[np.bool]:
    """
    Check whether sources are within the MOC.

    Parameters
    ----------
    ra : array_like
        Right Ascension of sources, in degrees.
    dec : array_like
        Declination of sources, in degrees.
    moc : pymoc.MOC
        The MOC to check against.

    Returns
    -------
    np.ndarray of bool
        Boolean mask, True for sources within the MOC.
    """
    source_healpix_pixels = get_healpix_pixels(moc.order, ra, dec, unique = False)

    moc_healpix_pixels = np.array(list(moc.flattened()))

    return np.isin(source_healpix_pixels, moc_healpix_pixels)

def check_sources_in_tile(pixels: ArrayLike,
                          order: int,
                          ra: ArrayLike,
                          dec: ArrayLike
                          ) -> NDArray[np.bool]:
    """
    Check if sources fall within the given HEALPix pixels.

    Parameters
    ----------
    pixels : array_like of int
        HEALPix pixel index (or indices).
    order : int
        HEALPix order of ``pixels``.
    ra : array_like
        Right Ascension of sources, in degrees.
    dec : array_like
        Declination of sources, in degrees.

    Returns
    -------
    np.ndarray of bool
        Boolean mask, True for sources within the pixels.
    """
    pixels = np.atleast_1d(pixels)

    pixels_moc = MOC()
    pixels_moc.add(order, pixels)

    return check_in_moc(ra, dec, pixels_moc)


def get_large_tile(order_small: int, tile_small: int, order_large: int) -> int:
    """
    Return HEALPix index of large (i.e. parent) tile.

    Parameters
    ----------
    order_small : int
        HEALPix order of the small tile.
    tile_small : int
        HEALPix index of the small tile.
    order_large : int
        HEALPix order of the large tile.

    Returns
    -------
    int
        HEALPix index of the large tile.
    """
    theta, phi = pixelfunc.pix2ang(2**order_small, tile_small, nest = True)

    return pixelfunc.ang2pix(2**order_large, theta, phi, nest = True)


def create_fitting_area(order: int, pixels: ArrayLike, padding_order: int = 11) -> MOC:
    """
    Create MOC from given HEALPix pixel(s). Optionally expand this MOC by a
    ring of ``padding_order`` tiles.
    
    Parameters
    ----------
    order : int
        HEALPix order of ``pixels``.
    pixels : array_like of int
        HEALPix pixel index (or indices).
    padding_order : int, optional
        HEALPix order of the padded region.

    Returns
    -------
    pymoc.MOC
        MOC covering the fitting area.
    """
    pixels = np.atleast_1d(pixels)

    if padding_order == -1:
        pixels_moc = MOC()
        pixels_moc.add(order, pixels)

        return pixels_moc


    level_diff = padding_order - order

    if level_diff < 0:
        raise ValueError(f"padding_order ({padding_order}) must be >= order ({order}).")

    # Number of subpixels
    nsub = 4**level_diff

    offsets = np.arange(nsub)

    # Using the hierarchical HEALPix property.
    # Tile 0 at order x is composed of Tiles 0->3 at order x+1 (i.e.  4**(x+1 - x) subpixels).
    # This does the same but for a generalised order difference, and for mulitple tiles at once.
    # Basically converts our main tile to its equivalent in a different order.
    subpixels = np.ravel(nsub * pixels[:,None] + offsets[None, :])
    subpixels = np.unique(subpixels)

    # Gets all direct neighbours of the main tiles
    neighbours = np.unique(pixelfunc.get_all_neighbours(2**padding_order, subpixels, nest = True))
    neighbours = neighbours[neighbours >= 0]

    # Union of interior + ring
    all_pixels = np.unique(np.concatenate([subpixels, neighbours]))

    moc_tile = MOC()
    moc_tile.add(padding_order, all_pixels)

    return moc_tile