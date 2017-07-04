from __future__ import print_function, division
import os
import time
import tempfile
import numpy as np
import astropy.io.fits as fits
from subprocess import call


class binnedmap(object):
    def __init__(self, x, y, v=None, w=None, npix=None, pixsize=None):
        # TODO
        pass


def bin2d(x, y, v=None, w=None, npix=None, verbose=False, timed=False):
    """
    Bin values v according to position (x, y), taking the average of values
    falling into the same bin. Averages are weighted if w is provided. If
    v is not given, return the bin count map.

    Bin edges are computed according to npix such that in each dimension,
    the min (max) position value lies at the center of its first (last) bin.

    Parameters
    ----------
    x, y: array_like
        Position arrays.
    v: array, optional
        Values to bin, potentially many arrays of len(x) as [v1, v2, ...].
    w: array, optional
        Weight values for v.
    npix: int list as [nx, ny], optional
        If npix = N, use [N, N]. Defaults to [16, 16] if not provided.
    verbose: bool
        If true, print details.
    timed: bool
        If true, print total time taken.

    Returns
    -------
    2d array of values v binned into pixels.
    """
    start_time = time.time()

    # TODO: verify inputs
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if v is not None:
        v = np.atleast_1d(v)
        if len(v.shape) == 1:
            v = [v]
    if w is not None:
        w = np.atleast_1d(w)
    else:
        w = np.ones_like(x)

    if npix is not None:
        npix = map(int, np.atleast_1d(npix)) # Note: map() returns a list
        if len(npix) == 2:
            n = npix
        elif len(npix) == 1:
            n = 2 * npix
        else:
            print("Invalid npix. Returning None.")
            return None
    else:
        n = [16, 16]

    # Determine 2D space geometry
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    halfdx = float(xmax - xmin) / (2 * n[0] - 2)
    halfdy = float(ymax - ymin) / (2 * n[1] - 2)
    xlow = xmin - halfdx
    xhigh = xmax + halfdx
    ylow = ymin - halfdy
    yhigh = ymax + halfdy
    xedges = np.linspace(xlow, xhigh, n[0] + 1)
    yedges = np.linspace(ylow, yhigh, n[1] + 1)

    if verbose:
        print("xmin, xmax:  {0}, {1}".format(xmin, xmax))
        print("xlow, xhigh: {0}, {1}".format(xlow, xhigh))
        print("dx size:   {}".format(halfdx * 2))
        print("ymin, ymax:  {0}, {1}".format(ymin, ymax))
        print("ylow, yhigh: {0}, {1}".format(ylow, yhigh))
        print("dy size:   {}".format(halfdy * 2))

    # Do fast binning on 1D arrays
    indx = np.digitize(x, xedges) - 1
    indy = np.digitize(y, yedges) - 1
    size_1d = n[1] * n[0]
    ind_1d = indy * n[0] + indx

    if v is None:
        # Determine pixel counts and return
        nmap = np.bincount(ind_1d, minlength=size_1d)
        if timed:
            print("Time: {0:.3f} s".format(time.time() - start_time))
        return nmap.reshape(n[1], n[0])

    # Weight sums map
    wmap = np.bincount(ind_1d, weights=w, minlength=size_1d)
    # Avoid division by zero in empty pixels
    nmap = np.copy(wmap) # TODO Can we do this without copying ?
    nmap[wmap == 0] = 1
    # Compute binned v maps
    vmaps = [np.bincount(ind_1d, weights=(v[i] * w), minlength=size_1d) / nmap
             for i in range(len(v))]
    if len(vmaps) == 1:
        binnedmap = np.reshape(vmaps[0], (n[1], n[0]))
    else:
        binnedmap = np.reshape(vmaps, (len(v), n[1], n[0]))

    if timed:
        print("Time: {0:.3f} s".format(time.time() - start_time))
    return binnedmap


def bin3d(x, y, z, v=None, w=None, npix=None, verbose=False):
    # TODO
    pass


def pix2radec(extent, npix, X, Y):
    print("NEED TO CHECK THIS.")
    # Based on bin2d binning scheme
    npix = np.atleast_1d(npix)
    if len(npix) == 1:
        nx = npix
        ny = npix
    elif len(npix) == 2:
        nx = npix[0]
        ny = npix[1]
    else:
        print("Invalid npix.")
        return

    ra_min, ra_max = extent[:2]
    dec_min, dec_max = extent[2:]
    dra = float(ra_max - ra_min) / (nx - 1)
    ddec = float(dec_max - dec_min) / (ny - 1)
    ra = ra_min + X * dra
    dec = dec_min + Y * ddec
    return ra, dec


# def radec2pix(extent, ra, dec):
#     # Fetch config parameters
#     ra0, dec0, X0, Y0, npix, pixel_size = get_config(source)
#
#     if not isinstance(ra, u.quantity.Quantity):
#         ra = ra * u.degree
#     if not isinstance(dec, u.quantity.Quantity):
#         dec = dec * u.degree
#
#     X = X0 + (ra.to(u.degree) - ra0.to(u.degree)) / pixel_size
#     Y = Y0 + (dec.to(u.degree) - dec0.to(u.degree)) / pixel_size
#     return X.value, Y.value


def mr_transform(image, nscales=5, verbose=False):
    """Compute the multiresolution starlet transform of an image."""
    # Create temporary directory to hold image and its transform
    tmpdir = tempfile.mkdtemp()
    saved_umask = os.umask(0077)
    image_path = os.path.join(tmpdir, 'image.fits')
    mr_path = os.path.join(tmpdir, 'image.mr')
    if verbose:
        print("Creating {}".format(image_path))
        print("Creating {}".format(mr_path))

    # Call mr_transform on the saved image
    try:
        if verbose:
            print("Calling mr_transform.")
        fits.writeto(image_path, image)
        call(['mr_transform', '-n', str(nscales + 1), image_path, mr_path])
        mr = fits.getdata(mr_path)
    except IOError as e:
        print("IOError")
    # If successful, remove file paths
    else:
        os.remove(image_path)
        os.remove(mr_path)
        if verbose:
            print("Success.")
            print("Removing {}".format(image_path))
            print("Removing {}".format(mr_path))
    # Remove temporary directory
    finally:
        os.umask(saved_umask)
        os.rmdir(tmpdir)
        if verbose:
            print("Removing {}".format(tmpdir))

        if (os.path.exists(tmpdir) or os.path.exists(image_path) or
            os.path.exists(mr_path)):
            print("Oops, not all files or directories were removed.")

    return mr


def annulus(image, x0, y0, r1, r2, **kwargs):
    """
    Returns the values of image lying within the annulus defined by
    r1 <= r <= r2 pixels about the point (x0, y0).
    """
    nx, ny = image.shape
    xv, yv = np.meshgrid(range(nx), range(ny), sparse=False, indexing='ij')
    mask = (((xv - x0)**2 + (yv - y0)**2 >= int(r1)**2) &
            ((xv - x0)**2 + (yv - y0)**2 <= int(r2)**2))

    return image[mask]


def ks_inversion(e1map, e2map):
    """
    Compute Kaiser-Squires inversion mass map from binned e1 and e2 maps.
    Return E-mode and B-mode kappa maps.
    """
    # e1map and e2map should be the same size
    (nx, ny) = e1map.shape
    # Compute Fourier space grid
    # NOTE: need to reverse the order of nx, ny to achieve proper k1, k2 shapes
    k1, k2 = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))
    # Compute Fourier transforms of e1 and e2
    g1 = np.fft.fft2(e1map)
    g2 = np.fft.fft2(e2map)
    # Apply convolution kernel (multiplication in Fourier space)
    numer = ((k1 * k1 - k2 * k2) - 2j * (k1 * k2)) * (g1 + 1j * g2)
    denom = k1 * k1 + k2 * k2
    denom[0, 0] = 1  # avoid division by 0
    kappa = numer / denom
    # Transform back to real space
    kappaE = np.real(np.fft.ifft2(kappa))
    kappaB = np.imag(np.fft.ifft2(kappa))

    return kappaE, kappaB

# def ks_inversion_gen(e1, e2):
#     """
#     Implementation of generalized Kaiser & Squires inversion as in
#     Seitz & Schneider (2001).
#     """
#     pass
