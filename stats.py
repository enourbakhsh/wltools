from __future__ import print_function, division
import numpy as np
from scipy.stats import gaussian_kde
from .utils import get_column_names, angular_separation


def gaussian_kde2d(x, y, x_prior, y_prior, gridsize=512):
    xmin, xmax = x_prior
    ymin, ymax = y_prior
    X, Y = np.mgrid[xmin : xmax : gridsize * 1j, ymin : ymax : gridsize * 1j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values, bw_method='silverman')
    Z = np.reshape(kernel(positions).T, X.shape)

    raveled = Z.ravel()
    inds = np.argsort(raveled)[::-1]
    normed = raveled / kernel.integrate_box([xmin, ymin], [xmax, ymax])
    one_sig_index = (normed[inds].cumsum() / normed.sum() <= 0.6827).sum()
    two_sig_index = (normed[inds].cumsum() / normed.sum() <= 0.9545).sum()
    three_sig_index = (normed[inds].cumsum() / normed.sum() <= 0.9973).sum()
    level1 = raveled[inds[one_sig_index]]
    level2 = raveled[inds[two_sig_index]]
    level3 = raveled[inds[three_sig_index]]

    return Z, [level3, level2, level1]


def power_spectrum(image):
    # Compute fft of image
    transformed = np.fft.fft2(image)
    # Shift low frequencies to center
    shifted = np.fft.fftshift(transformed)
    # Compute squared amplitude
    amplitude = np.abs(shifted)**2
    # Compute mean power over k modes
    power = azimuthal_average(amplitude)

    npix = max(image.shape)
    nk = len(power)
    k = np.linspace(0, nk / npix, nk)

    return k, power / npix**2


def azimuthal_average(image, center=None, mode='trunc'):
    """
    Compute the azimuthally averaged radial profile of image about center.
    If a center (x0, y0) is not provided, it will be taken to be the pixel
    center of the image.

    Keyword mode is either 'trunc' or 'round'. This describes whether the
    distance values from center should be truncated or rounded to the nearest
    integer. Default is 'trunc'.
    """
    # Check inputs
    if mode not in ['trunc', 'round']:
        mode = 'trunc'

    # Generate index grids
    y, x = np.indices(image.shape)

    # Determine image center
    if center is None:
        y0, x0 = (np.array(image.shape) - [1, 1]) / 2

    # Compute distances from center
    radii = np.hypot(x - x0, y - y0)

    # Reorder image pixels (in 1D) by distance from center
    inds = np.argsort(radii.flat)
    radii_1d = radii.flat[inds]
    image_1d = image.flat[inds]

    # Determine radial binning according to mode
    if mode == 'trunc':
        rbins = radii_1d.astype(int)
    else:
        rbins = radii_1d.round().astype(int)

    # deltar = rbins[1:] - rbins[:-1]  # Assumes all radii represented
    # rind = np.where(deltar)[0]       # location of changed radius
    # nr = rind[1:] - rind[:-1]        # number of radius bin
    #
    # # Cumulative sum to figure out sums for each radius bin
    # csim = np.cumsum(image_1d, dtype=float)
    # tbin = csim[rind[1:]] - csim[rind[:-1]]
    #
    # radial_prof = tbin / nr
    # return radial_prof

    # Compute average radial profile
    numbins = np.bincount(rbins)
    radial_profile = np.bincount(rbins, weights=image_1d) / numbins

    return radial_profile


def mean_radial_profile(x, y, v, center=(0, 0), bin_edges=None, nbins=None):
    """
    Compute the mean radial profile of values v about the center point.

    If bin_edges is not provided, default to 10 bins linearly spaced between
    0 and the radius of the smallest circle that fits inside the field
    about the center point.

    Returns
    -------
    bin_edges, profile, error

    Error is computed as the standard error on the mean profile.
    """
    if bin_edges is None:
        rmax = min(x.max() - center[0], center[0] - x.min(),
                   y.max() - center[1], center[1] - y.min())
        if nbins is None:
            nbins = 10
        bin_edges = np.linspace(0, rmax, nbins + 1)
    else:
        nbins = len(bin_edges) - 1

    # Compute distances from center
    # TODO Add option to use wltools.utils.angular_separation if on the sky
    dx = x - center[0]
    dy = y - center[1]
    radii = np.hypot(dx, dy)

    # Do radial binning
    inds = np.digitize(radii, bin_edges) - 1
    # Leave out points that fall inside the smallest radial bin
    valid = (inds >= 0)
    inds = inds[valid]
    v = v[valid]

    # Compute mean profile and error
    bincounts = np.bincount(inds, minlength=nbins)
    # Avoid division by zero for empty bins
    bad_inds = (bincounts == 0)
    bincounts[bad_inds] = 1
    # Mean profile
    binsums = np.bincount(inds, weights=v, minlength=nbins)
    binsums[bad_inds] = 0
    profile = binsums / bincounts
    # Std error on the mean
    delta = v - profile[inds]
    errsums = np.bincount(inds, weights=delta**2, minlength=nbins)
    errsums[bad_inds] = 0
    error = np.sqrt(errsums) / bincounts
    # Leave out values beyond the maximum bin edge
    profile = profile[:nbins]
    error = error[:nbins]

    return bin_edges, profile, error


# def random_ift(size=128, N=1):
#     # Set N live pixels in a field of zeros (size, size)
#     image = np.zeros((size, size))
#     for n in range(N):
#         x = np.random.randint(size)
#         y = np.random.randint(size)
#         image[x, y] = np.random.randint(5)
#     # Compute FFT
#     transformed = np.fft.fft2(image)
#     print(transformed[0, 0])
#     # Shift low frequencies to center
#     shifted = np.fft.fftshift(transformed)
#     print(shifted[size / 2, size / 2])
#     # Plot
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#     plot_map(image, fig, ax1, cmap='magma')
#     plot_map(np.abs(shifted), fig, ax2, cmap='bone')
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#     plot_map(image, fig, ax1, cmap='magma')
#     plot_map(np.abs(shifted), fig, ax2, cmap='bone')
#     plt.show()
