# Functions related to peak finding, etc.
from __future__ import print_function, division
import numpy as np
from .utils import mad


def find_peaks(image, threshold):
    """
    Finds peaks in an image above the specified threshold.

    Parameters
    ----------
    image: ndarray
        2D array containing the image in which to find peaks.

    Returns
    -------
    (x, y): ndarray
        Tuple of x and y arrays containing indices of peaks.
    """
    # Extract center map
    map0 = image[1:-1, 1:-1]

    # Extract shifted maps
    map1 = image[0:-2, 0:-2]
    map2 = image[1:-1, 0:-2]
    map3 = image[2:,   0:-2]
    map4 = image[0:-2, 1:-1]
    map5 = image[2:,   1:-1]
    map6 = image[0:-2, 2:  ]
    map7 = image[1:-1, 2:  ]
    map8 = image[2:,   2:  ]

    # Compare center map with shifted maps
    merge = ( (map0 > map1) & (map0 > map2) & (map0 > map3) & (map0 > map4)
            & (map0 > map5) & (map0 > map6) & (map0 > map7) & (map0 > map8) )

    bordered = np.lib.pad(merge, (1, 1), 'constant', constant_values=(0, 0))
    peaks = image * bordered
    indices = np.nonzero(peaks > threshold)

    return indices


def find_peaks2d(image, threshold=None, ordered=True, mask=None):
    """
    Simple and fast 2D peak finder that identifies peaks in an image with
    amplitude above a given threshold.

    Parameters
    ----------
    image : array_like, 2-dimensional
        Input image.
    threshold : float, optional
        Minimum pixel amplitude to be considered as a peak.
        Default value is set to MAD(image).
    ordered : boolean
        If True, return peaks in decreasing order.
    mask : array_like, same shape as image, optional
        Boolean array identifying which pixels of M to consider in finding
        peaks. If no absolute `threshold' is provided, the default is computed
        as MAD(image[mask]).

    Returns
    -------
    X, Y : 1D numpy arrays
        Pixel indices of peak positions.

    Notes
    -----
    Peaks on the border of the image are not found by this method.

    Examples
    --------
    TODO
    """
    # Deal with mask first
    if mask is not None:
        if mask.shape != image.shape:
            print("Warning: mask not compatible with image. Ignoring.")
            mask = np.ones_like(image)
    else:
        mask = np.ones_like(image)

    # Determine threshold level
    if threshold is None:
        threshold = mad(image[mask.astype('bool')])

    # Extract center map
    map0 = image[1:-1, 1:-1]

    # Extract shifted maps
    map1 = image[0:-2, 0:-2]
    map2 = image[1:-1, 0:-2]
    map3 = image[2:,   0:-2]
    map4 = image[0:-2, 1:-1]
    map5 = image[2:,   1:-1]
    map6 = image[0:-2, 2:  ]
    map7 = image[1:-1, 2:  ]
    map8 = image[2:,   2:  ]

    # Compare center map with shifted maps
    merge = ( (map0 > map1) & (map0 > map2) & (map0 > map3) & (map0 > map4)
            & (map0 > map5) & (map0 > map6) & (map0 > map7) & (map0 > map8) )

    bordered = np.lib.pad(merge, (1, 1), 'constant', constant_values=(0, 0))
    peaksmap = image * bordered * mask
    X, Y = np.nonzero(peaksmap > threshold)

    if ordered:
        amps = image[X, Y]
        inds = np.argsort(amps)[::-1]
        return X[inds], Y[inds]

    return X, Y


def get_ordered_peaks(image, threshold=None, mask=None):
    """
    Find peaks in image with amplitude above threshold.
    Return positions and amplitude values ordered from highest to lowest.
    """
    if mask is not None:
        if mask.shape == image.shape:
            image[~mask] = 0
        else:
            print("Mask not compatible with image. Ignoring...")
    else:
        mask = np.ones_like(image).astype('bool')

    if threshold is None:
        x, y = find_peaks(image, mad(image[mask]))
    else:
        x, y = find_peaks(image, threshold)

    amps = np.array([image[xx, yy] for (xx, yy) in zip(x, y)])
    inds = np.argsort(amps)[::-1]

    return x[inds], y[inds], amps[inds]

    # PREVIOUS VERSION
    # y, x = find_peaks(image, threshold)
    # amps = np.array([image[yy, xx] for (xx, yy) in zip(x, y)])
    # inds = np.argsort(amps)[::-1]
    # x = x[inds]
    # y = y[inds]
    # amps = amps[inds]
    # if verbose:
    #     for (xx, yy, aa) in zip(x, y, amps):
    #         print("({}, {}) : {}".format(xx, yy, aa))
    #
    # return x, y, amps


def get_nearest_peaks(image, targets, threshold, max_dist):
    """
    Return the position of the nearest peak in image to the target position
    if its amplitude is above threshold and within max_dist of target.
    """
    # Find peaks in image above threshold
    x, y, amps = get_ordered_peaks(image, threshold)
    if x is None:
        print("Found no peaks above threshold.")
        return

    def min_dist_match(target, x, y):
        x0, y0 = target
        distances = (x - x0)**2 + (y - y0)**2
        if min(distances) <= max_dist**2:
            index = np.argmin(distances)
            # match = (x[index], y[index], amps[index])
            return x[index], y[index]
        else:
            return None, None

    matches = np.array([min_dist_match(t, x, y) for t in targets])
    return matches


def match_peaks2d(image, targets, threshold=None, maxdist=None, extent=None):
    """
    Find the peaks in image closest to the given targets.

    !!-----------------------------------------------------!!
    !! WARNING : extent option does not work properly yet. !!
    !!-----------------------------------------------------!!

    Parameters
    ----------
    image : array_like, 2-dimensional
        Input image.
    targets : array_like with shape (N, 2)
        Pixel or coordinate positions (see `extent' option) of N targets.
        Ex. if positions are referenced as x=[x1, x2, ...], y=[y1, y2, ...],
        then `targets' can be passed as zip(x, y).
    threshold : float, optional
        Minimum pixel amplitude to be considered as a peak. Default value is
        set to MAD(image). (cf. wltools.find_peaks2d)
    maxdist : float, optional
        Search radius in pixel space around each target.
    extent : array_like, optional
        Coordinate boundary values as given as [xmin, xmax, ymin, ymax].
        If provided, distances are computed in coordinate space instead of
        pixel space, and therefore `maxdist' should

    Returns
    -------
    X, Y : 1d_arrays of length N
        Indices of peak positions matched to targets. If no matching peak
        is found for target at index i, X[i], Y[i] = (None, None).

    Notes
    -----
    Since this method uses wltools.find_peaks2d, peaks on the border of the
    input image will not be found. Distances are computed using the
    Euclidean metric, so this method is likely not appropriate for
    applications on the curved sky.

    Examples
    --------
    TODO
    """
    import matplotlib.pyplot as plt
    from wltools.plotting import plot_map
    from shapely.geometry import Point

    # Unpack targets
    tx, ty = np.atleast_2d(targets).T
    ntargets = len(tx)

    # Prepare return arrays
    resX = np.array([None] * ntargets)
    resY = np.array([None] * ntargets)

    # Find peaks in pixel space
    px, py = find_peaks2d(image, threshold)
    npeaks = len(px)
    if npeaks == 0:
        print("Warning: no peaks found.")
        return resX, resY

    if maxdist is None:
        maxdist = np.inf

    # if extent is not None:
    #     # Determine image size
    #     NX, NY = np.atleast_2d(image).shape
    #     print("size : {} x {}".format(NX, NY))
    #     # Unpack coordinate edge values
    #     xmin, xmax, ymin, ymax = extent
    #     # Compute coordinate values
    #     tx = xmin + tx * (xmax - xmin) / NX
    #     ty = ymin + ty * (ymax - ymin) / NY
    #     px = xmin + px * (xmax - xmin) / NX
    #     py = xmin + py * (ymax - ymin) / NY

    tx_matrix = np.tile(tx, (npeaks, 1)).T
    ty_matrix = np.tile(ty, (npeaks, 1)).T
    px_matrix = np.tile(px, (ntargets, 1))
    py_matrix = np.tile(py, (ntargets, 1))

    dx = tx_matrix - px_matrix
    dy = ty_matrix - py_matrix
    dists = np.hypot(dx, dy)

    mininds = np.argmin(dists, axis=1)
    mindists = dists[range(ntargets), mininds]
    keepinds = mindists <= maxdist

    resX[keepinds] = px_matrix[keepinds, mininds[keepinds]]
    resY[keepinds] = py_matrix[keepinds, mininds[keepinds]]

    # Visual sanity check
    fig, ax = plt.subplots(1, 1)
    plot_map(image, fig, ax, cmap='bone', extent=extent)
    ax.scatter(py, px)
    # for i, (px0, py0) in enumerate(zip(px, py)):
    #     ax.text(py0, px0, "{}".format(i + 1))
    ax.scatter(ty, tx, marker='x')
    if maxdist is not np.inf:
        for (tx0, ty0) in zip(tx, ty):
            circle = Point(ty0, tx0).buffer(maxdist)
            cx, cy = circle.exterior.xy
            ax.plot(cx, cy, linestyle='dashed')

    return resX, resY


def find_peaks_v2(image, threshold):
    """
    Find peaks in image as local maxima using the 24 surrounding neighbors
    of each pixel.

    Parameters
    ----------
    image: ndarray
        2D array containing the image in which to find peaks.

    Returns
    -------
    (x, y): ndarray
        Tuple of x and y arrays containing indices of peaks.
    """
    y1, x1 = find_peaks(image, threshold)
    indices1 = zip(x1, y1)
    mask = np.zeros_like(image)
    for index in indices1:
        mask[index[1], index[0]] = 1
    # print(indices1)
    # print("Found {} peaks from method one.".format(len(indices1)))

    # Extract center map
    map0 = image[2:-2, 2:-2]

    # Extract shifted maps
    map1  = image[0:-4, 0:-4]
    map2  = image[1:-3, 0:-4]
    map3  = image[2:-2, 0:-4]
    map4  = image[3:-1, 0:-4]
    map5  = image[4:,   0:-4]
    map6  = image[0:-4, 1:-3]
    map7  = image[4:,   1:-3]
    map8  = image[0:-4, 2:-2]
    map9  = image[4:,   2:-2]
    map10 = image[0:-4, 3:-1]
    map11 = image[4:,   3:-1]
    map12 = image[0:-4, 4:  ]
    map13 = image[1:-3, 4:  ]
    map14 = image[2:-2, 4:  ]
    map15 = image[3:-1, 4:  ]
    map16 = image[4:,   4:  ]

    # Compare center map with shifted maps
    merge = ( (map0 > map1) & (map0 > map2) & (map0 > map3) & (map0 > map4)
            & (map0 > map5) & (map0 > map6) & (map0 > map7) & (map0 > map8)
            & (map0 > map9) & (map0 > map10) & (map0 > map11) & (map0 > map12)
            & (map0 > map13) & (map0 > map14) & (map0 > map15) & (map0 > map16)
            )

    bordered = np.lib.pad(merge, (2, 2), 'constant', constant_values=(0, 0))
    peakmap = image * bordered
    y2, x2 = np.nonzero(peakmap >= threshold)

    indices2 = zip(x2, y2)
    # print(indices2)
    peaks = [p for p in indices2 if p in indices1]
    # print("Found {} peaks from method two.".format(len(peaks)))

    return (y2, x2)
