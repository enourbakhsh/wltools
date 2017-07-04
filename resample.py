from __future__ import print_function
import numpy as np
from .utils import get_column_names


def randomize(cat, orientation=True, position=False, verbose=False):
    """
    Generate a noise realization for a galaxy catalogue. The catalogue (cat)
    should contain columns with easily identifiable names for position and
    ellipticity components.

    E.g., Right Ascension will be identified by any of 'x', 'ra', 'ra_gal',
          ignoring case.

    Parameters
    ----------
    cat : FITS_rec
        Galaxy catalogue to randomize.
    orientation : bool
        If true, randomize galaxy position angles.
        (Default is True.)
    position : bool
        If true, randomize galaxy positions within original bounds.
        (Default is False.)
    """
    # Determine (ra, dec, e1, e2) column names of cat
    names = get_column_names(cat)
    if names is None:
        if verbose:
            print("Returning original catalogue.")
        return cat

    x_name, y_name, e1_name, e2_name = names
    if verbose:
        print("Found columns ('{0}', '{1}', '{2}', '{3}')".format(x_name,
              y_name, e1_name, e2_name))

    # Make a copy to modify and return
    new_cat = cat.copy()
    ngal = len(cat)

    # Randomize positions
    if position:
        xmin, xmax = min(cat[x_name]), max(cat[x_name])
        ymin, ymax = min(cat[y_name]), max(cat[y_name])
        new_cat[x_name] = (xmax - xmin) * np.random.random(size=ngal) + xmin
        new_cat[y_name] = (ymax - ymin) * np.random.random(size=ngal) + ymin

        if verbose:
            print("Randomized positions.")

    # Randomize orientations
    if orientation:
        theta = 2 * np.pi * np.random.random(size=ngal)
        emag = np.sqrt(cat[e1_name]**2 + cat[e2_name]**2)
        new_cat[e1_name] = emag * np.cos(theta)
        new_cat[e2_name] = emag * np.sin(theta)

    return new_cat


def bootstrap(x, y, v=None):
    pass
