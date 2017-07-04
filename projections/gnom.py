from __future__ import print_function, division
import numpy as np
import astropy.units as u


def radec2xy(ra0, dec0, ra, dec, degrees=False):
    """
    Gnomonic (tangent) projection of (ra, dec) coordinates to the tangent
    plane about the point (ra0, dec0).

    Parameters
    ----------
    ra0 : float
        RA coordinate of projection origin.
    dec0 : float
        Dec coordinate of projection origin.
    ra : array_like
        RA value(s) to project.
    dec : array_like
        Dec value(s) to project.
    degrees : bool, optional
        If True, RETURN values in degrees about origin at (ra0, dec0).

    NOTE: All inputs can accommodate compatible astropy units (angles).
          If none are given, degrees are assumed.

    Returns
    -------
    x : array_like
        Projected (tangent space) RA value(s) in rad, unless degrees=True.
    y : array_like
        Projected (tangent space) Dec value(s) in rad, unless degrees=True.
    """
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)

    def verify_units(x):
        try:
            x = x.to(u.degree)
        except AttributeError:
            return verify_units(x * u.degree)
        except u.UnitConversionError:
            print("Bad units for {}. Assuming degrees instead.".format(x))
            return verify_units(x.value * u.degree)

        return x

    # Verify inputs
    ra0 = verify_units(ra0)
    dec0 = verify_units(dec0)
    ra = verify_units(ra)
    dec = verify_units(dec)

    # Convert the input coordinates to radians
    alpha0 = ra0.to(u.rad).value
    delta0 = dec0.to(u.rad).value
    alpha = ra.to(u.rad).value
    delta = dec.to(u.rad).value

    # Compute projected values
    denom = (np.cos(delta0) * np.cos(delta) * np.cos(alpha - alpha0) +
             np.sin(delta0) * np.sin(delta))
    x = np.cos(delta) * np.sin(alpha - alpha0) / denom
    y = ((np.cos(delta0) * np.sin(delta) -
          np.sin(delta0) * np.cos(delta) * np.cos(alpha - alpha0)) / denom)

    if degrees:
        return np.rad2deg(x) * u.degree + ra0, np.rad2deg(y) * u.degree + dec0

    return x, y


def xy2radec(ra0, dec0, x, y):
    """
    Inverse Gnomonic (tangent) projection from the tangent plane (x, y)
    coordinates to the sphere about the point (ra0, dec0).

    Parameters
    ----------
    ra0 : float
        RA coordinate of projection origin [degrees].
    dec0 : float
        Dec coordinate of projection origin [degrees].
    x : array_like
        RA value(s) to deproject relative to (0, 0) [radians].
    y : array_like
        Dec value(s) to deproject relative to (0, 0) [radians].
    radians : bool, optional
        If True, RETURN values in degrees about origin at (ra0, dec0).

    Returns
    -------
    ra : array_like
        De-projected RA value(s) on the sphere [degrees].
    dec : array_like
        De-projected DEC value(s) on the sphere [degrees].
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    def verify_degrees(x):
        try:
            x = x.to(u.degree)
        except AttributeError:
            return verify_degrees(x * u.degree)
        except u.UnitConversionError:
            print("Bad units for {}. Assuming degrees instead.".format(x))
            return verify_degrees(x.value * u.degree)
        return x

    def verify_radians(x):
        try:
            x = x.to(u.rad)
        except AttributeError:
            return verify_radians(x * u.rad)
        except u.UnitConversionError:
            print("Bad units for {}. Assuming radians instead.".format(x))
            return verify_radians(x.value * u.rad)
        return x

    # Verify origin coordinates
    ra0 = verify_degrees(ra0)
    dec0 = verify_degrees(dec0)

    # Convert projection center to radians
    # x0 = np.deg2rad(ra0)
    # y0 = np.deg2rad(dec0)
    x0 = ra0.to(u.rad).value
    y0 = dec0.to(u.rad).value

    # Compute tangent space positions relative to projection center
    # x = np.atleast_1d(np.deg2rad(x)) - x0
    # y = np.atleast_1d(np.deg2rad(y)) - y0

    # Verify tangent space coordinates
    x = verify_radians(x).value
    y = verify_radians(y).value

    # Compute deprojected coordinates
    z = np.sqrt(x * x + y * y)
    c = np.arctan(z)

    # Prevent division by zero
    factor = np.ones(len(z))
    inds = (z != 0)
    factor[inds] = y[inds] / z[inds]

    delta = np.arcsin(np.cos(c) * np.sin(y0) + factor * np.cos(y0) * np.sin(c))
    denom = z * np.cos(y0) * np.cos(c) - y * np.sin(y0) * np.sin(c)
    alpha = x0 + np.arctan2(x * np.sin(c), denom)

    # if radians:
    #     return alpha, delta

    # return np.rad2deg(alpha), np.rad2deg(delta)
    return np.rad2deg(alpha) * u.degree, np.rad2deg(delta) * u.degree


def tangent_square(ra0, dec0, size):
    """
    Determine the box in RA/Dec space that gives a square of side length size
    in the tangent space centered at (ra0, dec0).

    Parameters
    ----------
    ra0 : float, optionally with compatible astropy.units
        Right Ascension coordinate of the tangent point.
    dec0 : float, optionally with compatible astropy.units
        Declination coordinate of the tangent point.
    size : float, optionally with compatible astropy.units
        Size of square in the tangent space.

    Note : ra0, dec0 units are assumed to be degrees if not speficied,
           while size units are assumed to be radians.

    Return
    ------
    Extent of the minimum box in RA/Dec space required to achieve the
    tangent space square. Implied units are degrees.
    """

    halfsize = size / 2
    # ramin = max(ra0 - halfsize, 0)
    # ramax = min(ra0 + halfsize, 360)
    # decmin = max(dec0 - halfsize, -90)
    # decmax = min(dec0 + halfsize, 90)
    # top_lft = xy2radec(ra0, dec0, ramin, decmax)
    # top_mid = xy2radec(ra0, dec0, ra0, decmax)
    # top_rgt = xy2radec(ra0, dec0, ramax, decmax)
    # btm_lft = xy2radec(ra0, dec0, ramin, decmin)
    # btm_rgt = xy2radec(ra0, dec0, ramax, decmin)
    # extent = [top_lft[0][0], top_rgt[0][0], btm_lft[1][0], top_mid[1][0]]

    # Deproject the corners of the tangent-space square
    top_left = xy2radec(ra0, dec0, -halfsize, halfsize)
    top_mid = xy2radec(ra0, dec0, 0, halfsize)
    top_right = xy2radec(ra0, dec0, halfsize, halfsize)
    btm_left = xy2radec(ra0, dec0, -halfsize, -halfsize)
    btm_mid = xy2radec(ra0, dec0, 0, -halfsize)
    btm_right = xy2radec(ra0, dec0, halfsize, -halfsize)

    ramin = max(min(top_left[0][0], btm_left[0][0]), 0 * u.degree)
    ramax = min(max(top_right[0][0], btm_right[0][0]), 360 * u.degree)
    decmin = max(min(btm_left[1][0], btm_mid[1][0], btm_right[1][0]),
                     -90 * u.degree)
    decmax = min(max(top_left[1][0], top_mid[1][0], top_right[1][0]),
                     90 * u.degree)

    return [ramin.value, ramax.value, decmin.value, decmax.value] * u.degree
