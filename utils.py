# Miscellaneous functions useful for weak-lensing codes
from __future__ import print_function, division
import numpy as np
import astropy.units as u
from astropy import cosmology
from astropy.constants import G as G_Newt
from astropy.constants import c as c_light


def mad(image):
    """Compute median absolute deviation (MAD) of image."""
    return np.median(abs(image - np.median(image)))


def get_column_names(cat, verbose=False):
    """
    Get position and ellipticity column names from a galaxy catalogue.

    Parameters
    ----------
    cat : FITS_rec
        Galaxy catalogue.

    Returns column names (if found) as strings in the order (RA, DEC, E1, E2).
    Returns None if an error is encountered.
    """

    # Define acceptable column names
    x_names = ['X', 'RA', 'RA_GAL']
    y_names = ['Y', 'DEC', 'DEC_GAL']
    e1_names = ['E1', 'E1_GAL', 'G1', 'G1_GAL', 'GAMMA1', 'GAMMA1_GAL']
    e2_names = ['E2', 'E2_GAL', 'G2', 'G2_GAL', 'GAMMA2', 'GAMMA2_GAL']

    # Find which columns of cat match the possibilities above
    names = cat.names
    x_matches = [col for col in names if col.upper() in x_names]
    y_matches = [col for col in names if col.upper() in y_names]
    e1_matches = [col for col in names if col.upper() in e1_names]
    e2_matches = [col for col in names if col.upper() in e2_names]

    # Only proceed if a unique column name was found for each
    if (len(x_matches) != 1 or len(y_matches) != 1 or
        len(e1_matches) != 1 or len(e2_matches) != 1):
        if verbose:
            print("Could not determine columns.")
        return None

    # Get column names in galaxy catalogue
    x_name = x_matches[0]
    y_name = y_matches[0]
    e1_name = e1_matches[0]
    e2_name = e2_matches[0]

    return x_name, y_name, e1_name, e2_name


def angular_separation(ra1, dec1, ra2, dec2):
    """
    Vectorized computation of angular separation between points on a sphere.

    Parameters
    ----------
    ra(i), dec(i) : array_like
        Location of point (i) on the sphere, where ra is is the
        longitudinal angle, and dec is the latitudinal angle [degrees].

    Returns
    -------
    central_angle : array_like
         Central angle(s) between points (1) and (2) [degrees].
         Note : Distance = R * central_angle, in the same units as R.
    """
    def verify(x):
        try:
            x = x.to(u.rad)
        except AttributeError:
            return verify(x * u.degree)
        except u.UnitConversionError:
            print("Bad units for {}. Assuming degrees instead.".format(x))
            return verify(x.value * u.degree)
        return x

    phi1 = verify(ra1)
    theta1 = verify(dec1)
    phi2 = verify(ra2)
    theta2 = verify(dec2)
    numerator = np.sqrt((np.cos(theta2) * np.sin(phi2 - phi1))**2 +
                        (np.cos(theta1) * np.sin(theta2) -
                         np.sin(theta1) * np.cos(theta2) *
                         np.cos(phi2 - phi1))**2)
    denominator = (np.sin(theta1) * np.sin(theta2) +
                   np.cos(theta1) * np.cos(theta2) * np.cos(phi2 - phi1))
    central_angle = np.arctan2(numerator, denominator)

    return central_angle.to(u.degree)


def gamma_tx(x, y, gamma1, gamma2, center=(0, 0)):
    """
    Compute the tangential and cross components of shear (galaxy ellipticity)
    about the indicated center point.

    The equations are (Schneider 2005)
    gamma_t = -Re[gamma * exp(-2i * phi)],
    gamma_x = -Im[gamma * exp(-2i * phi)],
    where gamma is the complex shear and phi is the polar angle relative
    to the center.

    Parameters
    ----------
    x, y : array_like
        Projected sky positions of points/galaxies.
    gamma1, gamma2 : array_like
        Shear/ellipticity components corresponding to RA/Dec positions.
    center : float array
        Reference position on the sky.

    Returns
    -------
    gamma_t, gamma_x : array_like
        Tangential and cross shear/ellipticity component for each input.
    """
    # Compute distances from center
    delta_x = x - center[0]
    delta_y = y - center[1]
    radii = np.hypot(delta_x, delta_y)

    # Determine polar angles
    phi = np.arctan2(delta_y, delta_x) # range is [-pi, pi]
    angle = 2 * phi

    # Compute tangential shear/ellipticity components
    gamma_t = -gamma1 * np.cos(angle) - gamma2 * np.sin(angle)
    gamma_x = gamma1 * np.sin(angle) - gamma2 * np.cos(angle)
    return gamma_t, gamma_x


def sigma_crit(zl, zs, H0, Om0):
    """
    Compute Sigma_critical in a flat Lambda-CDM universe for each zl-zs pair.

    Parameters
    ----------
    zl : array_like
        Lens redshift(s) [unitless].
    zs : array_like
        Source redshift(s) [unitless].
    Om0 : float
        Present-day matter density parameter of the universe [unitless].
    H0 : float
        Present-day Hubble parameter [km / s / Mpc].

    Returns
    -------
    sigma_crit : array_like
        Value(s) of Sigma_critical as an array of shape (len(zl), len(zs))
        for each lens-source pair.
    """
    zl = np.atleast_1d(zl)
    zs = np.atleast_1d(zs)
    cosmo = cosmology.FlatLambdaCDM(H0=H0, Om0=Om0)
    Dl = [[cosmo.angular_diameter_distance(z).value] * len(zs) for z in zl]
    Ds = [[cosmo.angular_diameter_distance(z).value] * len(zl) for z in zs]
    Dls = [cosmo.angular_diameter_distance_z1z2(z1, z2).value for z1 in zl
            for z2 in zs]
    Dl = np.array(Dl) * u.Mpc
    Ds = np.array(Ds).T * u.Mpc
    Dls = np.array(Dls).reshape(len(zl), len(zs)) * u.Mpc

    # Dermine where Dls <= 0
    bad_inds = (Dls <= 0 * u.Mpc)

    # Compute Sigma_critical for each lens-galaxy pair, avoiding div. by 0
    Dls[bad_inds] = 1 * u.Mpc
    sigma_c = c_light**2 * Ds / (4 * np.pi * G_Newt * Dl * Dls)
    sigma_c = sigma_c.to(u.solMass / u.Mpc**2)
    sigma_c[bad_inds] = 0 * sigma_c.unit

    # Clean up
    if (len(zl) == 1) and (len(zs) == 1):
        sigma_c = sigma_c[0, 0]
    elif len(zl) == 1:
        sigma_c = sigma_c.ravel()
    elif len(zs) == 1:
        sigma_c = sigma_c.T.ravel()

    return sigma_c


def m200(r200, z, H0, Om0):
    """
    Element-wise computation of halo m200 from r200.

    Units
    -----
    r200 : assumed Mpc, otherwise can be given as astropy units
           convertible to Mpc.
    m200 : returned in units of [M_sol].

    If len(r200) != len(z), the returned m200 has length equal to the larger
    of the two. For example,
    compute_m200([0.8, 1.1], 0.4) = compute_m200([0.8, 1.1], [0.4, 0.4]).
    """
    cosmo = cosmology.FlatLambdaCDM(H0=H0, Om0=Om0)
    r200 = np.atleast_1d(r200)
    z = np.atleast_1d(z)

    if len(r200) != len(z):
        # print("Warning: r200 and z are not the same length.")
        if len(r200) > len(z):
            z = np.array(list(z) + [z[-1]] * (len(r200) - len(z)))
        else:
            r200 = np.array(list(r200) + [r200[-1]] * (len(z) - len(r200)))

    try:
        r200.to(u.Mpc)
    except AttributeError:
        r200 = r200 * u.Mpc
    except units.UnitConversionError:
        print("Error: Invalid unit for r200.")
        return

    # Calculate density
    density = cosmo.critical_density0 * cosmo.Om0 / cosmo.Om(z)
    m200 = (4 * np.pi / 3) * 200 * r200**3 * density

    # return (m200 * cosmo.h).to(units.solMass)
    if len(m200) == 1:
        m200 = m200[0]

    return m200.to(u.solMass)


def match_cats(cat1x, cat1y, cat2x, cat2y, radius=None, mode='sphere'):
    """
    Extract matching galaxies between two catalogs based on sky position.
    Match radius variable is assumed to be in the same units as x and y of
    both cats.
    """
    if len(cat1x) != len(cat1y):
        print("Error: cat1 arrays must have the same length.")
        return
    if len(cat2x) != len(cat2y):
        print("Error: cat2 arrays must have the same length.")
        return

    n1 = len(cat1x)
    n2 = len(cat2x)

    def distance(x1, y1, x2, y2):
        x1 = np.atleast_1d(x1)
        y1 = np.atleast_1d(y1)
        x2 = np.atleast_1d(x2)
        y2 = np.atleast_1d(y2)

        if mode == 'plane':
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        elif mode == 'sphere':
            return angular_separation(x1, y1, x2, y2)
        else:
            print("Error: mode must be one of ['plane', 'sphere']")
            return -1

    def find_match(x, y):
        # Get unique closest match in cat1. If distance > radius, return None.
        target_x = np.array([x] * n1)
        target_y = np.array([y] * n1)
        dists = distance(cat1x, cat1y, target_x, target_y)
        idx_nearest = np.argmin(dists)
        if radius is not None:
            if dists[idx_nearest] < radius:
                return idx_nearest
            return
        else:
            return idx_nearest

    cat2 = zip(cat2x, cat2y)
    match_inds = np.array([find_match(x, y) for (x, y) in cat2])
    # print(match_inds)

    # Clean out Nones
    cleaned = np.array([(i, m) for i, m in enumerate(match_inds)
                        if m is not None])
    inds1 = cleaned[:, 1]
    inds2 = cleaned[:, 0]
    # print(inds1)
    # print(inds2)

    # Determine which indices of cat1 are multiple matches for a single index in cat2
    idx_sort = np.argsort(inds1)
    inds1_sorted = inds1[idx_sort]
    vals, idx_start, count = np.unique(inds1_sorted, return_counts=True,
                                       return_index=True)
    result = np.split(idx_sort, idx_start[1:])
    vals = vals[count > 1]
    # print(vals)
    mults = filter(lambda x: x.size > 1, result)
    # print("{} multiple matches.".format(len(mults)))
    # print(mults)

    for i, m in enumerate(mults):
        dists = [distance(cat1x[inds1[j]], cat1y[inds1[j]], cat2x[inds2[j]],
                          cat2y[inds2[j]]) for j in m]
        # print("({})".format(i))
        # print(dists)
        # print(m[np.argmin(dists)])

    def which_to_keep(m):
        dists = [distance(cat1x[inds1[j]], cat1y[inds1[j]], cat2x[inds2[j]],
                          cat2y[inds2[j]]) for j in m]
        return m[np.argmin(dists)]

    dlist = [which_to_keep(m) for m in mults]
    # print(dlist)

    # Pare down multiples to only keep a unique match
    pared = np.array([(ind1, ind2) for ind1, ind2 in zip(inds1, inds2) if
                      ind1 not in vals])
    inds1p = list(pared[:, 0]) + list(vals)
    inds2p = list(pared[:, 1]) + list(inds2[dlist])
    # print(inds1p)
    # print(inds2p)

    return inds1p, inds2p

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1, facecolor='w')
    # ax.scatter(cat1x, cat1y, label='cat1')
    # ax.scatter(cat2x, cat2y, label='cat2')
    # for ind1, ind2 in zip(inds1p, inds2p):
    #     ax.scatter(cat1x[ind1], cat1y[ind1], marker='+', c='c')
    #     ax.scatter(cat2x[ind2], cat2y[ind2], marker='x', c='m')
    # ax.set_aspect('equal')
    # plt.legend()
    # plt.show()


def solidangle(extent):
    """
    Compute the solid angle defined by a rectangle in RA/Dec.

    Parameters
    ----------
    extent: 4-array of type astropy.units.quantity.Quantity
        Field extent of the form [ra_min, ra_max, dec_min, dec_max].

    Returns
    -------
    Solid angle in square degrees.
    """
    if len(np.atleast_1d(extent)) != 4:
        print("ERROR: extent must be of the form " +
              "[ra_min, ra_max, dec_min, dec_max].")
        return

    # Unpack
    ramin, ramax, decmin, decmax = extent

    if ramax <= ramin:
        print("ERROR: ra_max <= ra_min.")
        return
    if decmax <= decmin:
        print("ERROR: dec_max <= dec_min.")
        return

    def convert_to_rad(x):
        try:
            converted = x.to(u.radian)
        except AttributeError:
            print("No units for {}. Assuming degrees.".format(x))
            return convert_to_rad(x * u.degree)
        except u.UnitConversionError:
            print("Bad units for {}. Assuming degrees instead.".format(x))
            return convert_to_rad(x.value * u.degree)
        return converted.value

    # Convert inputs to radians
    alpha0 = convert_to_rad(ramin)
    alpha1 = convert_to_rad(ramax)
    delta0 = convert_to_rad(decmin)
    delta1 = convert_to_rad(decmax)

    # Truncate if values are out of bounds
    # alpha0 = max(0, alpha0)
    # alpha1 = min(2 * np.pi, alpha1)
    # delta0 = max(-np.pi / 2, delta0)
    # delta1 = min(np.pi / 2, delta1)

    # Compute solid angle
    sa = (alpha1 - alpha0) * (np.sin(delta1) - np.sin(delta0)) * u.radian**2

    return sa.to(u.degree**2)


def apply_shear(e1, e2, g1, g2):
    """
    Shear galaxies with intrinsic ellipticity by reduced shear g.

    TODO : Improve this.
    """
    e1 = np.atleast_1d(e1)
    e2 = np.atleast_1d(e2)
    g1 = np.atleast_1d(g1)
    g2 = np.atleast_1d(g2)

    if len(np.unique([len(e1), len(e2), len(g1), len(g2)])) != 1:
        print("Error: arrays must all have the same length.")
        return

    e_int = e1 + 1j * e2
    # Compute observed ellipticities
    g = g1 + 1j * g2
    e_obs = (e_int + g) / (1. + g.conjugate() * e_int)
    # Locate indices where reduced shear exceeds 1
    inds = (g * g.conjugate()).real >= 1
    # Use alternative formula for positions with g > 1
    if sum(inds) > 0:
        numer = 1. + g[inds] * e_int.conjugate()[inds]
        denom = e_int.conjugate()[inds] + g.conjugate()[inds]
        e_obs[inds] = numer / denom
        print("{} galaxies with g >= 1".format(sum(inds)))

    return e_obs.real, e_obs.imag
