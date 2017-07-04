from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from astropy import cosmology
from astropy import units as u
from ..utils import sigma_crit, m200

class nfw_profile(object):
    def __init__(self, r200, c, z):
        """
        Using flat LCDM model with H0=70 km/s/Mpc and Om0=0.3.

        Parameters
        ----------
        r200 : float, optionally as a compatible astopy.units quantity
            Radius within which the mass density equals 200 * rho_crit. [Mpc]
        c : float
            Concentration parameter. [dimensionless]
        z : float
            Redshift of NFW halo.
        """
        try:
            r200 = r200.to(u.Mpc)
        except AttributeError:
            r200 = float(r200) * u.Mpc
        except u.UnitConversionError:
            print("Error: Invalid unit for r200.")
            return
        c, z = float(c), float(z)
        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        self.delta_c = 200 * c**3 / 3 / (np.log(1 + c) - (c / (1 + c)))
        self.rs = r200 / c
        self.rho_c = cosmo.critical_density(z)
        self.m200 = m200(r200, z, cosmo.H0, cosmo.Om0)
        DA = cosmo.angular_diameter_distance(z)
        self.angular_size = (u.rad * 2 * r200 / DA).to(u.arcmin)

        self.r200 = r200
        self.c = c
        self.z = z
        self.cosmo = cosmo

    def rho(self, R):
        """Compute profile density at R [Mpc]."""
        try:
            R = R.to(u.Mpc)
        except AttributeError:
            R = float(R) * u.Mpc
        except u.UnitConversionError:
            print("Error: Invalid unit for R.")
            return 0

        denom = (R / self.rs) * (1 + R / self.rs)**2
        return self.delta_c * self.rho_c / denom

    def _kfunc(self, theta, z_src):
        """Function to perform the common calculations of kappa and gamma."""
        # Compute Sigma_critical for each lens-galaxy pair
        # NOTE sigma_crit handles avoiding division by zero, source galaxies
        #      in the foreground, etc.
        sig_cr = sigma_crit(self.z, z_src, self.cosmo.H0, self.cosmo.Om0)
        kfunc = (2 * self.rs * self.delta_c * self.rho_c / sig_cr).decompose()
        Dl = self.cosmo.angular_diameter_distance(self.z)
        x = (theta / (self.rs / Dl)).decompose()
        return kfunc, x

    def _verify_theta(self, theta):
        # Check that theta is in units of arcsec, arcmin, or deg.
        theta = np.atleast_1d(theta)
        if type(theta) is u.quantity.Quantity:
            theta = theta.to(u.arcmin)
        else:
            print("Input theta must have units of arcsec, arcmin, or deg.")
            print("Returning 0.")
            return np.zeros_like(theta)

        # Keep only theta > 0
        if len(theta) != sum(theta > 0):
            print("Warning: ignoring theta values <= 0.")
            theta = theta[theta > 0]

        return theta

    def _verify_z_src(self, z_src, N):
        # If z_src is a single float, return np.array([z_src] * N)
        # If z_src is already an array, make it length N if not already by
        #   truncating or repeating the final value
        z_src = np.atleast_1d(z_src)

        # Single value case
        if len(z_src) == 1:
            if N > 1:
                return np.array([z_src[0]] * N)
            else:
                return z_src

        # Multiple values case
        if len(z_src) < N:
            z_src = list(z_src) + [z_src[-1]] * (N - len(z_src))
            z_src = np.array(z_src)
        elif len(z_src) > N:
            z_src = z_src[:N]
        return z_src

    def kappa(self, theta, z_src=1.0, trunc=False):
        """
        Compute convergence profile as a function of angular separation
        from the NFW halo's center.

        Parameters
        ----------
        theta: array_like, in angular astropy.units [arcsec, arcmin, or deg].
            Angular separation value(s) from center. Will be converted to
            arcmin if not already.
        z_src: float or array_like
            Redshift of source galaxies (default 1.0).
        trunc: bool
            Whether to use truncated profile.
        """
        theta = self._verify_theta(theta)
        z_src = self._verify_z_src(z_src, len(theta))

        if theta.value.any():
            kfunc, x = self._kfunc(theta, z_src)
            kappa = (kfunc * self._G_kappa(x.value, trunc)).value
        else:
            kappa = np.zeros_like(theta)

        return kappa

    def gamma(self, theta, z_src=1.0, trunc=False):
        """
        Compute shear magnitude profile as a function of angular separation
        from the NFW halo's center.

        Parameters
        ----------
        theta: array_like, in angular astropy.units [arcsec, arcmin, or deg].
            Angular separation value(s) from center. Will be converted to
            arcmin if not already.
        z_src: float
            Redshift of source galaxies (default 1.0).
        trunc: bool
            Whether to use truncated profile.
        """
        theta = self._verify_theta(theta)
        z_src = self._verify_z_src(z_src, len(theta))

        if theta.value.any():
            kfunc, x = self._kfunc(theta, z_src)
            gamma = (kfunc * self._G_gamma(x.value, trunc)).value
        else:
            gamma = np.zeros_like(theta)

        return gamma

    def _G_kappa(self, x, trunc):
        c = self.c
        G = np.zeros(len(x))
        if trunc:
            # From Takada & Jain (2003)
            case1 = x < 1
            case2 = x == 1
            case3 = (x > 1) & (x <= c)
            case4 = x > c

            # Case 1: x < 1
            y = x[case1]
            G[case1] = (np.sqrt(c**2 - y**2) / ((y**2 - 1.) *
                       (1. + c)) + np.arccosh((y**2 + c) /
                       (y * (1. + c))) / ((1. - y**2)**1.5))
            # Case 2: x == 1
            G[case2] = (np.sqrt(c**2 - 1.) * (2. + c) /
                       (3. * (1. + c)**2))
            # Case 3: 1 < x <= c
            y = x[case3]
            G[case3] = (np.sqrt(c**2 - y**2) / ((y**2 - 1.) *
                       (1. + c)) - np.arccos((y**2 + c) /
                       (y * (1. + c))) / ((y**2 - 1.)**1.5))
            # Case 4: x > c --> G = 0 by default
        else:
            # Eq. (???) of Wright & Brainerd (2000)
            less = x < 1
            more = x > 1
            egal = x == 1

            # Case : x < 1
            y = x[less]
            G[less] = (1. - (2. * np.arctanh(np.sqrt((1. - y) / (1. + y))) /
                       np.sqrt(1. - y**2))) / (y**2 - 1.)
            # Case : x > 1
            y = x[more]
            G[more] = (1. - (2. * np.arctan2(np.sqrt(y - 1.),
                                             np.sqrt(1. + y)) /
                       np.sqrt(y**2 - 1.))) / (y**2 - 1.)
            # Case : x == 1
            G[egal] = 1. / 3

        return G

    def _G_gamma(self, x, trunc):
        c = self.c
        G = np.zeros(len(x))
        if trunc:
            # Eq. (17) of Takada & Jain (2003)
            case1 = x < 1
            case2 = x == 1
            case3 = (x > 1) & (x <= c)
            case4 = x > c

            # Case 1: x < 1
            y = x[case1]
            G[case1] = (((2. - y**2) * np.sqrt(c**2 - y**2) / (1. - y**2) -
                        2. * c) / y**2 / (1. + c) + 2. * np.log(y * (1. + c) /
                        (c + np.sqrt(c**2 - y**2))) / y**2 + (2. - 3. * y**2) *
                        np.arccosh((y**2 + c) / y / (1. + c)) / y**2 /
                        (1. - y**2)**1.5)
            # Case 2: x == 1
            y = x[case2]
            G[case2] = (((11. * c + 10.) * np.sqrt(c**2 - 1.) /
                         (1. + c) - 6. * c) / (3. * (1. + c)) +
                         2. * np.log((1. + c) / (c + np.sqrt(c**2 - 1.)) /
                         y**2))
            # Case 3: 1 < x <= c
            y = x[case3]
            G[case3] = (((2. - y**2) * np.sqrt(c**2 - y**2) / (1. - y**2) -
                        2. * c) / y**2 / (1. + c) + 2. * np.log(y * (1. + c) /
                        (c + np.sqrt(c**2 - y**2))) / y**2 - (2. - 3. * y**2) *
                        np.arccos((y**2 + c) / y / (1. + c)) / y**2 /
                        (y**2 - 1.)**1.5)
            # Case 4: x > c
            y = x[case4]
            G[case4] = 2. * (np.log(1. + c) - c / (1. + c)) / y**2
        else:
            # Eq. (???) of Wright & Brainerd (2000)
            less = x < 1
            more = x > 1
            egal = x == 1

            # Case : x < 1
            y = x[less]
            G[less] = ((4 * np.arctanh(np.sqrt((1 - y) / (1 + y))) /
                       (y**2 * np.sqrt(1 - y**2))) +
                       (2 * np.log(y / 2) / y**2) -
                       (1 / (y**2 - 1)) +
                       (2 * np.arctanh(np.sqrt((1 - y) / (1 + y))) /
                       ((y**2 - 1) * np.sqrt(1 - y**2))))
            # Case : x > 1
            y = x[more]
            G[more] = ((4 * np.arctan2(np.sqrt(y - 1), np.sqrt(1 + y)) /
                       (y**2 * np.sqrt(y**2 - 1))) +
                       (2 * np.log(y / 2) / y**2) -
                       (1 / (y**2 - 1)) +
                       (2 * np.arctan2(np.sqrt(y - 1), np.sqrt(1 + y)) /
                       ((y**2 - 1)**1.5)))
            # Case : x == 1
            G[egal] = 2 * np.log(0.5) + 5. / 3

        return G

    def _plot(self, option):
        fig, ax = plt.subplots(1, 1, facecolor='w')
        theta = np.linspace(0.1, 15, 128) * u.arcmin
        if option == 'kappa':
            ax.plot(theta.value, self.kappa(theta, trunc=False), label='orig.')
            ax.plot(theta.value, self.kappa(theta, trunc=True), label='trunc.')
            ax.set_ylim(0, 0.5)
            ax.set_ylabel(r"$\kappa$")
        elif option == 'gamma':
            ax.plot(theta.value, self.gamma(theta, trunc=False), label='orig.')
            ax.plot(theta.value, self.gamma(theta, trunc=True), label='trunc.')
            ax.set_ylim(0, 0.25)
            ax.set_ylabel(r"$|\gamma|$")
        else:
            print("Invalid selection.")
            return
        ax.set_xlim(0.1, 15)
        ax.set_xlabel(r"$\theta$ [arcmin]")
        plt.legend(loc=0)
        plt.show()

    def plot_kappa_profile(self):
        self._plot('kappa')

    def plot_gamma_profile(self):
        self._plot('gamma')


class einasto_profile(object):
    def __init__(self):
        pass

class sis_profile(object):
    def __init__(self):
        pass
