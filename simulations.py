from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.io.fits as fits
from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.ndimage.filters import gaussian_filter
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from .halos.profiles import nfw_profile
from .plotting import plot_map, plot_sticks
from .mapping import bin2d
from .utils import angular_separation, gamma_tx, apply_shear
from .stats import mean_radial_profile
from .distributions import zdist
from .projections import gnom
from a520.tests import a520
from a520.footprint import hst_footprint
from a520.utils import approximate_centroids


class source_catalog(object):
    def __init__(self, extent, density, mode='random', zdist=None, zsrc=None):
        # mode is either 'random' or 'grid'
        # NOTE does not deal with projection, and we are working in [arcmin]
        ramin, ramax = extent[:2]
        decmin, decmax = extent[:2]

        # Compute number of galaxies
        self.area = (ramax - ramin) * (decmax - decmin) * u.arcmin**2
        ngal = int(self.area.value * density)

        # Distribute galaxies according to mode option
        if mode == 'random':
            ra = (ramax - ramin) * np.random.random(ngal) + ramin
            dec = (decmax - decmin) * np.random.random(ngal) + decmin
        elif mode == 'grid':
            # ngal1d = int(np.sqrt(self.ngal))
            nx = int(np.sqrt(density) * (ramax - ramin))
            ny = int(np.sqrt(density) * (decmax - decmin))
            x = np.linspace(ramin, ramax, nx)
            y = np.linspace(decmin, decmax, ny)
            xx, yy = np.meshgrid(x, y)
            ra = xx.flatten()
            dec = yy.flatten()

        self.ra = ra
        self.dec = dec
        self.ngal = len(self.ra)

        # TODO fix this to actually sample from different distributions
        if zdist is None:
            self.z = np.ones_like(self.ra)
        elif zdist == 'ludo':
            pdfz = zdist(a=0.88, b=1.4, z0=0.78)
            self.z = np.ones_like(self.ra)
        elif zdist == 'const':
            self.z = np.tile([zsrc], len(self.ra))
        else:
            self.z = np.ones_like(self.ra)

        # self.ngal = len(self.ra)


class nfw_simulation(object):
    def __init__(self, extent=[-5, 5, -5, 5], N=1, x0=None, y0=None,
                 r200=None, c=None, z=None, mode='random', density=30):
        """
        Scatter NFW halos in a field (default 5 x 5 arcmin^2) and compute the
        convergence and shear fields.

        If x0 and y0 are provided, place the N halos at these positions.

        For all parameters, each should be given as an array of size N.
        Exceptions are treated as follows:
        If len(param) < N, the final value is repeated to achieve a size of N.
        If len(param) > N, only the first N values are kept.
        """
        # Create N NFW halos and distribute in the field.
        self.extent = extent # [arcmin]

        def validate(param, pmin=None, pmax=None):
            if param is not None:
                param = np.atleast_1d(param)
                if len(param) < N:
                    param = list(param) + [param[-1]] * (N - len(param))
                    param = np.array(param)
                elif len(param) > N:
                    param = param[:N]
            else:
                param = (pmax - pmin) * np.random.random(N) + pmin

            return param

        # Set halo positions at (x0, y0) if provided, randomly otherwise avoiding edges
        xmin = extent[0] * 0.9
        xmax = extent[1] * 0.9
        self.x0 = validate(x0, xmin, xmax) * u.arcmin
        ymin = extent[2] * 0.9
        ymax = extent[3] * 0.9
        self.y0 = validate(y0, ymin, ymax) * u.arcmin

        # Generate NFW halos
        r200 = validate(r200, 0.5, 1.5) * u.Mpc
        c = validate(c, 2, 6)
        z = validate(z, 0.1, 0.9)
        self.halos = np.array([nfw_profile(rr, cc, zz) for (rr, cc, zz) in
                              zip(r200, c, z)])
        self.cosmo = self.halos[0].cosmo

        # Generate source galaxy catalog
        cat = source_catalog(extent, density=density, mode=mode,
                             zdist='const', zsrc=0.85)
        self.ra_gal = cat.ra
        self.dec_gal = cat.dec
        self.z_gal = cat.z
        print("{} source galaxies".format(cat.ngal))

        # Compute lensing signal

        # Order halos by redshift, decreasing
        inds = np.argsort(z)

        gamma1 = np.zeros(cat.ngal)
        gamma2 = np.zeros(cat.ngal)
        kappa = np.zeros(cat.ngal)
        for x0, y0, halo in zip(self.x0[inds], self.y0[inds], self.halos[inds]):
            dx = self.ra_gal - x0.value
            dy = self.dec_gal - y0.value
            radii = np.sqrt(dx**2 + dy**2) * u.arcmin
            phi = np.arctan2(dy, dx)
            # Compute shear components for this halo
            gamma_mag = halo.gamma(radii, self.z_gal, trunc=False)
            gamma1 += -gamma_mag * np.cos(2 * phi)
            gamma2 += -gamma_mag * np.sin(2 * phi)
            kappa += halo.kappa(radii, self.z_gal, trunc=False)

        jee = a520('jee')
        ngal = len(self.ra_gal)
        e1 = jee.e1_gal.std() * np.random.randn(ngal) + jee.e1_gal.mean()
        e2 = jee.e2_gal.std() * np.random.randn(ngal) + jee.e2_gal.mean()
        g1 = gamma1 / (1 - kappa)
        g2 = gamma2 / (1 - kappa)
        self.e1_gal, self.e2_gal = apply_shear(e1, e2, g1, g2)

        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.kappa = kappa

    def show_kappa(self, npix=64, ker=2, fig=None, ax=None):
        kappamap = bin2d(self.ra_gal, self.dec_gal, v=self.kappa, npix=npix)
        print(kappamap.shape)
        print("max(kappa) : {}".format(kappamap.max()))
        smooth = gaussian_filter(kappamap, ker)
        plot_map(smooth, fig=fig, ax=ax, extent=self.extent, cmap='magma')

    def show_gamma(self, ax=None, color='k', scl=0.1, noisy=False):
        if noisy:
            g1 = self.e1_gal
            g2 = self.e2_gal
        else:
            g1 = self.gamma1 / (1 - self.kappa)
            g2 = self.gamma2 / (1 - self.kappa)
        x = self.ra_gal
        y = self.dec_gal
        plot_sticks(x, y, g1, g2, ax=ax, color=color, scl=scl)
        gamma = self.gamma1 + 1j * self.gamma2
        print("max(gamma) : {}".format(max(abs(gamma))))
        print("max(kappa) : {}".format(max(self.kappa)))
        print("max(g)     : {}".format(max(abs(g1 + 1j * g2))))

    def show_sources(self):
        xmin, xmax = self.extent[:2]
        ymin, ymax = self.extent[2:]
        # Plot source galaxy distribution
        fig, ax = plt.subplots(1, 1)
        ax.scatter(self.ra_gal, self.dec_gal, s=5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        plt.show()

    def show(self, npix=64, scl=0.1):
        fig, ax = plt.subplots(1, 1, facecolor='w')
        self.show_kappa(npix, fig=fig, ax=ax)
        self.show_gamma(ax=ax, color='w', scl=scl)
        ax.set_xlabel('RA [arcmin]')
        ax.set_ylabel('Dec [arcmin]')
        plt.show()

    def write_cat(self, outfile, noisy=False):
        # # Field limits
        xmin, xmax = self.extent[0], self.extent[1]
        ymin, ymax = self.extent[2], self.extent[3]

        if noisy:
            # Use galaxy ellipticities with dists. matching Jee A520 data
            g1 = self.e1_gal
            g2 = self.e2_gal
        else:
            # Compute true reduced shear
            g1 = self.gamma1 / (1 - self.kappa)
            g2 = self.gamma2 / (1 - self.kappa)
            # g1 = self.gamma1
            # g2 = self.gamma2

        ngal = len(self.ra_gal)
        weight = [1] * ngal
        # Write out
        try:
            RA = fits.Column(name='ra', format='D', array=self.ra_gal)
            DEC = fits.Column(name='dec', format='D', array=self.dec_gal)
            Z = fits.Column(name='z', format='D', array=self.z_gal)
            ZSIG = fits.Column(name='z_sig', format='D', array=([0.2] * ngal))
            # E1 = fits.Column(name='e1', format='D', array=self.gamma1)
            # E2 = fits.Column(name='e2', format='D', array=self.gamma2)
            E1 = fits.Column(name='e1', format='D', array=g1)
            E2 = fits.Column(name='e2', format='D', array=g2)
            W = fits.Column(name='w', format='D', array=weight)
            columns = [RA, DEC, Z, ZSIG, E1, E2, W]
            tbhdu = fits.BinTableHDU.from_columns(columns)
            tbhdu.writeto(outfile, overwrite=True)
            print("Wrote shear catalogue to {}".format(outfile))
        except IOError:
            print("Error: Could not write to file.")

    def gamma_t(self, center=(0, 0), bin_edges=None, nbins=None):
        gamma_t, gamma_x = gamma_tx(self.ra_gal, self.dec_gal, self.gamma1,
                                    self.gamma2, center=center)
        bins, mean_t, err_t = mean_radial_profile(self.ra_gal, self.dec_gal,
                              gamma_t, bin_edges=bin_edges, nbins=nbins)
        bins, mean_x, err_x = mean_radial_profile(self.ra_gal, self.dec_gal,
                              gamma_x, bin_edges=bin_edges, nbins=nbins)

        fig, ax = plt.subplots(1, 1)
        ax.plot(bins, mean_t, label=r"$\gamma_t$")
        ax.plot(bins, mean_x, label=r"$\gamma_\times$")
        ax.set_xlabel(r"$\theta$ [arcmin]")
        ax.set_ylabel(r"$\langle\gamma_t\rangle$")
        ax.legend()
        plt.show()


class a520_simulation(object):
    def __init__(self):
        cat = a520('jee')
        # HST footprint
        footprint = hst_footprint()
        ra, dec = footprint.exterior.xy # [degrees]
        x, y = gnom.radec2xy(cat.ra0, cat.dec0, ra, dec) * u.rad
        # Projected footprint
        ftp = Polygon(zip(x.to(u.arcmin).value, y.to(u.arcmin).value))
        area = (footprint.area * u.deg**2).to(u.arcmin**2)
        # Source galaxies
        x_gal = (cat.x_gal * u.rad).to(u.arcmin)
        y_gal = (cat.y_gal * u.rad).to(u.arcmin)

        self.ra0 = cat.ra0
        self.dec0 = cat.dec0
        self.x_gal = x_gal
        self.y_gal = y_gal
        self.z_gal = np.tile(0.85, len(self.x_gal))
        self.e1_gal = cat.e1_gal
        self.e2_gal = cat.e2_gal
        self.footprint = ftp
        self.area = area

    def plot(self, option='sources'):
        fig, ax = plt.subplots(1, 1)
        if option == 'sources':
            ax.scatter(self.x_gal, self.y_gal, s=3, c='k')
        elif option == 'shear':
            plot_sticks(self.x_gal, self.y_gal, self.e1_gal, self.e2_gal,
                        scl=0.03, ax=ax)

        peaks = approximate_centroids('jee')
        p4 = peaks['P4']
        p4x, p4y = gnom.radec2xy(self.ra0, self.dec0, p4[0], p4[1]) * u.rad
        p4x = p4x.to(u.arcmin)
        p4y = p4y.to(u.arcmin)
        ax.scatter(p4x, p4y, marker='x', c='r')
        x, y = self.footprint.exterior.xy
        ax.plot(x, y)
        ax.set_aspect('equal')
        # ax.set_xlim(ax.get_xlim()[::-1])
        plt.show()

    def shuffle(self, pos=True, spin=True):
        # Randomize galaxy positions
        if pos:
            extent = [-4, 4, -4, 4] * u.arcmin
            xmin, xmax, ymin, ymax = extent
            box_area = (xmax - xmin) * (ymax - ymin)
            ngood = 0
            factor = 1.1
            n = len(self.x_gal)
            while ngood < n:
                # Generate (factor * n) > n points within extent box
                N = int(factor * n * box_area / self.area)
                x = (xmax - xmin) * np.random.random(N) + xmin
                y = (ymax - ymin) * np.random.random(N) + ymin
                # Keep the first n that all lie inside geometry area
                points = [Point(xx, yy) for (xx, yy) in zip(x.value, y.value)]
                fpt = self.footprint
                inside = np.array([fpt.contains(pt) for pt in points])
                ngood = sum(inside)
                # Increase factor and try again if ngood < n
                factor += 0.1
            self.x_gal = x[inside][:n]
            self.y_gal = y[inside][:n]

        # Rotate galaxy orientations
        if spin:
            theta = 2 * np.pi * np.random.random(len(self.e1_gal))
            e1_new = np.cos(theta) * self.e1_gal - np.sin(theta) * self.e2_gal
            e2_new = np.sin(theta) * self.e1_gal + np.cos(theta) * self.e2_gal
            self.e1_gal = e1_new
            self.e2_gal = e2_new

    def reset(self):
        cat = a520('jee')
        x_gal = (cat.x_gal * u.rad).to(u.arcmin)
        y_gal = (cat.y_gal * u.rad).to(u.arcmin)
        self.x_gal = x_gal
        self.y_gal = y_gal
        self.e1_gal = cat.e1_gal
        self.e2_gal = cat.e2_gal
        try:
            delattr(self, 'e1_sheared')
            delattr(self, 'e2_sheared')
            delattr(self, 'gamma1')
            delattr(self, 'gamma2')
            delattr(self, 'kappa')
        except:
            pass

    def inject_halo(self, x0=-1.4, y0=-2.5, r200=0.95, c=3.5, z=0.2, show=False):
        r200 = r200 * u.Mpc
        x0 = x0 * u.arcmin
        y0 = y0 * u.arcmin
        # Generate NFW halo
        halo = nfw_profile(r200, c, z)
        # Compute distances from the halo to each source galaxy
        dx = self.x_gal.value - x0.value
        dy = self.y_gal.value - y0.value
        radii = np.hypot(dx, dy) * u.arcmin
        phi = np.arctan2(dy, dx)
        # Compute shear components
        gamma_mag = halo.gamma(radii, self.z_gal, trunc=False)
        gamma1 = -gamma_mag * np.cos(2 * phi)
        gamma2 = -gamma_mag * np.sin(2 * phi)
        kappa = halo.kappa(radii, self.z_gal, trunc=False)
        # Apply halo shear to source galaxies
        g1 = gamma1 / (1 - kappa)
        g2 = gamma2 / (1 - kappa)
        e1, e2 = apply_shear(self.e1_gal, self.e2_gal, g1, g2)
        self.e1_sheared = e1
        self.e2_sheared = e2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.kappa = kappa

        if show:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            plot_sticks(self.x_gal, self.y_gal, gamma1, gamma2, ax=ax1, scl=0.2)
            plot_sticks(self.x_gal, self.y_gal, e1, e2, ax=ax2, scl=0.05)
            x, y = self.footprint.exterior.xy
            ax1.plot(x, y)
            ax1.set_aspect('equal')
            ax2.plot(x, y)
            ax2.set_aspect('equal')
            plt.show()

    def write(self, filepath=None, clean=False):
        if filepath == None:
            filepath = '/Users/apeel/Dropbox/Data/a520sims/sim.fits'
        print("Writing {}".format(filepath))
        ngal = len(self.x_gal)
        RA = fits.Column(name='ra', format='D', array=self.x_gal)
        DEC = fits.Column(name='dec', format='D', array=self.y_gal)
        Z = fits.Column(name='z', format='D', array=self.z_gal)
        ZSIG = fits.Column(name='z_sig', format='D', array=np.tile(0.2, ngal))
        if clean:
            e1 = self.gamma1 / (1 - self.kappa)
            e2 = self.gamma2 / (1 - self.kappa)
        else:
            try:
                e1 = self.e1_sheared
                e2 = self.e2_sheared
            except AttributeError:
                e1 = self.e1_gal
                e2 = self.e2_gal
        E1 = fits.Column(name='e1', format='D', array=e1)
        E2 = fits.Column(name='e2', format='D', array=e2)
        W = fits.Column(name='w', format='D', array=np.tile(1, ngal))
        cols = [RA, DEC, Z, ZSIG, E1, E2, W]
        tbhdu = fits.BinTableHDU.from_columns(cols)
        tbhdu.writeto(filepath, overwrite=True)

    def generate(self, N=1):
        import os
        cwd = os.getcwd()
        path = os.path.join(cwd, 'nfwsims')
        if not os.path.exists(path):
            os.mkdir(path)

        # datapath = '/Users/apeel/Dropbox/Data/a520sims/'
        simpaths = [os.path.join(path, 'sim_{}.fits'.format(i)) for i in range(N)]
        for sp in simpaths:
            self.reset()
            self.shuffle()
            self.inject_halo(show=False)
            self.write(sp)


# NOTE This is not general: only for testing with A520
# class noise_simulation(object):
#     def __init__(self, extent=[-5, 5, -5, 5], density=30, like=None):
#         """
#         Generate a noise simulation of A520 with galaxies randomly distributed
#         within extent and according to the specified number density.
#
#         NOTE: extent values and density are assumed in terms of arcmin.
#         """
#         self.like = like
#         if self.like == 'clowe_hst':
#             # Load catalogue
#             cat = a520('clowe_hst')
#             self.ngal = cat.ngal
#
#             # Compute and sample from ellipticity distributions
#             self.e1dist = gaussian_kde(cat.data['e1'], 'silverman')
#             self.e2dist = gaussian_kde(cat.data['e2'], 'silverman')
#             self.e1 = self.e1dist.resample(self.ngal)[0]
#             self.e2 = self.e2dist.resample(self.ngal)[0]
#
#             # Define sky area in which to distribute galaxies
#             self.polygon = Polygon(cat.vertices)
#             area = self.polygon.area * 3600. # [arcmin^(-2)]
#             ra0 = (max(cat.data['ra']) + min(cat.data['ra'])) / 2.
#             dec0 = (max(cat.data['dec']) + min(cat.data['dec'])) / 2.
#             radius = max(max(cat.data['ra']) - min(cat.data['ra']),
#                          max(cat.data['dec']) - min(cat.data['dec'])) / 2.
#             self.extent = [ra0 - radius, ra0 + radius,
#                            dec0 - radius, dec0 + radius]
#             density = self.ngal / float(area)
#             self.cat = cat
#         elif self.like == 'clowe_magellan':
#             # Load catalogue
#             cat = a520('clowe_magellan')
#             self.ngal = cat.ngal
#
#             # Compute and sample from ellipticity distributions
#             self.e1dist = gaussian_kde(cat.data['e1'], 'silverman')
#             self.e2dist = gaussian_kde(cat.data['e2'], 'silverman')
#             self.e1 = self.e1dist.resample(self.ngal)[0]
#             self.e2 = self.e2dist.resample(self.ngal)[0]
#
#             # Magellan circle
#             ra0, dec0 = (73.54153, 2.92231) # [degrees]
#             radius = 0.2781 / 2
#             self.polygon = Point([ra0, dec0]).buffer(radius)
#             area = self.polygon.area * 3600. # [arcmin^(-2)]
#             self.extent = [ra0 - radius, ra0 + radius,
#                            dec0 - radius, dec0 + radius]
#             density = self.ngal / float(area)
#             self.cat = cat
#         else:
#             self.polygon = None
#             self.extent = extent
#             ramin, ramax = self.extent[:2]
#             decmin, decmax = self.extent[2:]
#             area = (ramax - ramin) * (decmax - decmin)
#             self.ngal = int(density * area)
#             sigma = 0.3
#             tol = 1.0
#             e1 = norm.rvs(loc=0, scale=sigma, size=self.ngal)
#             e2 = norm.rvs(loc=0, scale=sigma, size=self.ngal)
#             badinds = e1**2 + e2**2 > tol
#             while sum(badinds) > 0:
#                 nbad = sum(badinds)
#                 e1[badinds] = norm.rvs(loc=0, scale=sigma, size=nbad)
#                 e2[badinds] = norm.rvs(loc=0, scale=sigma, size=nbad)
#                 badinds = e1**2 + e2**2 > tol
#             self.e1 = e1
#             self.e2 = e2
#
#         # Set source redshift plane
#         z_src = 1.0
#
#         # Generate ngal random positions within appropriate boundary
#         xmin, xmax, ymin, ymax = self.extent
#         if self.polygon is not None:
#             A_box = (xmax - xmin) * (ymax - ymin)
#             N = int(1.25 * self.ngal * A_box / self.polygon.area)
#             print("Generating {} galaxies.".format(N))
#             ra = (xmax - xmin) * np.random.random(N) + xmin
#             dec = (ymax - ymin) * np.random.random(N) + ymin
#             points = [Point(x, y) for (x, y) in zip(ra, dec)]
#             inside = np.array([self.polygon.contains(pt) for pt in points])
#             outside = np.logical_not(inside)
#             print("{} / {}".format(sum(inside), N))
#             # Keep only ngal galaxies generated inside
#             ra = ra[inside][:self.ngal]
#             dec = dec[inside][:self.ngal]
#         else:
#             ra = (xmax - xmin) * np.random.random(self.ngal) + xmin
#             dec = (ymax - ymin) * np.random.random(self.ngal) + ymin
#         self.ra = ra
#         self.dec = dec
#         self.z = [z_src] * self.ngal
#         self.z_sig = [0.1] * self.ngal
#         self.w = [1.0] * self.ngal
#
#         print("Area : {0:.3f} arcmin^2".format(area))
#         print("density : {0:.3f} / arcmin^2".format(density))
#         print("ngal : {}".format(self.ngal))
#         print("Max |e| = {0:.3f}".format(max(self.e1**2 + self.e2**2)))
#
#     # def refresh(self, verbose=False):
#     #     # Re-randomize positions
#     #     ramin, ramax = self.extent[:2]
#     #     decmin, decmax = self.extent[:2]
#     #     self.ra = (ramax - ramin) * np.random.random(self.ngal) + ramin
#     #     self.dec = (decmax - decmin) * np.random.random(self.ngal) + decmin
#     #
#     #     # Re-randomize ellipticities
#     #     if self.like == 'clowe_hst':
#     #         self.e1 = self.e1dist.resample(self.ngal)[0]
#     #         self.e2 = self.e2dist.resample(self.ngal)[0]
#     #     else:
#     #         sigma = 0.3
#     #         tol = 1.0
#     #         e1 = norm.rvs(loc=0, scale=sigma, size=self.ngal)
#     #         e2 = norm.rvs(loc=0, scale=sigma, size=self.ngal)
#     #         badinds = e1**2 + e2**2 > tol
#     #         while sum(badinds) > 0:
#     #             nbad = sum(badinds)
#     #             e1[badinds] = norm.rvs(loc=0, scale=sigma, size=nbad)
#     #             e2[badinds] = norm.rvs(loc=0, scale=sigma, size=nbad)
#     #             badinds = e1**2 + e2**2 > tol
#     #         self.e1 = e1
#     #         self.e2 = e2
#     #
#     #     if verbose:
#     #         print("ngal : {}".format(self.ngal))
#     #         print("Max |e| = {}".format(max(self.e1**2 + self.e2**2)))
#
#     def show(self):
#         fig, ax = plt.subplots(1, 1, facecolor='w')
#         # Plot boundary
#         points = [Point(x, y) for (x, y) in zip(self.ra, self.dec)]
#         inside = np.array([self.polygon.contains(pt) for pt in points])
#         outside = np.logical_not(inside)
#         if self.like.endswith('hst'):
#             ax.plot(self.cat.vertices[:, 0], self.cat.vertices[:, 1], 'b')
#         else:
#             xc, yc = self.polygon.exterior.xy
#             ax.plot(xc, yc, 'b')
#         # Plot points
#         ax.scatter(self.ra[inside], self.dec[inside], color='k', s=1)
#         ax.scatter(self.ra[outside], self.dec[outside], color='r', s=1)
#         ax.set_xlim(self.extent[:2])
#         ax.set_ylim(self.extent[2:])
#         ax.set_xlim(ax.get_xlim()[::-1])
#         ax.set_aspect('equal')
#         plt.show()
#
#     def show_gamma(self):
#         fig, ax = plt.subplots(1, 1, facecolor='w')
#         plot_sticks(self.ra, self.dec, self.e1, self.e2, 0.04, ax=ax)
#         ax.set_xlim(ax.get_xlim()[::-1])
#         ax.set_aspect('equal')
#         ax.grid(True)
#         plt.show()
#
#     def write_to_fits(self, outfile):
#         try:
#             RA = fits.Column(name='ra', format='D', array=self.ra)
#             DEC = fits.Column(name='dec', format='D', array=self.dec)
#             Z = fits.Column(name='z', format='D', array=self.z)
#             ZSIG = fits.Column(name='z_sig', format='D', array=([0.2] * self.ngal))
#             E1 = fits.Column(name='e1', format='D', array=self.e1)
#             E2 = fits.Column(name='e2', format='D', array=self.e2)
#             W = fits.Column(name='w', format='D', array=self.w)
#             columns = [RA, DEC, Z, ZSIG, E1, E2, W]
#             tbhdu = fits.BinTableHDU.from_columns(columns)
#             tbhdu.writeto(outfile, clobber=True)
#             print("Wrote noise catalogue to {}".format(outfile))
#         except IOError:
#             print("Error: Could not write to file.")
