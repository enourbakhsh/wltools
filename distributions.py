# Copyright (c) 2017, CosmoStat Laboratory
# Licensed under CeCILL 2.1 - see LICENSE.rst
# Author: Austin Peel
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import gamma
from scipy.stats import laplace, norm
from scipy.integrate import romberg
from common import home
import time

# NOTE This module is basically a copy of what is in ~/Software/pythons/mice
#      It could no doubt use some improvements for generality in wltools.


class zdist(object):
    """
    Redshift distribution class implementing van Waerbeke parameterization.
    n(z) is proportional to (x**alpha) * exp(-(x**beta)) with x = z / z0
    """
    def __init__(self, a, b, z0):
        # MICE: a=0.88, b=1.4, z0=0.78
        self.a = a
        self.b = b
        self.z0 = z0
        self.zdata = None
        self.normalization = self._normalize()

    def _normalize(self):
        if self.zdata is not None:
            zmin = self.zdata.min()
            zmax = self.zdata.max()
            func = (lambda z: ((z / self.z0) ** self.a) *
                    np.exp(-((z / self.z0) ** self.b)))
            norm = 1. / romberg(func, zmin, zmax)
        else:
            norm = self.b / self.z0 / gamma((1. + self.a) / self.b)
        return norm

    def pdf(self, z):
        z = np.atleast_1d(z)
        x = z / self.z0
        return self.normalization * (x ** self.a) * np.exp(-(x ** self.b))

    def cdf(self, z):
        z = np.atleast_1d(z)
        v = [romberg(lambda x: self.pdf(x), 0, z0, divmax=12)[0] for z0 in z]
        return np.array(v)

    def cdfinv(self, p):
        pass

    def rvs(self, N):
        # Generate N random numbers between 0 and 1
        randos = np.random.random(N)
        return self.cdfinv(randos)

    def fit(self, zarray, verbose=True):
        self.zdata = zarray
        n, bins = np.histogram(zarray, bins=512, density=True)
        zvals = (bins[:-1] + bins[1:]) / 2

        fitfunc = (lambda z, A, alpha, beta, z0:
                   A * ((z / z0) ** alpha) * np.exp(-((z / z0) ** beta)))

        params, cov = curve_fit(fitfunc, zvals, n, p0=[1., 0.75, 1., 0.75])
        self.a = params[1]
        self.b = params[2]
        self.z0 = params[3]
        self.normalization = self._normalize()
        if verbose:
            print("a = %.3f, b = %.3f, z0 = %.3f" % (self.a, self.b, self.z0))

    def plot(self, nbins=32):
        fs = 18
        ls = 16
        histstyle = dict(facecolor='white', alpha=1, label='data',
                         histtype='step', linewidth=1.1, color='b')
        pdfstyle = dict(color='k', linewidth=1.2, alpha=1, linestyle='solid')
        fig, ax = plt.subplots(1, 1, facecolor='w', figsize=(7.5, 5.3))

        if self.zdata is not None:
            # Scale z distribution to visually match best fit curve
            n, bins = np.histogram(self.zdata, bins=nbins, normed=False)
            # scaling = romberg(self.pdf, bins[0], bins[-1])[0]
            scaling = 1.
            weight = scaling * len(n) / n.sum() / (bins[-1] - bins[0])
            n, bins, patches = ax.hist(self.zdata, bins=nbins,
                               weights=[weight] * len(self.zdata), **histstyle)
            # zvals = (bins[:-1] + bins[1:]) / 2
            zmax = self.zdata.max() * 1.05
            pdfstyle['label'] = r'$\mathrm{fit}$'
        else:
            zmax = 2
            pdfstyle['label'] = r'$n(z)$ $\mathrm{for}$ $0 < z < 2$'

        z = np.linspace(0, zmax, 128)
        ax.plot(z, self.pdf(z), **pdfstyle)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, zmax)
        ax.set_xlabel(r'$z$', fontsize=fs)
        ax.set_ylabel(r'$n(z)$', fontsize=fs)
        # ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(loc='best', fontsize=ls,
                           handles=reversed(handles), labels=reversed(labels),
                           title=(r'$\mathrm{Redshift\;\,distribution}$'))
        plt.setp(legend.get_title(), fontsize=ls)
        fig.subplots_adjust(bottom=0.12, right=0.96, top=0.95)
        plt.show()


class zmice(object):
    """Get true redshifts from full MICEv2.0 catalog."""
    ztot = 499609997

    def __init__(self):
        zcat = "mice_v2_0_redshift_order_by_unique_gal_id.ssv"
        data_path = home + "/Data/MICE/v2.0/ssv/"
        self.path = data_path + zcat
        self.z = None

    def getz(self, num=ztot):
        zlist = []
        ngal = 0
        counter = 0
        now = time.time()
        with open(self.path) as f:
            f.readline()
            for line in f:
                # Take only every hundredth one
                if (counter % 100 == 0):
                    columns = line.split()
                    zlist.append(float(columns[1]))
                    ngal += 1

                counter += 1
                if counter % 5e7 == 0:
                    print("  %d million: %.1f s" % (counter / 1e6,
                                                  time.time() - now))
                    now = time.time()

                if counter == num:
                    break

        print("Extracted %d redshifts from %d rows." % (ngal, counter))
        self.z = np.array(zlist)

    def sample(self, num):
        samples = []
        N = 0
        for z in self.z:
            N += 1
            if len(samples) < num:
                samples.append(z)
            else:
                s = int(np.random.random() * N)
                if s < num:
                    samples[s] = z

        return samples


class edist(object):
    """Ellipticity distribution class."""
    # Global std. dev. variable : achieves truncated Laplace distribution with sigma = 0.3
    sig = 0.334

    def __init__(self, edata):
        self.e = np.copy(edata)
        self.emod = []
        #self.xbins = []
        #self.ybins = []
        #self.slopes = []

    def transform(self, sigma=sig):
        # Get indices that would sort ellipticities
        inds = np.argsort(self.e)

        # Set scale of Laplace distribution with std sigma
        scl = sigma / np.sqrt(2)

        # Generate len(self.e) random values from truncated Laplace distribution
        samples = laplace.rvs(size=len(self.e), loc=0, scale=scl)
        while (np.abs(samples) > 1).sum() > 0:
            badinds = np.abs(samples) > 1
            newvals = laplace.rvs(size=badinds.sum(), loc=0, scale=scl)
            samples[badinds] = newvals
        samples.sort()

        # Check that statistics are what we want
        print("mean: %.4f, std: %.4f" % (samples.mean(), samples.std()))

        # Replace
        self.emod = np.copy(self.e)
        self.emod[inds] = samples

    def plot(self):
        self.transform()
        plt.clf()
        n, bins, patches = plt.hist(self.emod, 128, normed=True,
                          histtype='stepfilled', facecolor='green', alpha=0.3)
        evals = (bins[:-1] + bins[1:]) / 2
        scl = 0.3 / np.sqrt(2)
        plt.plot(evals, (laplace.pdf(evals, loc=0, scale=scl)), 'r-')
        plt.xlim(-1.05, 1.05)
        return


class ecosmos(object):
    """COSMOS field ellipticity distribution."""
    def __init__(self, cat='new'):

        if not cat in ['old', 'new']:
            print("Invalid catalog. Must be either 'old' or 'new'.")
            return

        if cat is 'old':
            self.path = home + "/Data/cosmos/cosmos_morphology_2005.tbl"
            # Initialize data lists
            a_image = []      # SExtractor semi-major axis measurement
            b_image = []      # SExtractor semi-minor axis measurement
            theta_image = []  # SExtractor position angle measurement
            majoraxis = []    # MORPHIOUS semi-major axis measurement
            minoraxis = []    # MORPHIOUS semi-minor axis measurement
            morph_phi = []    # MORPHIOUS axis ratio determination
            morph_ratio = []  # MORPHIOUS position angle measurement

            with open(self.path) as cosmos:
                for i in range(35):    # Move past 33 header lines
                    cosmos.readline()

                for line in cosmos:
                    columns = line.split()
                    a_image.append(float(columns[2]))
                    b_image.append(float(columns[3]))
                    theta_image.append(float(columns[4]))

            self.a_image = np.array(a_image)
            self.b_image = np.array(b_image)
            self.theta_image = np.array(theta_image)
            q = (self.b_image / self.a_image)
            self.e_mag = (1 - q) / (1 + q)
            self.e1 = (self.e_mag_old *
                           np.cos(2 * np.deg2rad(self.theta_image)))
            self.e2 = (self.e_mag_old *
                           np.sin(2 * np.deg2rad(self.theta_image)))
            self.ngal = len(a_image)
            # self.majoraxis = np.array(majoraxis)
            # self.minoraxis = np.array(minoraxis)
            # self.morph_ratio = np.array(morph_ratio)
        else:
            self.path = home + "/Data/cosmos/cosmos_cat.txt"
            with open(self.path) as cosmos:
                e1_list = []      # e1 from newer cosmos
                e2_list = []      # e2 from newer cosmos
                for line in cosmos:
                    columns = line.split()
                    e1_list.append(float(columns[2]))
                    e2_list.append(float(columns[3]))

            self.e1 = np.array(e1_list)
            self.e2 = np.array(e2_list)
            self.ngal = len(e1_list)

    def show_hist(self, data):
        fig, ax = plt.subplots(1, 1, facecolor='w')
        n, bins, patches = plt.hist(data, 128, facecolor='green',
                                    alpha=0.3, normed=True)
        mu_g, sigma_g = norm.fit(data)
        mu_l, scl_l = laplace.fit(data)
        zspace = np.linspace(data.min(), data.max(), 128)
        ax.plot(zspace, norm.pdf(zspace, mu_g, sigma_g), 'r-',
                label=(r'gauss: $\mu=%.3f, \sigma=%.3f$' %
                (mu_g, sigma_g)))
        ax.plot(zspace, laplace.pdf(zspace, mu_l, scl_l), 'b-',
                 label=(r'laplace: $\mu=%.3f, \sigma=%.3f$' %
                 (mu_l, scl_l * np.sqrt(2))))
        ax.set_title("COSMOS data")
        ax.set_xlabel(r"$\epsilon_1$", fontsize=18)
        ax.set_ylabel("probability", fontsize=18)
        ax.legend(loc='best')
        plt.show()

    def rvs(self, component, size=1):
        if component is 'e1':
            data = self.e1
        elif component is 'e2':
            data = self.e1
        else:
            print("component must be either 'e1' or 'e2'.")
            return []

        n, bins = np.histogram(data, 256, normed=True)
        binsize = bins[1] - bins[0]
        probs = n / n.sum()
        inds = np.digitize(np.random.random(size), probs.cumsum())
        values = np.random.random(size) * binsize + bins[inds]

        return values
