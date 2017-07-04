import time
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from subprocess import call
from .mapping import bin2d
from .mapping import mr_transform


class KStest(object):
    def __init__(self, npix=600, tile=144):
        """Load the positions, shears, and convergence from a MICE tile"""
        start = time.time()
        tilespath = '/Users/apeel/Data/MICE/v2.0/tiles'
        tilepath = tilespath + '/mice_tile_' + str(tile) + '.fits'
        factor = -1
        hdulist = fits.open(tilepath)
        cat = hdulist[1].data
        self.ra = cat['ra_gal']
        self.dec = cat['dec_gal']
        # self.e1 = factor * cat['e1_gal']
        # self.e2 = factor * cat['e2_gal']
        self.g1 = factor * cat['gamma1']
        self.g2 = factor * cat['gamma2']
        self.kappa = cat['kappa']
        self.ngal = len(cat)
        self.npix = [npix, npix]
        self.galmap = bin2d(x=self.ra, y=self.dec, npix=self.npix)
        # v = [self.kappa, self.g1, self.g2, self.e1, self.e2]
        v = [self.kappa, self.g1, self.g2]
        maps = bin2d(x=self.ra, y=self.dec, v=v, npix=self.npix)
        self.kappamap = maps[0]
        self.g1map = maps[1]
        self.g2map = maps[2]
        # self.e1map = maps[3]
        # self.e2map = maps[4]

        # nzero = sum(self.galmap.flatten() == 0)
        # if nzero > 0:
        #     print("{} zero pixels".format(nzero))

        # Do Kaiser-Squires inversion
        self.KSgE, self.KSgB = self._compute_KS('g')
        # self.KSeE, self.KSeB = self._compute_KS('e')

        # print("%.3f seconds" % (time.time() - start))
        self.percent_error_decrease = (1 - self.rms(16) / self.rms(0)) * 100
        # print(tile, self.percent_error_decrease)

        # Do multiresolution transform
        self.kappamr = mr_transform(self.kappamap, 5, True)

    def show_scale(self, scl):
        fig, ax = plt.subplots(1, 1, facecolor='w')
        style = dict(origin='lower', interpolation='nearest', cmap='magma')
        img = ax.imshow(self.kappamr[scl], **style)
        fig.colorbar(img)
        plt.show()

    def show_mr(self, save=False):
        fig, axs = plt.subplots(1, 5, facecolor='w', figsize=(15, 3.5))
        style = dict(origin='lower', interpolation='nearest', cmap='bone')
        props = dict(boxstyle='round', fc='w', ec='k', pad=0.5, alpha=0.85)
        Xmin, Xmax = 300, 450
        Ymin, Ymax = 290, 440
        vmin = min(self.kappamr[3][Xmin: Xmax, Ymin: Ymax].flatten())
        vmax = max(self.kappamr[3][Xmin: Xmax, Ymin: Ymax].flatten())
        # style['vmin'] = -0.03
        # style['vmax'] = 0.1
        axs[0].imshow(self.kappamap[Xmin: Xmax, Ymin: Ymax], **style)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].text(15, 122, r"original $\kappa$", backgroundcolor='w',
                bbox=props, fontsize=16)
        style['vmin'] = vmin
        style['vmax'] = 0.019
        print(vmin, vmax)
        for scl, ax in enumerate(axs[1:]):
            img = ax.imshow(self.kappamr[scl][Xmin: Xmax, Ymin: Ymax], **style)
            # ax.set_title("Scale {}".format(scl + 1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(15, 124, "scale {} ".format(scl + 1), backgroundcolor='w',
                    bbox=props, fontsize=16)
            # if scl == 4:
            #     div = make_axes_locatable(ax)
            #     cax = div.append_axes("bottom", size="5%", pad=0.05)
            #     plt.colorbar(img, cax=cax)
        fig.subplots_adjust(left=0.01, right=0.99, wspace=0.05)
        plt.show()
        if save:
            path = '/Users/apeel/Dropbox/MICE_peak_counts/Resubmission/figures/'
            plt.savefig(path + 'kappa_mr.pdf', format='pdf')

    def show_galmap(self):
        fig, ax = plt.subplots(1, 1, facecolor='w')
        style = dict(origin='lower', interpolation='nearest', cmap='magma')
        img = ax.imshow(self.galmap, **style)
        fig.colorbar(img)
        plt.show()

    def show_kappa(self):
        fig, ax = plt.subplots(1, 1, facecolor='w')
        style = dict(origin='lower', interpolation='nearest', cmap='magma')
        img = ax.imshow(self.kappamap, **style)
        fig.colorbar(img)
        plt.show()

    def _compute_KS(self, option):
        k1, k2 = np.meshgrid(np.fft.fftfreq(self.npix[0]),
                             np.fft.fftfreq(self.npix[1]))
        if option == 'g':
            g1 = np.fft.fft2(self.g1map)
            g2 = np.fft.fft2(self.g2map)
        elif option == 'e':
            g1 = np.fft.fft2(self.e1map)
            g2 = np.fft.fft2(self.e2map)
        else:
            print("Invalid option.")
            return

        numer = ((k1 * k1 - k2 * k2) - 2j * (k1 * k2)) * (g1 + 1j * g2)
        denom = k1 * k1 + k2 * k2
        denom[0, 0] = 1  # avoid division by 0
        kappa = numer / denom
        kappaE = np.real(np.fft.ifft2(kappa))
        kappaB = np.imag(np.fft.ifft2(kappa))
        return kappaE, kappaB

    def show_KS(self, option='g'):
        if option == 'g':
            kappaE = self.KSgE
            kappaB = self.KSgB
        elif option == 'e':
            kappaE = self.KSeE
            kappaB = self.KSeB
        else:
            print("Invalid option.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, facecolor='w', figsize=(10, 4))
        style = dict(origin='lower', interpolation='nearest', cmap='magma')
        img1 = ax1.imshow(kappaE, **style)
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img1, cax=cax1)

        img2 = ax2.imshow(kappaB, **style)
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img2, cax=cax2)

        plt.show()

    def rms(self, border=0):
        # Truth is self.kappamap
        map1 = self.kappamap[border : self.npix[0] - border,
                             border : self.npix[1] - border]
        map2 = self.KSgE[border : self.npix[0] - border,
                         border : self.npix[1] - border]
        diff = (map1 - map2).flatten()
        rms = np.sqrt(sum(diff**2) / len(diff))
        return rms

    def rmsplot(self, bmax=10):
        rms0 = self.rms(0)
        percent = np.zeros(bmax)
        for b in range(1, bmax):
            percent[b] = (1 - self.rms(b) / rms0) * 100
        fig, ax = plt.subplots(1, 1)
        ax.plot(range(bmax), percent)
        plt.show()
