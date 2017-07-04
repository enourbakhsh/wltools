#! /usr/bin/env python

from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import ConfigParser
from matplotlib import animation
from wltools.plotting import plot_map
from glob import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot a 3d map.')
    parser.add_argument('file',
                        type=str,
                        help='Map fits file(s).')
    parser.add_argument('-a', '--animate',
                        action='store_true',
                        default=False,
                        help='Cycle through redshift slices.')
    parser.add_argument('-s', '--slice',
                        type=int,
                        default=0,
                        help='Redshift slice to show.')
    parser.add_argument('-r', '--reverse',
                        action='store_true',
                        default=False,
                        help='Reverse x-axis.')
    parser.add_argument('-m', '--cmap',
                        type=str,
                        default='gist_stern',
                        help='Color map.')
    args = parser.parse_args()
    anim = args.animate
    idx = args.slice
    flipx = args.reverse
    cmap = args.cmap

    try:
        imgs = fits.getdata(args.file)
    except IOError:
        print("Cannot open {}".format(args.file))
        sys.exit()

    try:
        config = ConfigParser.ConfigParser()
        configpath = glob(os.path.join(os.getcwd(), "config3d*"))[0]
        config.read(configpath)
        ra0 = float(config.get('survey', 'center_ra'))
        dec0 = float(config.get('survey', 'center_dec'))
        # config.get('survey', 'size')
        # config.get('survey', 'units', 'degree')
        # config.get('cosmology', 'Omega_m')
        # config.get('cosmology', 'h')
        pixel_size = float(config.get('field', 'pixel_size'))
        units = config.get('field', 'units')
        padding = int(config.get('field', 'padding'))
        # nlp = config.get('field', 'nlp')
        zmin = float(config.get('field', 'zlp_min'))
        zmax = float(config.get('field', 'zlp_max'))
        print("Loaded {}".format(configpath[len(os.getcwd()) + 1:]))
    except IOError:
        print("No config3d.ini found.")
        config = None

    if anim:
        nz = imgs.shape[0]
        shape = (imgs.shape[1], imgs.shape[2])
        vmax = imgs.max()
        print("Found {} z slices of size : {}".format(nz, shape))
        fig = plt.figure()
        data = imgs[0]
        img = plt.imshow(data, origin='lower', cmap=cmap, vmax=vmax)
        ax = img.axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r'{} $\times$ {}'.format(shape[0], shape[1]))
        plt.colorbar()

        def init():
            img.set_data(imgs[0])

        def animate(i):
            # img.set_data(np.zeros(shape))
            img.set_data(imgs[i])
            ax = img.axes
            if config:
                z_bin_edges = np.linspace(zmin, zmax, nz + 1)
                z_bin_centers = (z_bin_edges[:-1] + z_bin_edges[1:]) / 2
                ax.set_title("z = {:.3f}".format(z_bin_centers[i]))
            else:
                ax.set_title("z slice {}".format(i))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'{} $\times$ {}'.format(shape[0], shape[1]))
            return img

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nz,
                                       interval=800, repeat_delay=1000)
        plt.show()
    else:
        img = imgs[idx]
        fig, ax = plt.subplots(1, 1, facecolor='w')
        plot_map(img, fig=fig, ax=ax, flipx=flipx, cbar=True, cmap=cmap)
        ax.set_title("{0} x {1}".format(img.shape[0], img.shape[1]))
        ax.set_xlabel("z slice {}".format(args.slice))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    sys.exit()
