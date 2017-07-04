#! /usr/bin/env python

from __future__ import print_function
import argparse
import sys
import os
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from wltools.plotting import plot_map

def plot_reconstructions(filenames, flipx=False):
    data = []
    for f in filenames:
        try:
            data.append(fits.getdata(f))
        except IOError:
            print("Cannot open {}".format(f))
            sys.exit()

    fig, axs = plt.subplots(1, len(data), facecolor='w')
    axs = np.atleast_1d(axs)
    # style['vmax'] = max([max(img.flat) for img in data])
    for (ax, img) in zip(axs, data):
        plot_map(img, fig=fig, ax=ax, flipx=flipx, cmap='gist_stern')
        ax.set_title("{0} x {1}".format(img.shape[0], img.shape[1]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot kappa reconstructions.')
    parser.add_argument('files',
                        nargs='+',
                        help='kappa fits file(s).')
    parser.add_argument('-r', '--reverse',
                        action='store_true',
                        default=False,
                        help='reverse x-axis ?')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='verbose ?')
    args = parser.parse_args()
    plot_reconstructions(args.files, flipx=args.reverse)
    sys.exit()
