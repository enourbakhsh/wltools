#! /usr/bin/env python

from __future__ import print_function
import argparse
import sys
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from wltools.plotting import plot_map

def plot(filenames, flipx, cbar, cmap):
    def load_data(f):
        try:
            data = fits.getdata(f)
        except IOError:
            print("Cannot open {}".format(f))
            sys.exit()

        return data

    datalist = [load_data(f) for f in filenames]

    fig, axs = plt.subplots(1, len(datalist), facecolor='w')
    axs = np.atleast_1d(axs)
    # style['vmax'] = max([max(img.flat) for img in data])
    for ax, img in zip(axs, datalist):
        plot_map(img, fig=fig, ax=ax, flipx=flipx, cbar=cbar, cmap=cmap)
        ax.set_title("{0} x {1}".format(img.shape[0], img.shape[1]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot a 2d map.')
    parser.add_argument('files',
                        nargs='+',
                        help='Map fits file(s).')
    parser.add_argument('-r', '--reverse',
                        action='store_true',
                        default=False,
                        help='Reverse x-axis.')
    # parser.add_argument('-v', '--verbose',
    #                     action='store_true',
    #                     default=False,
    #                     help='in')
    parser.add_argument('-c', '--colorbar',
                        action='store_true',
                        default=True,
                        help='Add a color bar.')
    parser.add_argument('-m', '--cmap',
                        type=str,
                        default='gist_stern',
                        help='Color map.')
    args = parser.parse_args()
    plot(args.files, flipx=args.reverse, cbar=args.colorbar, cmap=args.cmap)
    sys.exit()
