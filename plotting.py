import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .stats import power_spectrum


def plot_map(array, fig=None, ax=None, cbar=True, pos='right', flipx=False,
             **kwargs):
    # Set default style
    style = dict(origin='lower', interpolation='nearest', cmap='gist_stern')
    # Incorporate keyword args into style
    for (k, v) in kwargs.items():
        style[k] = v
    # Plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, facecolor='w')
        img = ax.imshow(array, **style)
        if cbar:
            # print("cbar position: {}".format(pos))
            div = make_axes_locatable(ax)
            cax = div.append_axes(position=pos, size='5%', pad=0.05)
            cbar = fig.colorbar(img, cax=cax)
            if pos == 'left':
                cbar.ax.yaxis.set_ticks_position('left')
        if flipx:
            ax.set_xlim(ax.get_xlim()[::-1])
        # ax.grid(True)
        ax.set_aspect('equal')
        plt.show()
    else:
        img = ax.imshow(array, **style)
        if cbar:
            div = make_axes_locatable(ax)
            if pos is None or pos == 'right':
                cax = div.append_axes(position='right', size='5%', pad=0.05)
                cbar = fig.colorbar(img, cax=cax)
            else:
                cax = div.append_axes(position='left', size='5%', pad=0.05)
                cbar = fig.colorbar(img, cax=cax)
                cbar.ax.yaxis.set_ticks_position('left')
        if flipx:
            ax.set_xlim(ax.get_xlim()[::-1])
            

def plot_sticks(x, y, e1, e2, scl=0.1, color='k', ax=None, extent=None):
    """Plot galaxy shear map."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    e1 = np.atleast_1d(e1)
    e2 = np.atleast_1d(e2)

    if extent is None:
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
    else:
        xmin, xmax = extent[0], extent[1]
        ymin, ymax = extent[2], extent[3]

    if len(np.unique([len(x), len(y), len(e1), len(e2)])) != 1:
        raise Exception("Input lengths are not the same.")

    # Compute U, V
    length = np.sqrt(e1**2 + e2**2)
    theta = np.arctan2(e2, e1) / 2.
    u = length * np.cos(theta)
    v = length * np.sin(theta)

    kw = dict(angles='uv', scale_units='width', scale=1./scl, pivot='mid',
              color=color, headwidth=1, headlength=0.01, headaxislength=1)
    if ax is None:
        fig, ax = plt.subplots(1, 1, facecolor='w')
        # ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, **kw)
        ax.quiver(x, y, u, v, **kw)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.show()
    else:
        ax.quiver(x, y, u, v, **kw)


def plot_contours(Z, levels, x_prior, y_prior, ax=None, legend=True, **kwargs):
    "Plot 2D contours of field Z corresponding to indicated levels."
    # Set default style
    N = len(levels)
    style = dict(linewidths=([1] * N))
    # Incorporate keyword args into style
    for (k, v) in kwargs.items():
        style[k] = v

    extent = [x_prior[0], x_prior[1], y_prior[0], y_prior[1]]

    if ax is None:
        fig, ax = plt.subplots(1, 1, facecolor='w')

    # ls = ['solid', 'dashed'][outline]
    # cs = ax.contourf(Z.T, extent=extent, levels=(levels + [1e7]), alpha=1.0)
    cs = ax.contour(Z.T, levels=levels, extent=extent, **style)

    ax.set_xlim(x_prior)
    ax.set_ylim(y_prior)
    ax.set_aspect('equal')
    # ax.grid(True)
    if legend:
        artists, labels = cs.legend_elements()
        ax.legend(artists[::-1],
                  [r'${}\sigma$'.format(i + 1) for i in range(len(artists))],
                  loc=0)


def plot_power_spectrum(image, size):
    """size should be given in degrees"""
    k, power = power_spectrum(image)
    print(power)
    npix = max(image.shape)
    l = 2 * np.pi * k * max(image.shape) / np.deg2rad(size)
    cl = power * (np.deg2rad(size) / npix)**2
    fig, ax = plt.subplots(1, 1, facecolor='w')
    ax.loglog(l, (l * (l + 1)) / (2 * np.pi) * cl)
    ax.set_xlabel("$l$")
    ax.set_ylabel("$l(l+1) C_l / 2\pi$")
    plt.show()


def plot_linear_transform():
    npix = (51, 51)
    x = np.linspace(-3, 3, npix[0])
    y = np.linspace(-3, 3, npix[1])
    x0 = np.array([0] * len(x))
    y0 = np.array([0] * len(y))
    x1 = np.array([0.5] * len(x))
    y1 = np.array([0.5] * len(y))
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')

    style1 = dict(s=5, c='blue', alpha=0.2)
    style2 = dict(s=5, c='gray', alpha=0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, facecolor='w', figsize=((10, 5)))
    ax1.scatter(xv, yv, **style1)
    ax1.plot(x, y0, c='b', alpha=0.5)
    ax1.plot(x0, y, c='r', alpha=0.5)
    ax1.plot(x1, y, c='m', alpha=0.5)
    ax1.plot(x, y1, c='g', alpha=0.5)
    ax1.grid(True)
    ax1.set_aspect('equal')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)

    theta = np.pi / 20
    A00 = np.cos(theta)
    A01 = -np.sin(theta)
    A10 = np.sin(theta)
    A11 = np.cos(theta)
    def xprime(xx, yy):
        return A00 * xx + A01 * yy
    def yprime(xx, yy):
        return A10 * xx + A11 * yy

    ax2.scatter(xprime(xv, yv), yprime(xv, yv), **style2)
    ax2.plot(xprime(x, y0), yprime(x, y0), c='b', alpha=0.5)
    ax2.plot(xprime(x0, y), yprime(x0, y), c='r', alpha=0.5)
    ax2.plot(xprime(x1, y), yprime(x1, y), c='m', alpha=0.5)
    ax2.plot(xprime(x, y1), yprime(x, y1), c='g', alpha=0.5)
    ax2.grid(True)
    ax2.set_aspect('equal')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    plt.show()
