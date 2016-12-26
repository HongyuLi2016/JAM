#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from scipy import stats
from matplotlib import tri
from matplotlib import rc
from matplotlib.patches import Rectangle, Circle
import util_fig
rc('mathtext', fontset='stix')
sauron = util_fig.sauron
ticks_font = util_fig.ticks_font
text_font = util_fig.text_font
ticks_font1 = util_fig.ticks_font1


def lhy_scatter(ax, x, y, c=None, size=0.8, norm=None,
                cmap=None, shape='C'):
    h = size*2
    for i in range(x.size):
        if shape == 'R':
            squre = Rectangle(xy=(x[i]-h/2., y[i]-h/2.),
                              fc=cmap(norm(c[i])), width=h, height=h,
                              ec=cmap(norm(c[i])), zorder=1, lw=0.)
            squre = ax.add_artist(squre)
        if shape == 'C':
            circle = Circle(xy=(x[i], y[i]), fc=cmap(norm(c[i])),
                            radius=size, ec=cmap(norm(c[i])),
                            zorder=1, lw=0.)
            ax.add_artist(circle)


def lhy_map(ax, x, y, vel, vmin=None, vmax=None,
            ncolors=64, markersize=1.5, cmap=sauron,
            norm=None):
    if vmin is None:
        vmin = np.min(vel)
    if vmax is None:
        vmax = np.max(vel)
    levels = np.linspace(vmin, vmax, ncolors)
    triangles = tri.Triangulation(x.ravel(), y.ravel())
    ax.tricontourf(triangles, vel.clip(vmin, vmax).ravel(),
                   levels=levels, cmap=cmap, norm=norm)
    ax.plot(x, y, 'k.', markersize=markersize, alpha=1.)


def velocity_plot(x0, y0, v, ax=None, cmap=sauron, norm=None,
                  equal=True, text=None, bar=True, fig=None,
                  barlabel='$\mathbf{km/s}$', shape='C',
                  vmap='map', size=0.24, ncolors=64, markersize=1.5):
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    nans = np.isnan(v)
    vmin, vmax = stats.scoreatpercentile(v[~nans], [0.5, 99.5])
    if norm is None:
        norm = mpl.colors.Normalize(vmin=(vmin), vmax=(vmax))
    if vmap == 'dots':
        lhy_scatter(ax, x0, y0, c=v, size=size, norm=norm,
                    cmap=sauron, shape=shape)
        lhy_scatter(ax, x0[nans], y0[nans], c=y0[nans]*0.0+vmax, size=size*0.8,
                    norm=norm, cmap=sauron, shape=shape)
    elif vmap == 'map':
        lhy_map(ax, x0[~nans], y0[~nans], v[~nans], vmin=vmin, vmax=vmax,
                ncolors=64, markersize=markersize, norm=norm, cmap=cmap)
    else:
        raise ValueError('do not support vmap {}'.format(vmap))

    if text is not None:
        ax.text(0.05, 0.85, text, transform=ax.transAxes,
                fontproperties=text_font)

    if equal is True:
        lim = [min(np.append(x0, y0)), max(np.append(x0, y0))]
        util_fig.equal_limits(ax, lim=lim)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    util_fig.set_labels(ax)

    if bar is True:
        util_fig.add_colorbar(ax, sauron, norm, position='right',
                              barlabel=barlabel)
