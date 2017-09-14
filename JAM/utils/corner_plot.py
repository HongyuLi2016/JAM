#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : JAM/utils/corner_plot.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 14.09.2017
# Last Modified Date: 14.09.2017
# Last Modified By  : Hongyu Li <lhy88562189@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
import util_fig
from scipy.stats import gaussian_kde
rc('mathtext', fontset='stix')
sauron = util_fig.sauron
ticks_font = util_fig.ticks_font
text_font = util_fig.text_font
ticks_font1 = util_fig.ticks_font1
label_font = util_fig.label_font
ticks_font.set_size(16)
label_font.set_size(18)
text_font.set_size(20)


def quantile(x, q, weights=None):
    """
    Like numpy.percentile, but:

    * Values of q are quantiles [0., 1.] rather than percentiles [0., 100.]
    * scalar q not supported (q must be iterable)
    * optional weights on x

    """
    if weights is None:
        return np.percentile(x, [100. * qi for qi in q])
    else:
        idx = np.argsort(x)
        xsorted = x[idx]
        cdf = np.add.accumulate(weights[idx])
        cdf /= cdf[-1]
        return np.interp(q, cdf, xsorted).tolist()


def hist2d(x, y, *args, **kwargs):
    # KEYWORDS
    # ax=axes
    # extent= [[xmin,xmax],[ymin,ymax]]
    # bins= number of bins
    # clevel  confidence level [0.3,0.683,0.954,0.997]
    # color  color for each level  list between 0,1  [0.1,0.2,0.3,0.4]
    # linewidths =

    ax = kwargs.pop("ax", plt.gca())
    extent = kwargs.pop("extent", [[x.min(), x.max()], [y.min(), y.max()]])
    bins = kwargs.pop("bins", 30)
    clevel = kwargs.pop("clevel", [0.2, 0.3, 0.683, 0.864, 0.997])
    color = kwargs.pop("color", [0.8936, 0.6382, 0.5106, 0.2553, 0.01276])
    linewidths = kwargs.pop("linewidths", 0.8)

    X = np.linspace(extent[0][0], extent[0][1], bins + 1)
    Y = np.linspace(extent[1][0], extent[1][1], bins + 1)
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=[X, Y],
                                 weights=kwargs.get('weights', None))
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "`extent` argument.")
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    level = clevel[:]
    for i, v0 in enumerate(clevel):
        try:
            level[i] = Hflat[sm <= v0][-1]
        except:
            level[i] = Hflat[0]
    level = level[::-1]
    for i in range(len(level)-1):
        if level[i] >= level[i+1]:
            level[i+1] = level[i] * 1.0001
    # Z = np.meshgrid(X,Y)
    H = H.T
    ax.contour(H, extent=[extent[0][0], extent[0][1], extent[1][0],
                          extent[1][1]], levels=level,
               colors=sauron(color*255), antialiased=True,
               linewidths=linewidths)
    ax.set_xlim(extent[0])
    ax.set_ylim(extent[1])


def corner(xs, weights=None, labels=None, extents=None, truths=None,
           truth_color="r", scale_hist=False, quantiles=[], bins=30,
           verbose=False, fig=None, resample=True, sampleSize=100000,
           smooth='scott', **kwargs):
    # hbins  bin number for 1D histgram
    clevel = kwargs.pop("clevel", [0.2, 0.3, 0.683, 0.864, 0.997])
    color = kwargs.pop("color", [0.8936, 0.6382, 0.5106, 0.1553, 0.01276])
    linewidths = kwargs.pop("linewidths", 0.8)

    xs = xs.T
    if resample:
        kde = gaussian_kde(xs, bw_method=smooth)
        xs = kde.resample(sampleSize)
    K = len(xs)
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.05 * factor  # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim
    fig, axes = plt.subplots(K, K, figsize=(dim, dim))
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)
    if extents is None:
        extents = [np.percentile(x, [0.5, 99.5]) for x in xs]
        m = np.array([e[0] == e[1] for e in extents], dtype=bool)
        if np.any(m):
            raise ValueError(("It looks like the parameter(s) in column(s)"
                              "{0} have no dynamic range. Please provide an "
                              "`extent` argument.")
                             .format(", ".join(map("{0}".format,
                                               np.arange(len(m))[m]))))
    for i, x in enumerate(xs):
        ax = axes[i, i]
        # Plot the histograms.
        n, b, p = ax.hist(x, weights=weights, bins=kwargs.get("hbins", 80),
                          range=extents[i], color="w",
                          facecolor='b', histtype="stepfilled")
        if truths is not None:
            ax.axvline(truths[i], color=truth_color, linewidth=5.0, alpha=0.9)

        # Plot quantiles if wanted.
        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                ax.axvline(q, ls="dashed", color='g', linewidth=2)
            if verbose:
                print("Quantiles:")
                print(zip(quantiles, qvalues))

        # Set up the axes.
        ax.set_xlim(extents[i])
        if scale_hist:
            maxn = np.max(n)
            ax.set_ylim(-0.1 * maxn, 1.1 * maxn)
        else:
            ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(4))

        if i < K - 1:
            ax.set_xticklabels([])
        else:
            for l in ax.get_xticklabels():
                l.set_rotation(45)
                l.set_fontproperties(ticks_font)
            if labels is not None:
                ax.set_xlabel(labels[i], fontproperties=label_font)
                ax.xaxis.set_label_coords(0.5, -0.3)

        for j, y in enumerate(xs):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                ax.set_frame_on(False)
                continue
            elif j == i:
                continue
            hist2d(y, x, ax=ax, extent=[extents[j], extents[i]], bins=bins,
                   clevel=clevel, color=color, linewidths=linewidths,
                   weights=weights, **kwargs)

            if truths is not None:
                ax.plot(truths[j], truths[i], "o", color=truth_color,
                        markersize=10.0)
                ax.axvline(truths[j], color=truth_color,
                           linewidth=2.0, alpha=0.9)
                ax.axhline(truths[i], color=truth_color,
                           linewidth=2.0, alpha=0.9)

            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                for l in ax.get_xticklabels():
                    l.set_rotation(45)
                    l.set_fontproperties(ticks_font)
                tick = ax.get_xticklabels()
                tick[3].label1on = False
                if labels is not None:
                    ax.set_xlabel(labels[j], fontsize=20, fontweight='heavy')
                    ax.xaxis.set_label_coords(0.5, -0.3)

            if j > 0:
                ax.set_yticklabels([])
            else:
                for l in ax.get_yticklabels():
                    l.set_rotation(45)
                    l.set_fontproperties(ticks_font)
                if labels is not None:
                    ax.set_ylabel(labels[i], fontsize=20, fontweight='heavy')
                    ax.yaxis.set_label_coords(-0.3, 0.5)
    return fig


if __name__ == '__main__':
    samples = np.load('samples.npy')

    nstep = 130
    nwalker = 10
    ndim = 3
    chain = samples[:, 50:, :].reshape(-1, ndim)
    fig = corner(chain, truths=[0.5, 0.3, 1.7],
                 extents=[[0., 90.], [0., 0.4], [0., 1.], [-1.6, 0.]],
                 linewidths=0.2, quantiles=[0.16, 0.5, 0.84],
                 labels=['1', '2', '3'], bins=20)
    plt.show()
