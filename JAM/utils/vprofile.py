#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : JAM/utils/vprofile.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 29.09.2017
# Last Modified Date: 29.09.2017
# Last Modified By  : Hongyu Li <lhy88562189@gmail.com>
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import rc
import util_fig
# rc('mathtext', fontset='stix')
sauron = util_fig.sauron
ticks_font = util_fig.ticks_font
text_font = util_fig.text_font
ticks_font1 = util_fig.ticks_font1
label_font = util_fig.label_font


def _rotate(x0, y0, angle):
    angle_rad = np.radians(angle)
    xnew = x0 * np.cos(angle_rad) + y0 * np.sin(angle_rad)
    ynew = - x0 * np.sin(angle_rad) + y0 * np.cos(angle_rad)
    return xnew, ynew


def vprofile(x0, y0, v, vModel=None, ax=None, angle=0.0, width=1.5,
             text=None, xlabel='x arcsec',
             ylabel='$\\rm V_{rms} \ km/s$',
             **kwargs):
    if ax is None:
        ax = plt.gca()
    xnew, ynew = _rotate(x0, y0, angle)
    in_slit = (ynew > -width) * (ynew < width)
    ax.plot(xnew[in_slit], v[in_slit], '*b', markeredgecolor=None, **kwargs)
    if vModel is not None:
        ax.plot(xnew[in_slit], vModel[in_slit], '+r', **kwargs)
    if text is not None:
        ax.text(0.05, 0.85, text, transform=ax.transAxes,
                fontproperties=text_font)
    if ylabel is not None:
        ax.set_ylabel(r'{}'.format(ylabel))
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontproperties=label_font)
    util_fig.set_labels(ax)
