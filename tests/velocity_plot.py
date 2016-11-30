#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyfits
from optparse import OptionParser
import glob
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from scipy import special, signal, ndimage, stats
from matplotlib import tri, colors, colorbar
from matplotlib import rc
from matplotlib.patches import Rectangle,Circle
rc('mathtext',fontset='stix')
_cdict = {'red': ((0.000,   0.01,   0.01),
                  (0.170,   0.0,    0.0),
                  (0.336,   0.4,    0.4),
                  (0.414,   0.5,    0.5),
                  (0.463,   0.3,    0.3),
                  (0.502,   0.0,    0.0),
                  (0.541,   0.7,    0.7),
                  (0.590,   1.0,    1.0),
                  (0.668,   1.0,    1.0),
                  (0.834,   1.0,    1.0),
                  (1.000,   0.9,    0.9)),
         'green':((0.000,   0.01,   0.01), 
                  (0.170,   0.0,    0.0),
                  (0.336,   0.85,   0.85),
                  (0.414,   1.0,    1.0),
                  (0.463,   1.0,    1.0),
                  (0.502,   0.9,    0.9),
                  (0.541,   1.0,    1.0),
                  (0.590,   1.0,    1.0),
                  (0.668,   0.85,   0.85),
                  (0.834,   0.0,    0.0),
                  (1.000,   0.9,    0.9)),
          'blue':((0.000,   0.01,   0.01),
                  (0.170,   1.0,    1.0),
                  (0.336,   1.0,    1.0),
                  (0.414,   1.0,    1.0),
                  (0.463,   0.7,    0.7),
                  (0.502,   0.0,    0.0),
                  (0.541,   0.0,    0.0),
                  (0.590,   0.0,    0.0),
                  (0.668,   0.0,    0.0),
                  (0.834,   0.0,    0.0),
                  (1.000,   0.9,    0.9))
          }

sauron = colors.LinearSegmentedColormap('sauron', _cdict)

ticks_font = mpl.font_manager.FontProperties(family='times new roman', style='normal', size='small', weight='bold', stretch='normal')
text_font = mpl.font_manager.FontProperties(family='times new roman', style='normal', size='large', weight='bold', stretch='normal')
ticks_font1 = mpl.font_manager.FontProperties(family='times new roman', style='normal', size='x-small', weight='bold', stretch='normal')

def lhy_scatter(ax,x,y,c=None,size=0.8,norm=None, \
       cmap=None,shape='C'):
  h=size*2
  for i in range(x.size):    
    if shape=='R':
      squre=Rectangle(xy=(x[i]-h/2.,y[i]-h/2.),\
         fc=cmap(norm(c[i])),width=h, height=h,\
         ec=cmap(norm(c[i])),zorder=1,lw=0.)
      squre=ax.add_artist(squre)
    if shape=='C':
      circle=Circle(xy=(x[i],y[i]),\
         fc=cmap(norm(c[i])),radius=size,\
         ec=cmap(norm(c[i])),zorder=1,lw=0.)
      ax.add_artist(circle)

def velocity_plot(x0,y0,v,ax=None,cmap=None,norm=None,\
      equal=True,text=None,bar=True,fig=None,barlabel=\
      '$\mathbf{km/s}$',shape='C',size=0.24):
  if ax is None:
    ax=plt.gca()
  if fig is None:
    fig=plt.gcf()
  if cmap is None:
    cmap=sauron
  nans = np.isnan(v)
  vmin, vmax = stats.scoreatpercentile(v[~nans], [0.5, 99.5])
  if norm is None:    
    norm = mpl.colors.Normalize(vmin=(vmin), vmax=(vmax))
  point=ax.scatter(x0,y0,c=v)
  point.remove()
  lhy_scatter(ax,x0,y0,c=v,size=size,norm=norm, \
       cmap=sauron,shape=shape)
  lhy_scatter(ax,x0[nans],y0[nans],c=y0[nans]*0.0+vmax,size=size*0.8,norm=norm, \
       cmap=sauron,shape=shape)
 
  if text is not None:
    ax.text(0.05,0.85,text,transform=ax.transAxes,\
       fontproperties=text_font)

  if equal is True:
    ax.set_aspect('equal',adjustable='box',anchor='C')
    xlim=[x0.min()-0.5,x0.max()+0.5]
    ylim=[y0.min()-0.5,y0.max()+0.5]
    lim=[min(xlim[0],ylim[0]),max(xlim[1],ylim[1])]
    ax.set_xlim(lim[::-1])
    ax.set_ylim(lim)
  ax.xaxis.set_major_locator(MaxNLocator(5))
  ax.yaxis.set_major_locator(MaxNLocator(5))
  for l in ax.get_xticklabels():
    #l.set_rotation(45) 
    l.set_fontproperties(ticks_font)
  for l in ax.get_yticklabels():
    #l.set_rotation(45) 
    l.set_fontproperties(ticks_font)
  if bar is True:
    pos=ax.get_position()
    figsize=fig.get_size_inches()
    #print pos.x0,pos.x1,pos.y0,pos.y1,pos.height,pos.width
    height=min(pos.width*figsize[0],pos.height*figsize[1])
    ph=height/figsize[1]
    px=(pos.x0+pos.x1)/2.+height/2./figsize[0]
    py=(pos.y0+pos.y1)/2-height/2./figsize[1]
    axc=fig.add_axes([px,py,0.02,ph])
    mpl.colorbar.ColorbarBase(axc,orientation='vertical',norm=norm, cmap=sauron)
    axc.set_ylabel(barlabel,fontsize='xx-small')
    #axc.yaxis.set_major_locator(MaxNLocator(5))
    for l in axc.get_xticklabels():
      #l.set_rotation(45) 
      l.set_fontproperties(ticks_font1)
    for l in axc.get_yticklabels():
      #l.set_rotation(45) 
      l.set_fontproperties(ticks_font1)
