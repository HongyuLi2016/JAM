#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from contex import pyjam
from velocity_plot import velocity_plot
from scipy import stats
from matplotlib import colors
import matplotlib
text_font =\
    matplotlib.font_manager.FontProperties(family='times new roman', size=10)
ticks_font =\
    matplotlib.font_manager.FontProperties(family='times new roman', size=14)


def set_tick(ax, font=None):
    if font is None:
        font = ticks_font
    for l in ax.get_xticklabels():
        l.set_fontproperties(ticks_font)
    for l in ax.get_yticklabels():
        l.set_fontproperties(ticks_font)


def test_jam_axi_rms():
    np.random.seed(123)
    xbin, ybin = np.random.uniform(low=[-55, -40],
                                   high=[55, 40], size=[1000, 2]).T

    inc = 60.   # Assumed galaxy inclination
    r = np.sqrt(xbin**2 + (ybin/np.cos(np.radians(inc)))**2)
    # Radius in the plane of the disk
    a = 40  # Scale length in arcsec
    vr = 2000*np.sqrt(r)/(r+a)  # Assumed velocity profile
    vel = vr * np.sin(np.radians(inc))*xbin/r  # Projected velocity field
    sig = 8700/(r+a)  # Assumed velocity dispersion profile
    rms = np.sqrt(vel**2 + sig**2)  # Vrms field in km/s

    surf = np.array([39483., 37158., 30646., 17759., 5955.1, 1203.5,
                     174.36, 21.105, 2.3599, 0.25493])
    sigma = np.array([0.153, 0.515, 1.58, 4.22, 10, 22.4, 48.8, 105, 227, 525])
    qObs = np.full_like(sigma, 0.57)

    distance = 16.5   # Assume Virgo distance in Mpc (Mei et al. 2007)
    mbh = 1e8  # Black hole mass in solar masses
    beta = np.full_like(surf, 0.3)

    lum_mge = np.zeros([len(surf), 3])
    lum_mge[:, 0] = surf
    lum_mge[:, 1] = sigma
    lum_mge[:, 2] = qObs
    pot_mge = lum_mge.copy()

    sigmapsf = 0.0
    pixsize = 0.8
    goodbins = r > 10
    # Arbitrarily exclude the center to illustrate how to use goodbins
    lhy = pyjam.pyclass.jam(lum_mge, pot_mge, distance, xbin, ybin, mbh=mbh,
                            rms=rms, goodbins=goodbins, sigmapsf=sigmapsf,
                            pixsize=pixsize, shape='oblate', nrad=30, index=0.5)
    rmsModel = lhy.run(inc, beta)
    xbinC, ybinC, rmsModelC, mlC = np.load('Cappellair_oblate.npy')
    fig = plt.figure()
    vmin, vmax = stats.scoreatpercentile(rmsModelC, [0.5, 99.5])
    norm = colors.Normalize(vmin=(vmin), vmax=(vmax))
    ax1 = fig.add_subplot(221)
    velocity_plot(xbin, ybin, rmsModelC, ax=ax1, norm=norm, text='MC', size=2)
    ax2 = fig.add_subplot(222)
    velocity_plot(xbin, ybin, rmsModel, ax=ax2, norm=norm, text='LHY', size=2)
    ax3 = fig.add_subplot(223)
    vmin, vmax = stats.scoreatpercentile(rmsModelC-rmsModel, [0.5, 99.5])
    norm = colors.Normalize(vmin=(vmin), vmax=(vmax))
    velocity_plot(xbin, ybin, rmsModelC-rmsModel, ax=ax3, norm=norm,
                  text='MC-LHY', size=2)
    ax4 = fig.add_subplot(224)
    ax4.hist(rmsModelC-rmsModel, range=[-10, 10], bins=50)
    ax4.set_xlabel('km/s', fontproperties=text_font)
    set_tick(ax4)
    ax4.text(0.05, 0.25, 'max: {:.2f}'.format(max(rmsModelC-rmsModel)),
             transform=ax4.transAxes, fontproperties=text_font)
    ax4.text(0.05, 0.45, 'min: {:.2f}'.format(min(rmsModelC-rmsModel)),
             transform=ax4.transAxes, fontproperties=text_font)
    ax4.text(0.05, 0.65, 'MC-LHY',
             transform=ax4.transAxes, fontproperties=text_font)
    plt.show()


if __name__ == '__main__':
    test_jam_axi_rms()
