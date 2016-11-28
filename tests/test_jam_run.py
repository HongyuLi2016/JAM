#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath('..'))
import pyjam



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
    # beta = np.full_like(surf, 0.3)

    lum_mge = np.zeros([len(surf), 3])
    lum_mge[:, 0] = surf
    lum_mge[:, 1] = sigma
    lum_mge[:, 2] = qObs
    pot_mge = lum_mge.copy()

    sigmapsf = 0.
    pixsize = 0.8
    goodbins = r > 10
    # Arbitrarily exclude the center to illustrate how to use goodbins
    lhy = pyjam.jam(lum_mge, pot_mge, distance, xbin, ybin, mbh=mbh, rms=rms,
                    goodbins=goodbins, sigmapsf=sigmapsf, pixsize=pixsize,
                    shape='oblate')
    plt.plot(lhy.xGrid.ravel()/lhy.pc, lhy.yGrid.ravel()/lhy.pc, '.', markersize=5)
    plt.show()


if __name__ == '__main__':
    test_jam_axi_rms()
