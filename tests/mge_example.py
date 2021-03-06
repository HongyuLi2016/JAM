#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : mge_example.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 07.09.2017
# Last Modified Date: 07.09.2017
# Last Modified By  : Hongyu Li <lhy88562189@gmail.com>
import numpy as np
from contex import JAM
import JAM.utils.util_dm as udm  # import dark matter utility class
import JAM.utils.util_mge as umge  # import mge utility class
import matplotlib.pyplot as plt

r = np.logspace(np.log10(0.1), np.log10(100), 500)
logr = np.log10(r)

# 3 parameters for gNFW profile
gamma = -1.0  # usually between [-2.0, 0.0]
rs = 30.0     # [kpc]
rho_s = 1e5   # [M_solar/kpc^2]  (1[M_solar/kpc^3] = 1e9[M_solar/pc^3])

gnfw_dh = udm.gnfw1d(rho_s, rs, gamma)  # initialize a gNFW class
profile = np.log10(gnfw_dh.densityProfile(r))  # get the density profile


mge2d = gnfw_dh.mge2d()  # fit the density profile using MGE

# initialize the MGE class using the MGE from dark halo
mge = umge.mge(mge2d, np.pi/2.0, shape='oblate')
print('--------------------')
print('Enclosed Mass within 10kpc (gNFW profile): {:.3e}'
      .format(gnfw_dh.enclosedMass(1.0)))
print('Enclosed Mass within 10kpc (MGE approximation): {:.3e}'
      .format(mge.enclosed3Dluminosity(10.0*1e3)[0]))
# here are some unit conversion
profile_mge = np.log10(mge.luminosityDensity(r*1e3, 0)*1e9)
# plot the density profile of the dark halo and the mge approximation
line_dh, = plt.plot(logr, profile, 'k')
line_mge, = plt.plot(logr, profile_mge, 'r')
plt.xlabel('logR [kpc]')
plt.ylabel('logrho [M_solar/kpc^3]')
plt.legend([line_dh, line_mge], ['gNFW halo', 'MGE approximation'])
plt.savefig('profile.png')
