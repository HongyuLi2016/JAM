#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : run_fit_total.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 07.09.2017
# Last Modified Date: 08.09.2017
# Last Modified By  : Hongyu Li <lhy88562189@gmail.com>
# ============================================================================
#  DESCRIPTION: ---
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
# ORGANIZATION:
#      VERSION: 0.0
# ============================================================================
import numpy as np
from contex import JAM
import pickle
from optparse import OptionParser
# import matplotlib.pyplot as plt


def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()
    if len(args) != 1:
        print('Error - please provide a folder name')
        exit(1)

    # read data files
    with open('{}/profiles.dat'.format(args[0])) as f:
        profiles = pickle.load(f)
    with open('{}/rst.dat'.format(args[0])) as f:
        rst = pickle.load(f)
    mge2d, pa, eps = np.load('{}/mge_mass.npy'.format(args[0]))

    # print profiles.keys()
    # print rst.keys()

    # prepare input data arrays
    kpc = rst['dist'] * np.pi / 0.648 / 1e3
    rbin = np.sqrt(rst['xbin']**2 + rst['ybin']**2)
    dataRange_Kpc = np.percentile(rbin, 95.0) * 1.2 * kpc
    Re_kpc = rst['Re_arcsec'] * kpc
    inc_rad = np.arccos(rst['bestPars'][0])
    profile_per_den = np.percentile(np.log10(profiles['total'][:, :, 0]),
                                    [16, 50, 84], axis=1)
    profile_per_M = np.percentile(np.log10(profiles['total'][:, :, 1]),
                                  [16, 50, 84], axis=1)
    fit_range = (0.1 < profiles['r']) * (profiles['r'] < dataRange_Kpc)
    r = profiles['r'][fit_range]
    logrhoT = profile_per_den[1, fit_range]
    logrhoTerr = 0.5*(profile_per_den[2, fit_range] -
                      profile_per_den[0, fit_range])
    logMT = profile_per_M[1, fit_range]
    logMTerr = 0.5*(profile_per_M[2, fit_range] -
                    profile_per_M[0, fit_range])
    # input arguments
    galaxy = {}
    galaxy['mge2d'] = mge2d
    galaxy['distance'] = rst['dist']
    galaxy['logrhoT'] = logrhoT
    galaxy['logrhoTerr'] = logrhoTerr
    galaxy['logMT'] = logMT
    galaxy['logMTerr'] = logMTerr
    galaxy['r'] = r
    galaxy['Re_kpc'] = Re_kpc
    galaxy['inc_rad'] = inc_rad
    galaxy['errScale'] = 1.0
    galaxy['burnin'] = 600
    galaxy['runStep'] = 500
    galaxy['nwalkers'] = 100
    galaxy['threads'] = 4
    galaxy['fit'] = 'dens'
    galaxy['fname'] = 'mcmc_tprofile.dat'
    galaxy['outfolder'] = args[0]
    # galaxy[''] =
    mcmc = JAM.mcmc_total_profile.mcmc(galaxy)
    mcmc.set_boundary('logalpha', [-0.5, 1.2])
    mcmc.mge_gNFW()


if __name__ == '__main__':
    main()
