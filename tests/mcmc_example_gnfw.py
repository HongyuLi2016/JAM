#!/usr/bin/env python
import numpy as np
from contex import JAM


def test_emcee():
    '-----------------------------read data--------------------------'
    mge2d, dist, xbin, ybin, rms, erms = np.load('data/mock_sgnfw_data.npy')
    '---------------------configure model perameters-----------------'
    # see mcmc/mcmc_pyjam.py for a detialed description of allowed parameters
    galaxy = {'lum2d': mge2d, 'distance': dist, 'xbin': xbin, 'ybin': ybin,
              'rms': rms, 'errRms': erms, 'runStep': 3000, 'burnin': 500,
              'clipStep': 1000, 'shape': 'oblate', 'threads': 8,
              'clip': 'noclip', 'outfolder': 'data',
              'fname': 'mock_gNFW_out.dat'}

    lhy = JAM.mcmc_pyjam.mcmc(galaxy)
    # use set_config method to change model configurations
    # lhy.set_config('sigmapsf', 1.0)
    # lhy.set_config('pixsize', 1.0)
    # change prior/boundary values
    # lhy.set_prior('gamma', [-1.0, 1e-4])  # fix gamma as -1.0 (i.e. NFW)
    # lhy.set_boundary('gamma', [-2.0, 0.0])
    # lhy.massFollowLight()
    lhy.spherical_gNFW()


if __name__ == '__main__':
    test_emcee()
