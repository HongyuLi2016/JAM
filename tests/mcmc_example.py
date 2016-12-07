#!/usr/bin/env python
import numpy as np
from contex import JAM


def test_emcee():
    '-----------------------------prepare data--------------------------'
    mge2d, dist, xbin, ybin, rms, erms = np.load('data/64_8341-3704.npy')
    galaxy = {'lum2d': mge2d, 'distance': dist, 'xbin': xbin, 'ybin': ybin,
              'rms': rms, 'errRms': erms, 'runStep': 10, 'burnin': 5,
              'clipStep': 10, 'shape': 'oblate', 'threads': 8, 'clip': 'sigma'}

    lhy = JAM.mcmc_pyjam.mcmc(galaxy)
    # use set_config method to change model configurations
    lhy.set_config('sigmapsf', 1.0)
    # change prior/boundary values
    lhy.set_prior('gamma', [-1.0, 1e-4])  # fix gamma as -1.0 (i.e. NFW)
    lhy.set_boundary('gamma', [-2.0, 0.0])
    lhy.massFollowLight()

if __name__ == '__main__':
    test_emcee()
