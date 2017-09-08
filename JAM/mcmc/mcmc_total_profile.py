#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : ../JAM/mcmc/mcmc_total_profile.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 06.09.2017
# Last Modified Date: 08.09.2017
# Last Modified By  : Hongyu Li <lhy88562189@gmail.com>
# -*- coding: utf-8 -*-
# File              : ../JAM/mcmc/mcmc_total_profile.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 02.09.2017
# Last Modified: 02.09.2017
# ============================================================================
#  DESCRIPTION: ---
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
# ORGANIZATION:
#      VERSION: 0.0
# ============================================================================
import emcee
import numpy as np
import JAM.utils.util_mge as util_mge
import JAM.utils.util_dm as util_dm
# import matplotlib.pyplot as plt
import pickle
from time import time, localtime, strftime
import sys


def printModelInfo(model):
    print('--------------------------------------------------')
    print('Model Info')
    print('fit total profile run at {}'.format(model['date']))
    print('Model type: {}'.format(model.get('type', None)))
    print('Fit option: {}'.format(model['fit']))
    print('Number of mass MGEs: {}'.format(model['mge2d'].shape[0]))
    print('Number of good bins: {}/{}'
          .format(model['good'].sum(), len(model['good'])))
    print('errScale {:.2f}'.format(model['errScale']))
    print('Burning steps: {}'.format(model['burnin']))
    print('nwalkers: {}'.format(model['nwalkers']))
    print('Run steps: {}'.format(model['runStep']))
    print('--------------------------------------------------')


def printBoundaryPrior(model):
    parsNames = model['parsNames']
    prior = model['prior']
    boundary = model['boundary']
    for name in parsNames:
        print('{:10s} - prior: {:8.3f} {:10.3e}'
              '    - boundary: [{:8.3f}, {:8.3f}]'
              .format(name, prior[name][0], prior[name][1],
                      boundary[name][0], boundary[name][1]))


def dump(model):
    with open('{}/{}'.format(model['outfolder'], model['fname']), 'wb') as f:
        pickle.dump(model, f)


def check_boundary(parsDic, boundary=None):
    '''
    Check whether parameters are within the boundary limits
    input
      parsDic: parameter dictionary {'paraName', value}
    output
      True or False
    '''
    for key in parsDic.keys():
        if boundary[key][0] < parsDic[key] < boundary[key][1]:
            pass
        else:
            return False
    return True


def lnprior(parsDic, prior=None):
    '''
    Calculate the gaussian prior lnprob
    input
      parsDic: parameter dictionary {'paraName', value}
    output
      lnprob
    '''
    rst = 0.0
    for key in parsDic.keys():
        rst += -0.5 * (parsDic[key] - prior[key][0])**2/prior[key][1]**2
    return rst


def flat_initp(keys, nwalkers, boundary=None):
    '''
    create initital positions for mcmc. Flat distribution within prior.
    keys: List of parameter name
    nwalkers: number of emcee walkers
    '''
    ndim = len(keys)
    p0 = np.zeros([nwalkers, ndim])
    for i in range(ndim):
        p0[:, i] = np.random.uniform(low=boundary[keys[i]][0]+1e-4,
                                     high=boundary[keys[i]][1]-1e-4,
                                     size=nwalkers)
    return p0


def run_mcmc(sampler, p0, nstep, nprint=20):
    print_step = nstep // nprint
    for i, result in enumerate(sampler.sample(p0, iterations=nstep)):
        if (i+1) % print_step == 0:
            print("{0:5.1%}".format(float(i+1) / nstep))
    return result


def lnprob_mge_gNFW(pars, model=None, returnType='lnprob'):
    logalpha, logrho_s, rs, gamma = pars
    parsDic = {'logalpha': logalpha, 'logrho_s': logrho_s,
               'rs': rs, 'gamma': gamma}
    # check if parameters are in the boundary
    if not check_boundary(parsDic, boundary=model['boundary']):
        rst = {}
        rst['lnprob'] = -np.inf
        rst['chi2'] = np.inf
        rst['profiles'] = None
        return rst[returnType]
    # fit option
    fit = model['fit']
    # observed total density profile, error and goodbins
    logrhoT = model['logrhoT']
    logrhoTerr = model['logrhoTerr']
    logMT = model['logMT']
    logMTerr = model['logMTerr']
    good = model['good']
    # model stellar profile
    logrhoS = model['logrhoS_sps'] + logalpha
    logMS = model['logMS_sps'] + logalpha
    # model dark matter profile
    dh = util_dm.gnfw1d(10**logrho_s, rs, gamma)
    logrhoD = np.log10(dh.densityProfile(model['r']))
    # logMD = np.zeros(len(logMS))
    # for i in range(len(logMD)):
    #     logMD[i] = np.log10(dh.enclosedMass(model['r'][i]))
    tem_mge = util_mge.mge(dh.mge2d(), inc=np.pi/2.0)
    logMD = np.log10(tem_mge.enclosed3Dluminosity(model['r']*1e3))
    # model total profile
    logrhoTmodel = np.log10(10**logrhoS + 10**logrhoD)
    logMTmodel = np.log10(10**logMS + 10**logMD)
    chi2_dens = np.sum(((logrhoT[good] - logrhoTmodel[good]) /
                        logrhoTerr[good])**2)
    chi2_mass = np.sum(((logMT[good] - logMTmodel[good]) /
                       logMTerr[good])**2)
    if fit == 'dens':
        chi2 = chi2_dens
    elif fit == 'mass':
        chi2 = chi2_mass
    elif fit == 'all':
        chi2 = chi2_dens + chi2_mass
    else:
        raise ValueError('Invalid fit option {}'.format(fit))
    if returnType == 'profiles':
        # tem_mge = util_mge.mge(dh.mge2d(), inc=np.pi/2.0)
        # logMD = np.log10(tem_mge.enclosed3Dluminosity(model['r']*1e3))
        rst = {}
        rst['r'] = model['r']
        rst['rho_star'] = logrhoS
        rst['rho_dark'] = logrhoD
        rst['rho_total'] = logrhoTmodel
        rst['M_star'] = logMS
        rst['M_dark'] = logMD
        rst['M_total'] = logMTmodel
        rst['chi2'] = chi2
        rst['rho_total_obs'] = logrhoT
        rst['M_total_obs'] = logMT
        rst['good'] = good
        rst['pars'] = parsDic
        return rst
    elif returnType == 'lnprob':
        return -0.5*chi2
    elif returnType == 'chi2':
        return chi2
    else:
        raise KeyError('Invalid returnType {}'.format(returnType))


class mcmc:
    '''
    input parameter
      galaxy: A python dictionary which contains all the necessary data for
        running JAM model and arguments for running 'emcee'. See the text
        below for a detialed description of the keys. Parameters with * must
        be provided.

    1. Obervitional data
      *mge2d: 2D mge coefficents for galaxy's surface brightness or stellar
        mass surface density. N*3 array, density [L_solar/pc^2],
        sigma [arcsec], qobs [none]
      *distance: Galaxy's distance [Mpc]
      *logrhoT: input density to be fitted, length N 1D array [L_solar/kpc^2]
      logrhoTerr: error of the input density, length N 1D array
      *r: radii of the give density, length N 1D array [kpc]
      inc_rad: inclination of the model. Default: np.pi/2
      errSacle: logrhoTerr *= errSacle. Default: 1.0
      good: length N bool array, pixels with False value will not be used in
        fitting. Default: all bins are good.
    2. Emcee arguments
      burnin: Number of steps for burnin, integer. Default: 500 steps
      runStep: Number of steps for the final run, integer. Default: 1000 steps
      nwalkers: Number of walkers in mcmc, integer. Default: 200
      threads: Number of threads used for emcee, integer. Default: 1
      fit: profile to be fitted, 'dens', 'mass' or 'all'. Default: 'dens'
    3. Output arguments
      outfolder: output folder path. Default: '.'
      fname: output filename. Default: 'dump.dat'
    '''

    def __init__(self, galaxy):
        '''
        initialize model parameters and data
        '''
        self.boundary = {'cosinc': [0.0, 1.0], 'logrho_s': [5.0, 10.0],
                         'rs': [5.0, 45.0], 'gamma': [-1.6, 0.0],
                         'logdelta': [-1.0, 1.0], 'q': [0.1, 0.999],
                         'logalpha': [-0.5, 0.5]
                         }
        # parameter gaussian priors. [mean, sigma]
        self.prior = {'cosinc': [0.0, 1e4], 'logrho_s': [5.0, 1e4],
                      'rs': [10.0, 1e4], 'gamma': [-1.0, 1e4],
                      'logdelta': [-1.0, 1e4], 'q': [0.9, 1e4],
                      'logalpha': [0.0, 1e4]
                      }

        galaxy['prior'] = self.prior
        galaxy['boundary'] = self.boundary
        date = strftime('%Y-%m-%d %X', localtime())
        galaxy['date'] = date
        logrhoTerr = np.zeros_like(galaxy['logrhoT']) + 0.05
        galaxy.setdefault('logrhoTerr', logrhoTerr)
        galaxy.setdefault('inc_rad', np.pi/2.0)
        galaxy.setdefault('errScale', 1.0)
        good = np.ones_like(galaxy['logrhoT'], dtype=bool)
        galaxy.setdefault('good', good)
        galaxy.setdefault('burnin', 500)
        galaxy.setdefault('runStep', 1000)
        galaxy.setdefault('nwalkers', 200)
        galaxy.setdefault('threads', 1)
        galaxy.setdefault('fit', 'dens')
        galaxy.setdefault('fname', 'dump.dat')
        galaxy.setdefault('outfolder', '.')
        galaxy['logrhoTerr'] *= galaxy['errScale']
        galaxy['logMTerr'] *= galaxy['errScale']
        self.model = galaxy
        print('--------------------------------------------------')
        print('mcmc_total_profile initialse success!')

    def set_boundary(self, key, value):
        '''
        Reset the parameter boundary value
        key: parameter name. Sting
        value: boundary values. length two list
        '''
        if key not in self.boundary.keys():
            raise ValueError('parameter name \'{}\' not correct'.format(key))
        if len(value) != 2:
            raise ValueError('Boundary limits must be a length 2 list')
        print('Change {} limits to [{}, {}], defaults are [{}, {}]'
              .format(key, value[0], value[1], self.boundary[key][0],
                      self.boundary[key][1]))
        self.boundary[key] = value

    def set_prior(self, key, value):
        '''
        Reset the parameter prior value
        key: parameter name. Sting
        value: prior values. length two list
        '''
        if key not in self.prior.keys():
            raise ValueError('parameter name \'{}\' not correct'.format(key))
        if len(value) != 2:
            raise ValueError('prior must be a length 2 list')
        print('Change {} prior to [{}, {}], defaults are [{}, {}]'
              .format(key, value[0], value[1], self.prior[key][0],
                      self.prior[key][1]))
        self.prior[key] = value

    def mge_gNFW(self):
        print('--------------------------------------------------')
        print('mge_gNFW')
        self.model['lnprob'] = lnprob_mge_gNFW
        self.model['type'] = 'mge_gNFW'
        self.model['ndim'] = 4
        self.model['parsNames'] = ['logalpha', 'logrho_s', 'rs', 'gamma']
        printModelInfo(self.model)
        printBoundaryPrior(self.model)
        sys.stdout.flush()
        mass_mge = self.model['mge2d'].copy()
        mge = util_mge.mge(mass_mge, self.model['inc_rad'],
                           dist=self.model['distance'])
        logrhoS_sps = np.log10(mge.meanDensity(self.model['r']*1e3)) + 9.0
        logMS_sps = np.log10(mge.enclosed3Dluminosity(self.model['r']*1e3))
        self.model['logrhoS_sps'] = logrhoS_sps
        self.model['logMS_sps'] = logMS_sps
        nwalkers = self.model['nwalkers']
        threads = self.model['threads']
        ndim = self.model['ndim']
        parsNames = self.model['parsNames']
        burnin = self.model['burnin']
        runStep = self.model['runStep']
        p0 = flat_initp(parsNames, nwalkers, boundary=self.boundary)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.model['lnprob'],
                                        kwargs={'model': self.model},
                                        threads=threads)
        startTime = time()
        print('Start burnin')
        # pos, prob, state = sampler.run_mcmc(p0, burnin)
        pos, prob, state = run_mcmc(sampler, p0, burnin, nprint=20)
        sampler.reset()
        print('Time for burnin: {:.2f}s'.format(time()-startTime))
        print('Start running')
        sys.stdout.flush()
        # sampler.run_mcmc(pos, runStep)
        run_mcmc(sampler, pos, runStep, nprint=20)
        print('Finish! Total elapsed time: {:.2f}s'.format(time()-startTime))
        self.model['chain'] = sampler.chain
        self.model['lnprobability'] = sampler.lnprobability
        try:
            self.model['acor'] = sampler.acor
        except:
            self.model['acor'] = None
        self.model['acceptance_fraction'] = sampler.acceptance_fraction
        dump(self.model)
