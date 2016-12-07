#!/usr/bin/env python
import numpy as np
import emcee
import JAM.pyjam as pyjam
import JAM.utils.util_dm as util_dm
import JAM.utils.util_mge as util_mge
from astropy.cosmology import Planck13
from time import time, localtime, strftime
from scipy.stats import gaussian_kde
import pickle
# from emcee.utils import MPIPool
# import sys

# parameter boundaries. [lower, upper]
boundary = {'cosinc': [0.0, 1.0], 'beta': [0.0, 0.4], 'logrho_s': [3.0, 10.0],
            'rs': [5.0, 45.0], 'gamma': [-1.6, 0.0], 'ml': [0.5, 15]
            }
# parameter gaussian priors. [mean, sigma]
prior = {'cosinc': [0.0, 1e4], 'beta': [0.0, 1e4], 'logrho_s': [5.0, 1e4],
         'rs': [10.0, 1e4], 'gamma': [-1.0, 1e4], 'ml': [1.0, 1e4]
         }

model = {'boundary': boundary, 'prior': prior}


def check_boundary(parsDic):
    '''
    Check whether parameters are within the boundary limits
    input
      parsDic: parameter dictionary {'paraName', value}
    output
      -np.inf or 0.0
    '''
    for key in parsDic.keys():
        if boundary[key][0] < parsDic[key] < boundary[key][1]:
            pass
        else:
            return -np.inf
    return 0.0


def lnprior(parsDic):
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


def flat_initp(keys, nwalkers):
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


def estimatePrameters(flatchain, method='median', flatlnprob=None):
    '''
    '''
    if method == 'median':
        return np.percentile(flatchain, 50, axis=0)
    elif method == 'mean':
        return np.mean(flatchain, axis=0)
    elif method == 'peak':
        pars = np.zeros(flatchain.shape[1])
        for i in range(len(pars)):
            xmin = flatchain[:, i].min()
            xmax = flatchain[:, i].max()
            kernel = gaussian_kde(flatchain[:, i])
            x = np.linspace(xmin, xmax, 300)
            prob = kernel(x)
            pars[i] = np.mean(x[prob == prob.max()])
        return pars
    elif method == 'max':
        return np.mean(flatchain[flatlnprob == flatlnprob.max(), :], axis=0)
    else:
        raise ValueError('Do not support {} method'.format(method))


def analyzeRst(sampler, nburnin=0):
    '''
    analyze the mcmc and generage resutls
      chain: (nwalker, nstep, ndim)
      lnprobability: (nwalker, nstep)
    '''
    rst = {}
    rst['chain'] = sampler.chain
    rst['lnprobability'] = sampler.lnprobability
    rst['acor'] = sampler.acor
    print 'Mean autocorrelation time: {:.1f}'.format(np.mean(rst['acor']))
    rst['acceptance_fraction'] = sampler.acceptance_fraction
    rst['goodchains'] = ((rst['acceptance_fraction'] > 0.15) *
                         (rst['acceptance_fraction'] < 0.6))
    print ('Mean accept fraction: {:.3f}'
           .format(np.mean(rst['acceptance_fraction'])))
    if rst['goodchains'].sum() / float(model['nwalkers']) < 0.6:
        print 'Warning - goodchain fraction less than 0.6'
        goodchains = np.ones_like(rst['goodchains'], dtype=bool)
    else:
        goodchains = rst['goodchains']
    flatchain = \
        rst['chain'][goodchains, nburnin, :].reshape((-1, model['ndim']))
    flatlnprob = rst['lnprobability'][goodchains, nburnin].reshape(-1)
    medianPars = estimatePrameters(flatchain, flatlnprob=flatlnprob)
    meanPars = estimatePrameters(flatchain, flatlnprob=flatlnprob,
                                 method='mean')
    peakPars = estimatePrameters(flatchain, flatlnprob=flatlnprob,
                                 method='peak')
    maxPars = estimatePrameters(flatchain, flatlnprob=flatlnprob, method='max')
    rst['medianPars'] = medianPars
    rst['meanPars'] = meanPars
    rst['peakPars'] = peakPars
    rst['maxPars'] = maxPars
    print 'medianPars'
    printParameters(model['JAMpars'], medianPars)
    print 'meanPars'
    printParameters(model['JAMpars'], meanPars)
    print 'peakPars'
    printParameters(model['JAMpars'], peakPars)
    print 'maxPars'
    printParameters(model['JAMpars'], maxPars)
    return rst


def dump():
    with open('{}/{}'.format(model['outfolder'], model['fname']), 'wb') as f:
        pickle.dump(model, f)


def load():
    pass


def printBoundaryPrior(model):
    pass


def printParameters(names, values):
    temp = ['{}: {:.2f}  '.format(names[i], values[i])
            for i in range(len(names))]
    print ''.join(temp)


def printModelInfo(model):
    print '--------------------------------------------------'
    print 'Model Info'
    print 'pyJAM model run at {}'.format(model['date'])
    print 'Galaxy name: {}'.format(model.get('name', 'LHY'))
    print 'Number of tracer MGEs: {}'.format(model['lum2d'].shape[0])
    print ('Number of luminous potential MGEs: {}'
           .format(model['pot2d'].shape[0]))
    print ('Number of good observational bins: {}/{}'
           .format(model['goodbins'].sum(), len(model['goodbins'])))
    print 'Effective radius: {:.2f} arcsec'.format(model['Re_arcsec'])
    print 'errScale {:.2f}'.format(model['errScale'])
    print 'Model shape: {}'.format(model['shape'])
    print 'Sigmapsf: {:.2f}  Pixelsize: {:.2f}'.format(model['sigmapsf'],
                                                       model['pixsize'])
    print 'Burning steps: {}'.format(model['burnin'])
    print 'Clip:  {}'.format(model['clip'])
    if model['clip'] in ['sigma']:
        print 'Clip steps: {}'.format(model['clipStep'])
        if model['clip'] == 'sigma':
            print 'Sigma for sigmaclip: {:.2f}'.format(model['clipSigma'])
    print 'nwalkers: {}'.format(model['nwalkers'])
    print 'Run steps: {}'.format(model['runStep'])
    print 'Initial positons of mcmc chains: {}'.format(model['p0'])
    print '--------------------------------------------------'


def _sigmaClip(sampler, pos):
    maxN = model['clipMaxN']
    lnprob = model['lnprob']
    N = 0
    while True:
        oldGoodbins = model['goodbins'].copy()
        startTime = time()
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, model['clipStep'])
        flatchain = sampler.flatchain
        flatlnprob = sampler.flatlnprobability
        pars = estimatePrameters(flatchain, method='max',
                                 flatlnprob=flatlnprob)
        N += 1
        rmsModel = lnprob(pars, returnRms=True)
        chi2 = lnprob(pars, returnChi2=True)
        inThreeSigma = (abs(rmsModel - model['rms']) < model['errRms'] *
                        model['clipSigma'])
        model['goodbins'] *= inThreeSigma
        chi2dof = ((rmsModel[oldGoodbins] - model['rms'][oldGoodbins])**2 /
                   model['errRms'][oldGoodbins]).sum() / oldGoodbins.sum()

        print '--------------------------------------------------'
        print 'Time for clip {}: {:.2f}s'.format(N, time()-startTime)
        print 'best parameters:'
        printParameters(model['JAMpars'], pars)
        print 'Number of old goodbins: {}'.format(oldGoodbins.sum())
        print 'Number of new goodbins: {}'.format(model['goodbins'].sum())
        print 'Chi2: {:.2f}'.format(chi2)
        print 'Chi2/dof: {:.3f}'.format(chi2dof)
        if N >= maxN:
            print 'Warning - clip more than {}'.format(maxN)
            break
        if np.array_equal(model['goodbins'], oldGoodbins):
            print 'Clip srccess'
            break
        if model['goodbins'].sum() / float(model['initGoodbins'].sum()) <\
           model['minFraction']:
            print ('clip too many pixels: goodbin fraction < {:.2f}'
                   .format(model['minFraction']))
            break
    return sampler


def _runEmcee(sampler, p0):
    burninStep = model['burnin']
    runStep = model['runStep']
    # burnin
    startTime = time()
    pos, prob, state = sampler.run_mcmc(p0, burninStep)
    print 'Start running'
    print 'Time for burnin: {:.2f}s'.format(time()-startTime)
    flatchain = sampler.flatchain
    flatlnprob = sampler.flatlnprobability
    pars = estimatePrameters(flatchain, method='max', flatlnprob=flatlnprob)
    printParameters(model['JAMpars'], pars)
    sampler.reset()
    # clip run if set
    if model['clip'] == 'noclip':
        pass
    elif model['clip'] == 'sigma':
        sampler = _sigmaClip(sampler, pos)
    else:
        raise ValueError('clip value {} not supported'
                         .format(model['clip']))
    # Final run
    sampler.run_mcmc(pos, runStep)
    return sampler


def lnprob_massFollowLight(pars, returnRms=False, returnChi2=False):
    cosinc, beta, ml = pars
    # print pars
    parsDic = {'cosinc': cosinc, 'beta': beta, 'ml': ml}
    if np.isinf(check_boundary(parsDic)):
        return -np.inf
    lnpriorValue = lnprior(parsDic)
    inc = np.arccos(cosinc)
    Beta = np.zeros(model['lum2d'].shape[0]) + beta
    mflJAM = model['JAM']
    rmsModel = mflJAM.run(inc, Beta, ml=ml)
    if returnRms:
        return rmsModel
    chi2 = (((rmsModel[model['goodbins']] - model['rms'][model['goodbins']]) /
             model['errRms'][model['goodbins']])**2).sum()
    if returnChi2:
        return chi2
    if np.isnan(chi2):
        print ('Warning - JAM return nan value, beta={:.2f} may not'
               ' be correct'.format(beta))
        return -np.inf
    return -0.5*chi2 + lnpriorValue


class mcmc:
    '''
    input parameter
      galaxy: A python dictionary which contains all the necessary data for
        running JAM model and arguments for running 'emcee'. See the text
        below for a detialed description of the keys. Parameters with * must
        be provided.

    1. Obervitional data
      *lum2d: 2D mge coefficents for galaxy's surface brightness. N*3 array,
        density [L_solar/pc^2], sigma [arcsec], qobs [none]
      pot2d: 2D mge coefficents for galaxy's luminous matter potential. If
        not set, the same value will be used as lum2d.
      distance: Galaxy's distance [Mpc]
      redshift: Galaxy's redshift. (One only need to provied distance or
        redshift for a successful run.
      *xbin: x position of the data points [arcsec], lenght N array.
      *ybin: y position of the data points [arcsec], lenght N array.
      vel: velocity at (x, y) [km/s], lenght N array.
      errVel: velocity error [km/s], lenght N array.
      disp: velocity dispersion at (x, y) [km/s], lenght N array.
      errDisp: velocity dispersion error [km/s], lenght N array.
      *rms: root-mean-squared velocity [km/s], lenght N array.
      errRms: root-mean-squared velocity error [km/s], if rms and errRms
        are not provided, they will be calculated from vel, disp, errVel
        and errDisp, otherwise rms and errRms will be directly used for
        JAM modelling.
      errSacle: errRms *= errSacle. Default: 1.0
      goodbins: bool array, true for goodbins which are used in MCMC
        fitting, lenght N array. Default: all bins are good.
      bh: black hole mass [M_solar], scalar. Default: None
    2. JAM arguments
      sigmapsf: psf for observational data [arcsec], scalar. Default: 0.0
      pixsize: instrument pixsize with which kinematic data are obseved
        [arcsec], scalar. Default: 0.0
      shape: deprojection shape, bool. Default: oblate
      nrad: interpolation grid size, integer. Default: 25
    3. Emcee arguments
      burnin: Number of steps for burnin, integer. Default: 500 steps
      clip: if run mcmc with 3-sigma clip, bool. Default: False
      clipStep: Number of steps for each clip, integer. Default: 1000 steps
      runStep: Number of steps for the final run, integer. Default: 1000 steps
      nwalkers: Number of walkers in mcmc, integer. Default: 30
      p0: inital position distribution of the mcmc chains. String, flat or fit
    '''
    def __init__(self, galaxy):
        '''
        initialize model parameters and data
        '''
        self.lum2d = galaxy['lum2d']
        self.Re_arcsec = util_mge.Re(self.lum2d)
        self.pot2d = galaxy.get('pot2d', self.lum2d.copy())
        self.distance = galaxy.get('distance', None)
        self.redshift = galaxy.get('redshift', None)
        if self.distance is None:
            if self.redshift is None:
                raise RuntimeError('redshift or distance must be provided!')
            else:
                self.distance = \
                    Planck13.angular_diameter_distance(self.redshift).value
        self.xbin = galaxy.get('xbin', None)
        self.ybin = galaxy.get('ybin', None)
        self.vel = galaxy.get('vel', None)
        self.errVel = galaxy.get('errVel', None)
        self.disp = galaxy.get('disp', None)
        self.errDisp = galaxy.get('errDisp', None)
        self.errScale = galaxy.get('errScale', 1.0)
        self.rms = galaxy.get('rms', None)
        if self.rms is None:
            if (self.vel is None) or (self.disp is None):
                raise RuntimeError('rms or (vel, disp) must be provided')
            else:
                self.rms = np.sqrt(self.vel**2 + self.disp**2)
        self.errRms = galaxy.get('errRms', None)
        if self.errRms is None:
            if (self.errVel is not None) and (self.errDisp is not None) and \
                    (self.vel is not None) and (self.disp is not None):
                self.errRms = (np.sqrt((self.errVel*self.vel)**2 +
                                       (self.errDisp*self.disp)**2) /
                               np.sqrt(self.vel**2 + self.disp**2))
            else:
                self.errRms = self.rms*0.0 + np.median(self.rms)
        self.errRms *= self.errScale
        self.goodbins = galaxy.get('goodbins',
                                   np.ones_like(self.rms, dtype=bool))
        self.bh = galaxy.get('bh', None)
        self.shape = galaxy.get('shape', 'oblate')

        # save all the model parameters into global dictionary
        global model
        date = strftime('%Y-%m-%d %X', localtime())
        model['date'] = date
        model['name'] = galaxy.get('name', 'LHY')
        model['lum2d'] = self.lum2d
        model['Re_arcsec'] = self.Re_arcsec
        model['pot2d'] = self.pot2d
        model['distance'] = self.distance
        model['redshift'] = self.redshift
        model['xbin'] = self.xbin
        model['ybin'] = self.ybin
        model['vel'] = self.vel
        model['errVel'] = self.errVel
        model['disp'] = self.disp
        model['errDisp'] = self.errDisp
        model['rms'] = self.rms
        model['errRms'] = self.errRms
        model['errScale'] = self.errScale
        model['goodbins'] = self.goodbins
        model['initGoodbins'] = self.goodbins.copy()
        model['bh'] = self.bh
        model['sigmapsf'] = galaxy.get('sigmapsf', 0.0)
        model['pixsize'] = galaxy.get('pixsize', 0.0)
        model['shape'] = self.shape
        model['nrad'] = galaxy.get('nrad', 25)
        model['burnin'] = galaxy.get('burnin', 500)
        model['clip'] = galaxy.get('clip', 'noclip')
        model['clipStep'] = galaxy.get('clipStep', 1000)
        model['clipSigma'] = galaxy.get('clipSigma', 3.0)
        model['clipMaxN'] = galaxy.get('clipMaxN', 4)
        model['minFraction'] = galaxy.get('minFraction', 0.7)
        model['runStep'] = galaxy.get('runStep', 1000)
        model['nwalkers'] = galaxy.get('nwalkers', 30)
        model['p0'] = galaxy.get('p0', 'flat')
        model['threads'] = galaxy.get('threads', 1)
        model['outfolder'] = galaxy.get('outfolder', './')
        model['fname'] = galaxy.get('fname', 'dump.dat')
        # initialize the JAM class and pass to the global parameter
        model['JAM'] = \
            pyjam.axi_rms.jam(model['lum2d'], model['pot2d'], model['distance'],
                              model['xbin'], model['ybin'], mbh=model['bh'],
                              quiet=True, sigmapsf=model['sigmapsf'],
                              pixsize=model['pixsize'], nrad=model['nrad'],
                              shape=model['shape'])
        # self.model[''] = self.
        # set cosinc and beta priors to aviod JAM crashing
        if self.shape == 'oblate':
            qall = np.append(self.lum2d[:, 2], self.pot2d[:, 2])
            boundary['cosinc'][1] = np.min((qall**2 - 0.003)**0.5)
        elif self.shape == 'prolate':
            qall = np.append(self.lum2d[:, 2], self.pot2d[:, 2])
            if np.any(qall) < 0.101:
                raise ValueError('Input qobs smaller than 0.101 for'
                                 ' prolate model')
            boundary['cosinc'][1] = np.min(((100.0 - 1.0/qall**2)/99.0)**0.5)
            boundary['beta'][0] = -1.0
            boundary['beta'][1] = 0.0
        self.startTime = time()
        print '**************************************************'
        print 'Initialize mcmc success!'

    def set_boundary(self, key, value):
        '''
        Reset the parameter boundary value
        key: parameter name. Sting
        value: boundary values. length two list
        '''
        if key not in boundary.keys():
            raise ValueError('parameter name \'{}\' not correct'.format(key))
        if len(value) != 2:
            raise ValueError('Boundary limits must be a length 2 list')
        print ('Change {} limits to [{}, {}], defaults are [{}, {}]'
               .format(key, value[0], value[1], boundary[key][0],
                       boundary[key][1]))
        boundary[key] = value

    def set_prior(self, key, value):
        '''
        Reset the parameter prior value
        key: parameter name. Sting
        value: prior values. length two list
        '''
        if key not in prior.keys():
            raise ValueError('parameter name \'{}\' not correct'.format(key))
        if len(value) != 2:
            raise ValueError('prior must be a length 2 list')
        print ('Change {} prior to [{}, {}], defaults are [{}, {}]'
               .format(key, value[0], value[1], prior[key][0], prior[key][1]))
        prior[key] = value

    def set_config(self, key, value):
        '''
        Reset the configuration parameter
        key: parameter name. Sting
        value: allowed values
        '''
        if key not in model.keys():
            print 'key {} does not exist, create a new key'.format(key)
            model[key] = value
        else:
            print ('Change parameter {} to {}, original value is {}'
                   .format(key, value, model[key]))
            model[key] = value

    def get_config(self, key):
        '''
        Get the configuration parameter value of parameter key
        '''
        return model.get(key, None)

    def massFollowLight(self):
        print '--------------------------------------------------'
        print 'Mass follow light model'
        printModelInfo(model)
        printBoundaryPrior(model)
        model['lnprob'] = lnprob_massFollowLight
        model['ndim'] = 3
        model['JAMpars'] = ['cosinc', 'beta', 'ml']
        nwalkers = model['nwalkers']
        threads = model['threads']
        ndim = model['ndim']
        JAMpars = model['JAMpars']
        if model['p0'] == 'flat':
            p0 = flat_initp(JAMpars, nwalkers)
        elif model['p0'] == 'fit':
            print ('Calculate maximum lnprob positon from optimisiztion - not'
                   'implemented yet')
            exit(0)
        else:
            raise ValueError('p0 must be flat or fit, {} is '
                             'not supported'.format(model['p0']))
        # pool = MPIPool()
        # if not pool.is_master():
        #     pool.wait()
        #     sys.exit(0)
        # Initialize sampler
        initSampler = \
            emcee.EnsembleSampler(nwalkers, ndim, lnprob_massFollowLight,
                                  threads=threads)
        sampler = _runEmcee(initSampler, p0)
        # pool.close()
        print '--------------------------------------------------'
        print ('Finish! Total elapsed time: {:.2f}s'
               .format(time()-self.startTime))
        rst = analyzeRst(sampler)
        model['rst'] = rst
        dump()

    def spherical_gNFW(self):
        pass
