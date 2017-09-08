#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : /Users/lhy/src/JAM/JAM/utils/util_rst_fitTP.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 07.09.2017
# Last Modified Date: 07.09.2017
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
import pickle
from scipy.stats import gaussian_kde
import corner_plot
import util_fig
import matplotlib.pyplot as plt
from scipy import interpolate
util_fig.text_font.set_size(16)


def load(name, path='.'):
    with open('{}/{}'.format(path, name), 'rb') as f:
        data = pickle.load(f)
    return data


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


class modelRst(object):

    def __init__(self, name, path='.', best='median'):
        self.model = load(name, path=path)
        ndim = self.model['ndim']
        modelType = self.model['type']
        self.goodchain = ((self.model['acceptance_fraction'] > 0.15) *
                          (self.model['acceptance_fraction'] < 0.75))
        if self.goodchain.sum()/float(len(self.goodchain)) < 0.3:
            print('Goodchain fraction less than 0.3')
            self.goodchain[:] = True
        self.flatchain = \
            self.model['chain'][self.goodchain, :, :].reshape(-1, ndim)
        self.flatlnprob = \
            self.model['lnprobability'][self.goodchain, :].reshape(-1)
        self.medianPars = estimatePrameters(self.flatchain,
                                            flatlnprob=self.flatlnprob)
        self.maxPars = estimatePrameters(self.flatchain,
                                         flatlnprob=self.flatlnprob)
        self.meanPars = estimatePrameters(self.flatchain,
                                          flatlnprob=self.flatlnprob)
        self.peakPars = estimatePrameters(self.flatchain,
                                          flatlnprob=self.flatlnprob)
        if modelType == 'mge_gNFW':
            self.labels = [r'$\mathbf{\alpha_{IMF}}$', r'$\mathbf{log\rho_s}$',
                           '$\mathbf{R_s}$', '$\mathbf{\gamma}$']
        else:
            raise ValueError('Invalid model type {}'.format(modelType))
        print self.model.keys()

    def plotChain(self, figname='chain_fitTP.png', outpath='.'):
        ndim = self.model['ndim']
        chain = self.model['chain']
        figsize = (8.0, ndim*2.0)
        fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=figsize)
        for i in range(ndim):
            axes[i].plot(chain[:, :, i].T, color='k', alpha=0.2)
            axes[i].set_ylabel(self.labels[i])
        axes[-1].set_xlabel('nstep')
        fig.savefig('{}/{}'.format(outpath, figname), dpi=100)

    def cornerPlot(self, figname='mcmc_fitTP.png', outpath='.',
                   clevel=[0.683, 0.95, 0.997], truths='max', true=None,
                   hbins=30, color=[0.8936, 0.5106, 0.2553], **kwargs):
        switch = {'median': self.medianPars, 'mean': self.meanPars,
                  'peak': self.peakPars, 'max': self.maxPars,
                  'true': true}
        truthsValue = switch[truths]
        kwargs['labels'] = kwargs.get('labels', self.labels)
        fig = corner_plot.corner(self.flatchain, clevel=clevel, hbins=hbins,
                                 truths=truthsValue, color=color,
                                 quantiles=[0.16, 0.5, 0.84], **kwargs)
        fig.savefig('{}/{}'.format(outpath, figname), dpi=100)

    def plotProfile(self, nlines=200, figname='profile_fitTP.png',
                    fname='profile_fitTP.dat', outpath='.',):
        self.profiles = {}  # dictionary containing profiles
        modelType = self.model['type']
        Re_kpc = self.model['Re_kpc']
        r = self.model['r']
        logr = np.log10(r)
        logrhoT = self.model['logrhoT']
        logrhoTerr = self.model['logrhoTerr']
        logMT = self.model['logMT']
        logMTerr = self.model['logMTerr']
        good = self.model['good']
        lnprob = self.model['lnprob']
        ntotal = self.flatchain.shape[0]
        step = ntotal // nlines
        parsAll = self.flatchain[::step, :]
        self.profiles['r'] = r
        self.profiles['nprofiles'] = parsAll.shape[0]

        if modelType == 'mge_gNFW':
            self.alpha_IMF = self.medianPars[0]
            stellarProfiles = np.zeros([len(r), parsAll.shape[0], 2])
            darkProfiles = np.zeros([len(r), parsAll.shape[0], 2])
            totalProfiles = np.zeros([len(r), parsAll.shape[0], 2])
            for i in range(parsAll.shape[0]):
                pars = parsAll[i, :]
                rst = lnprob(pars, model=self.model, returnType='profiles')
                stellarProfiles[:, i, 0] = 10**rst['rho_star']
                stellarProfiles[:, i, 1] = 10**rst['M_star']
                darkProfiles[:, i, 0] = 10**rst['rho_dark']
                darkProfiles[:, i, 1] = 10**rst['M_dark']
                totalProfiles[:, i, 0] = 10**rst['rho_total']
                totalProfiles[:, i, 1] = 10**rst['M_total']
        else:
            raise ValueError('Invalid model type {}'.format(modelType))
        self.profiles['stellar'] = stellarProfiles
        self.profiles['dark'] = darkProfiles
        self.profiles['total'] = totalProfiles
        stellar_Re, dark_Re, total_Re, fdm_Re = self.enclosed3DMass(Re_kpc)
        MassRe = {}
        MassRe['Re_kpc'] = Re_kpc
        MassRe['stellar'] = np.log10(stellar_Re)
        MassRe['dark'] = np.log10(dark_Re)
        MassRe['total'] = np.log10(total_Re)
        MassRe['fdm'] = fdm_Re
        # plt.plot(logr, np.log10(totalProfiles[:, :, 1]))
        # plt.savefig('telm.png')
        # exit()
        self.profiles['MassRe'] = MassRe
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        fig.subplots_adjust(left=0.15, bottom=0.10, right=0.98,
                            top=0.98, wspace=0.1, hspace=0.1)
        # plot stellar density
        if self.profiles['stellar'] is not None:
            axes[0].plot(logr, np.log10(stellarProfiles[:, :, 0]),
                         'y', alpha=0.1)
            axes[1].plot(logr, np.log10(stellarProfiles[:, :, 1]),
                         'y', alpha=0.1)
        # plot dark density
        if self.profiles['dark'] is not None:
            axes[0].plot(logr, np.log10(darkProfiles[:, :, 0]),
                         'r', alpha=0.1)
            axes[1].plot(logr, np.log10(darkProfiles[:, :, 1]),
                         'r', alpha=0.1)
        # plot total density
        if self.profiles['total'] is not None:
            axes[0].plot(logr, np.log10(totalProfiles[:, :, 0]),
                         'c', alpha=0.1)
            axes[1].plot(logr, np.log10(totalProfiles[:, :, 1]),
                         'c', alpha=0.1)
        axes[0].errorbar(logr[good], logrhoT[good], yerr=logrhoTerr[good],
                         fmt='o', color='k', capsize=3, markersize=3.5)
        axes[1].errorbar(logr[good], logMT[good], yerr=logMTerr[good],
                         fmt='o', color='k', capsize=3, markersize=3.5)
        axes[0].axvline(np.log10(Re_kpc), ls="dashed", color='b', linewidth=2)
        axes[1].axvline(np.log10(Re_kpc), ls="dashed", color='b', linewidth=2)
        axes[1].set_xlabel(r'$\mathbf{log_{10}\, R \, [kpc]}$',
                           fontproperties=util_fig.label_font)
        axes[1].set_ylabel(r'$\mathbf{log_{10}\, M(R)\, [M_{\odot}]}$',
                           fontproperties=util_fig.label_font)
        axes[0].set_ylabel(r'$\mathbf{log_{10}\, \rho \,[M_{\odot}\, '
                           'kpc^{-3}]}$', fontproperties=util_fig.label_font)
        axes[1].text(0.8, 0.1, 'Total', color='c', transform=axes[1].transAxes,
                     fontproperties=util_fig.text_font)
        axes[1].text(0.8, 0.2, 'Dark', color='r', transform=axes[1].transAxes,
                     fontproperties=util_fig.text_font)
        axes[1].text(0.8, 0.3, 'Stellar', color='y',
                     transform=axes[1].transAxes,
                     fontproperties=util_fig.text_font)

        axes[0].text(0.05, 0.05, '$\mathbf{M^*(R_e)}$: %5.2f'
                     % (MassRe['stellar']), transform=axes[0].transAxes,
                     fontproperties=util_fig.text_font)
        axes[0].text(0.35, 0.05, '$\mathbf{M^T(R_e)}$: %5.2f'
                     % (MassRe['total']), transform=axes[0].transAxes,
                     fontproperties=util_fig.text_font)
        axes[0].text(0.05, 0.25, r'$\mathbf{\alpha_{IMF}}$: %5.2f'
                     % (self.alpha_IMF), transform=axes[0].transAxes,
                     fontproperties=util_fig.text_font)
        axes[0].text(0.35, 0.25, '$\mathbf{f_{DM}(R_e)}$: %5.2f'
                     % (MassRe['fdm']), transform=axes[0].transAxes,
                     fontproperties=util_fig.text_font)
        util_fig.set_labels(axes[0])
        util_fig.set_labels(axes[1])
        fig.savefig('{}/{}'.format(outpath, figname), dpi=100)
        with open('{}/{}'.format(outpath, fname), 'wb') as f:
            pickle.dump(self.profiles, f)

    def enclosed3DMass(self, r):
        if self.profiles['total'] is not None:
            total_median = np.percentile(self.profiles['total'][:, :, 1],
                                         50, axis=1)
            ftotal = \
                interpolate.interp1d(self.profiles['r'], total_median,
                                     kind='linear', bounds_error=False,
                                     fill_value=np.nan)
            total = ftotal(r)
        else:
            total = np.nan

        if self.profiles['stellar'] is not None:
            stellar_median = np.percentile(self.profiles['stellar'][:, :, 1],
                                           50, axis=1)
            fstellar = \
                interpolate.interp1d(self.profiles['r'], stellar_median,
                                     kind='linear', bounds_error=False,
                                     fill_value=np.nan)
            stellar = fstellar(r)
        else:
            stellar = np.nan

        if self.profiles['dark'] is not None:
            dark_median = np.percentile(self.profiles['dark'][:, :, 1],
                                        50, axis=1)
            fdark = \
                interpolate.interp1d(self.profiles['r'], dark_median,
                                     kind='linear', bounds_error=False,
                                     fill_value=np.nan)
            dark = fdark(r)
        else:
            dark = np.nan
        fdm = dark / total
        return stellar, dark, total, fdm
