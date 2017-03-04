#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import util_mge
import util_dm
import matplotlib.pyplot as plt
import util_fig
from util_rst import modelRst
import pickle
ticks_font = util_fig.ticks_font
text_font = util_fig.text_font
ticks_font1 = util_fig.ticks_font1
label_font = util_fig.label_font
ticks_font.set_size(11)
text_font.set_size(14)
label_font.set_size(14)


def _extractProfile(mge, r):
    '''
    mge is the mge object, defined in util_mge.py
    r is the radius in kpc, 1d array
    '''
    mass = np.zeros_like(r)
    density = np.zeros_like(r)
    for i in range(len(r)):
        mass[i] = mge.enclosed3Dluminosity(r[i]*1e3)
        density[i] = mge.meanDensity(r[i]*1e3) * 1e9
    return mass, density


def _plotProfile(r, profiles, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.plot(r, profiles, **kwargs)


class profile(modelRst):
    def __init__(self, name, path='.', burnin=0, nlines=200, r=None):
        super(profile, self).__init__(name, path=path, burnin=burnin,
                                      best='median')
        self.profiles = {}  # dictionary containing profiles
        if r is None:
            r = np.logspace(np.log10(0.5), np.log10(100.0), 100)
        self.profiles['r'] = r
        # select a subchain to calculate density profile
        ntotal = self.flatchain.shape[0]
        step = ntotal / nlines
        if step == 0:
            print('Warning - nlines > total number of samples')
            step = 1
        ii = np.zeros(ntotal, dtype=bool)
        ii[::step] = True
        self.profiles['nprofiles'] = ii.sum()

        stellarProfiles = np.zeros([len(r), ii.sum(), 2])
        mls = self.flatchain[ii, 2].ravel()
        if self.data['type'] in ['spherical_gNFW', 'spherical_gNFW_gas']:
            # Calculate stellar mass profiles
            mass, density = _extractProfile(self.LmMge, r)
            for i in range(ii.sum()):
                stellarProfiles[:, i, 0] = mls[i] * density
                stellarProfiles[:, i, 1] = mls[i] * mass
            self.profiles['stellar'] = stellarProfiles
            # Calculate dark matter profiles
            if self.DmMge is None:
                self.profiles['dark'] = None
            else:
                darkProfiles = np.zeros_like(stellarProfiles)
                logrho_s = self.flatchain[ii, 3].ravel()
                rs = self.flatchain[ii, 4].ravel()
                gamma = self.flatchain[ii, 5].ravel()
                for i in range(ii.sum()):
                    tem_dh = util_dm.gnfw1d(10**logrho_s[i], rs[i], gamma[i])
                    darkProfiles[:, i, 0] = tem_dh.densityProfile(r)
                    for j in range(len(r)):
                        darkProfiles[j, i, 1] = tem_dh.enclosedMass(r[j])
                self.profiles['dark'] = darkProfiles
        elif self.data['type'] == 'spherical_gNFW_gradient':
            # Calculate stellar mass profiles
            pot_ng = self.pot2d.copy()
            pot_tem = np.zeros([1, 3])
            sigma = pot_ng[:, 1]/self.data['Re_arcsec']
            delta = self.flatchain[ii, 3].ravel()
            mass = np.zeros([len(r), pot_ng.shape[0]])
            density = np.zeros([len(r), pot_ng.shape[0]])
            for i in range(pot_ng.shape[0]):
                pot_tem[:, :] = pot_ng[i, :]
                mge_pot = util_mge.mge(pot_tem, inc=self.inc,
                                       shape=self.shape, dist=self.dist)
                mass[:, i], density[:, i] = _extractProfile(mge_pot, r)
            for i in range(ii.sum()):
                ML = util_mge.ml_gradient_gaussian(sigma, delta[i], ml0=mls[i])
                stellarProfiles[:, i, 0] = np.sum(ML * density, axis=1)
                stellarProfiles[:, i, 1] = np.sum(ML * mass, axis=1)
            self.profiles['stellar'] = stellarProfiles

            # Calculate dark matter profiles
            if self.DmMge is None:
                self.profiles['dark'] = None
            else:
                darkProfiles = np.zeros_like(stellarProfiles)
                logrho_s = self.flatchain[ii, 4].ravel()
                rs = self.flatchain[ii, 5].ravel()
                gamma = self.flatchain[ii, 6].ravel()
                for i in range(ii.sum()):
                    tem_dh = util_dm.gnfw1d(10**logrho_s[i], rs[i], gamma[i])
                    darkProfiles[:, i, 0] = tem_dh.densityProfile(r)
                    for j in range(len(r)):
                        darkProfiles[j, i, 1] = tem_dh.enclosedMass(r[j])
                self.profiles['dark'] = darkProfiles
        else:
            raise ValueError('model type {} not supported'
                             .format(self.data['type']))

        # calculate total profiles
        totalProfiles = stellarProfiles.copy()
        if self.profiles['dark'] is not None:
            totalProfiles += self.profiles['dark']
        if self.data['bh'] is not None:
            totalProfiles[:, :, 1] += self.data['bh']  # add black hole mass
        self.profiles['total'] = totalProfiles
        self.gas3d = self.data.get('gas3d', None)
        Re_kpc = self.data['Re_arcsec'] * self.pc / 1e3
        MassRe = {}
        MassRe['stellar'] = \
            np.log10(self.enclosed3DStellarMass(Re_kpc)[0])
        MassRe['dark'] = self.enclosed3DdarkMass(Re_kpc)
        MassRe['total'] = np.log10(self.enclosed3DTotalMass(Re_kpc)[0])
        if MassRe['dark'] is not None:
            MassRe['dark'] = np.log10(MassRe['dark'][0])
            MassRe['fdm'] = 10**(MassRe['dark']-MassRe['total'])
        else:
            MassRe['fdm'] = None
        self.profiles['MassRe'] = MassRe

    def save(self, fname='profiles.dat', outpath='.'):
        with open('{}/{}'.format(outpath, fname), 'wb') as f:
            pickle.dump(self.profiles, f)

    def enclosed2DStellarMass(self, R):
        '''
        R in Kpc, could be a 1D array
        '''
        R = np.atleast_1d(R) * 1e3
        mass = np.zeros_like(R)
        for i in range(len(R)):
            mass[i] = self.LmMge.enclosed2Dluminosity(R[i])
        return self.ml * mass

    def enclosed3DStellarMass(self, r):
        '''
        r in Kpc, could be a 1D array
        '''
        r = np.atleast_1d(r) * 1e3
        mass = np.zeros_like(r)
        for i in range(len(r)):
            mass[i] = self.LmMge.enclosed3Dluminosity(r[i])
        return self.ml * mass

    def enclosed2DdarkMass(self, R):
        if self.DmMge is not None:
            R = np.atleast_1d(R) * 1e3
            mass = np.zeros_like(R)
            for i in range(len(R)):
                mass[i] = self.DmMge.enclosed2Dluminosity(R[i])
            return mass
        else:
            print 'No dark matter halo in current model'
            return None

    def enclosed3DdarkMass(self, r):
        '''
        r in Kpc, could be a 1D array
        '''
        r = np.atleast_1d(r) * 1e3
        if self.DmMge is not None:
            mass = np.zeros_like(r)
            for i in range(len(r)):
                mass[i] = self.DmMge.enclosed3Dluminosity(r[i])
            return mass
        else:
            print 'No dark matter halo in current model'
            return None

    def enclosed2DTotalMass(self, R):
        R = np.atleast_1d(R) * 1e3
        mass = self.enclosed2DStellarMass(R)
        if self.DmMge is not None:
            mass += self.enclosed2DdarkMass(R)
        if self.data['bh'] is not None:
            mass += self.data['bh']
        return mass

    def enclosed3DTotalMass(self, r):
        mass = self.enclosed3DStellarMass(r)
        if self.DmMge is not None:
            mass += self.enclosed3DdarkMass(r)
        if self.data['bh'] is not None:
            mass += self.data['bh']
        return mass

    def plotProfiles(self, outpath='.', figname='profiles.png', Range=None,
                     true=None, nre=3.5, **kwargs):
        Re_kpc = self.data['Re_arcsec'] * self.pc / 1e3
        dataRange = np.percentile(np.sqrt(self.xbin**2+self.ybin**2), 95) * \
            self.pc / 1e3
        if Range is None:
            Range = [0.5, nre*Re_kpc]
        MassRe = self.profiles['MassRe']
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        fig.subplots_adjust(left=0.12, bottom=0.08, right=0.98,
                            top=0.98, wspace=0.1, hspace=0.1)
        r = self.profiles['r']
        ii = (r > Range[0]) * (r < Range[1])
        logr = np.log10(r[ii])
        # plot stellar density
        stellarDens = np.log10(self.profiles['stellar'][ii, :, 0])
        stellarMass = np.log10(self.profiles['stellar'][ii, :, 1])
        axes[0].plot(logr, stellarDens, 'y', alpha=0.1, **kwargs)
        axes[1].plot(logr, stellarMass, 'y', alpha=0.1, **kwargs)
        # plot dark density and total density
        if self.profiles['dark'] is not None:
            darkDens = np.log10(self.profiles['dark'][ii, :, 0])
            totalDens = np.log10(self.profiles['total'][ii, :, 0])
            darkMass = np.log10(self.profiles['dark'][ii, :, 1])
            totalMass = np.log10(self.profiles['total'][ii, :, 1])
            axes[0].plot(logr, darkDens, 'r', alpha=0.1, **kwargs)
            axes[0].plot(logr, totalDens, 'c', alpha=0.1, **kwargs)
            axes[1].plot(logr, darkMass, 'r', alpha=0.1, **kwargs)
            axes[1].plot(logr, totalMass, 'c', alpha=0.1, **kwargs)
        # axes[1].plot(np.log10(Re_kpc), MassRe['total'], 'ok')
        # axes[1].plot(np.log10(Re_kpc), MassRe['stellar'], 'oy')
        # axes[1].plot(np.log10(Re_kpc), MassRe['dark'], 'or')
        util_fig.set_labels(axes[0])
        util_fig.set_labels(axes[1])
        axes[1].set_xlabel(r'$\mathbf{log_{10}\ \ \! R \, \ [kpc]}$',
                           fontproperties=label_font)
        axes[1].set_ylabel(r'$\mathbf{log_{10}\ \ \! M(R) \, \  [M_{\odot}]}$',
                           fontproperties=label_font)
        axes[0].set_ylabel(r'$\mathbf{log_{10}\ \ \! \rho \,'
                           ' \ [M_{\odot}\ \ \! kpc^{-3}]}$',
                           fontproperties=label_font)
        if self.gas3d is not None:
            gas2d = util_mge.projection(self.gas3d, self.inc)
            GasMge = util_mge.mge(gas2d, self.inc)
            mass, density = _extractProfile(GasMge, r)
            GasProfile = np.zeros([1, len(r), 2])
            GasProfile[0, :, 0] = density
            GasProfile[0, :, 1] = mass
            self.profiles['gas'] = GasProfile
            axes[0].plot(logr, np.log10(density[ii]), 'b', alpha=0.8, **kwargs)
            axes[1].plot(logr, np.log10(mass[ii]), 'b', alpha=0.8, **kwargs)
        if true is not None:
            rtrue = true.get('r', None)
            if rtrue is None:
                raise RuntimeError('r must be provided for true profile')
            if 'stellarDens' in true.keys():
                axes[0].plot(np.log10(rtrue), true['stellarDens'], 'oy')
            if 'stellarMass' in true.keys():
                axes[1].plot(np.log10(rtrue), true['stellarMass'], 'oy')
            if 'darkDens' in true.keys():
                axes[0].plot(np.log10(rtrue), true['darkDens'], 'or')
            if 'darkMass' in true.keys():
                axes[1].plot(np.log10(rtrue), true['darkMass'], 'or')
            if 'totalDens' in true.keys():
                axes[0].plot(np.log10(rtrue), true['totalDens'], 'oc')
            if 'totalMass' in true.keys():
                axes[1].plot(np.log10(r), true['totalMass'], 'oc')
        for ax in axes:
            ax.set_xlim([np.min(logr), np.max(logr)])
            ax.axvline(np.log10(Re_kpc), ls="dashed", color='b', linewidth=2)
            ax.axvline(np.log10(dataRange), ls="dashed", color='g',
                       linewidth=2)
        axes[0].text(0.05, 0.05, '$\mathbf{M^*(R_e)}$: %4.2f'
                     % (MassRe['stellar']), transform=axes[0].transAxes,
                     fontproperties=text_font)
        axes[0].text(0.35, 0.05, '$\mathbf{M^T(R_e)}$: %4.2f'
                     % (MassRe['total']), transform=axes[0].transAxes,
                     fontproperties=text_font)
        axes[0].text(0.05, 0.25, '$\mathbf{M^*/L}$: %4.2f' % (self.ml),
                     transform=axes[0].transAxes, fontproperties=text_font)
        if MassRe['fdm'] is not None:
            axes[0].text(0.35, 0.25, '$\mathbf{f_{DM}(R_e)}$: %4.2f'
                         % (MassRe['fdm']), transform=axes[0].transAxes,
                         fontproperties=text_font)
        if MassRe['fdm'] is not None:
            axes[1].text(0.85, 0.05, 'Total', color='c',
                         transform=axes[1].transAxes, fontproperties=text_font)

            axes[1].text(0.85, 0.15, 'Dark', color='r',
                         transform=axes[1].transAxes, fontproperties=text_font)
            axes[1].text(0.85, 0.25, 'Stellar', color='y',
                         transform=axes[1].transAxes, fontproperties=text_font)
        fig.savefig('{}/{}'.format(outpath, figname), dpi=300)
