#!/usr/bin/env python
import numpy as np
from scipy.interpolate import interp1d
import pickle


class extract_basic:
    def __init__(self, filepath, rstname=None,
                 profilename=None, modelfile=None):
        if rstname is not None:
            with open('{}/{}'.format(filepath, rstname), 'rb') as f:
                self.rst = pickle.load(f)
            self.Re_kpc =\
                self.rst['Re_arcsec'] * self.rst['dist'] * np.pi / 0.648 / 1e3
        if profilename is not None:
            with open('{}/{}'.format(filepath, profilename), 'rb') as f:
                self.profile = pickle.load(f)
            self.r = self.profile['r']
            self.stellar = self.profile['stellar']
            self.dark = self.profile['dark']
            self.total = self.profile['total']

        if modelfile is not None:
            with open('{}/{}'.format(filepath, modelfile), 'rb') as f:
                self.model = pickle.load(f)

    def stellarMass(self, r):
        '''
        r: scalar or 1d array
        Return the stellar mass and the 1sigma error within r
        '''
        r = np.atleast_1d(r)
        self.stellarPer = \
            np.percentile(np.log10(self.stellar[:, :, 1]), [16, 50, 84], axis=1)
        fmed = interp1d(self.r, self.stellarPer[1, :])
        ferrlow = interp1d(self.r, self.stellarPer[0, :])
        ferrupp = interp1d(self.r, self.stellarPer[2, :])
        mass = np.zeros([len(r), 3])
        mass[:, 0] = ferrlow(r) - fmed(r)
        mass[:, 1] = fmed(r)
        mass[:, 2] = ferrupp(r) - fmed(r)
        return mass

    def darkMass(self, r):
        '''
        r: scalar or 1d array
        Return the dark mass and the 1sigma error within r
        '''
        r = np.atleast_1d(r)
        mass = np.zeros([len(r), 3])
        if self.dark is None:
            mass[:, :] = np.nan
        else:
            self.darkPer = \
                np.percentile(np.log10(self.dark[:, :, 1]),
                              [16, 50, 84], axis=1)
            fmed = interp1d(self.r, self.darkPer[1, :])
            ferrlow = interp1d(self.r, self.darkPer[0, :])
            ferrupp = interp1d(self.r, self.darkPer[2, :])
            mass[:, 0] = ferrlow(r) - fmed(r)
            mass[:, 1] = fmed(r)
            mass[:, 2] = ferrupp(r) - fmed(r)
        return mass

    def totalMass(self, r):
        '''
        r: scalar or 1d array
        Return the total mass and the 1sigma error within r
        '''
        r = np.atleast_1d(r)
        self.totalPer = \
            np.percentile(np.log10(self.total[:, :, 1]), [16, 50, 84], axis=1)
        fmed = interp1d(self.r, self.totalPer[1, :])
        ferrlow = interp1d(self.r, self.totalPer[0, :])
        ferrupp = interp1d(self.r, self.totalPer[2, :])
        mass = np.zeros([len(r), 3])
        mass[:, 0] = ferrlow(r) - fmed(r)
        mass[:, 1] = fmed(r)
        mass[:, 2] = ferrupp(r) - fmed(r)
        return mass

    def fdm(self, r):
        '''
        r: scalar or 1d array
        Return the dark matter fraction and the 1sigma error within r
        '''

        r = np.atleast_1d(r)
        dm_frac = np.zeros([len(r), 3])
        if self.dark is None:
            dm_frac[:, :] = np.nan
        else:
            fdm_profile = self.dark[:, :, 1]/self.total[:, :, 1]
            self.fdmPer = \
                np.percentile(fdm_profile, [16, 50, 84], axis=1)
            fmed = interp1d(self.r, self.fdmPer[1, :])
            ferrlow = interp1d(self.r, self.fdmPer[0, :])
            ferrupp = interp1d(self.r, self.fdmPer[2, :])

            dm_frac[:, 0] = ferrlow(r) - fmed(r)
            dm_frac[:, 1] = fmed(r)
            dm_frac[:, 2] = ferrupp(r) - fmed(r)
        return dm_frac
