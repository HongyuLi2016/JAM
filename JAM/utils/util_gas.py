import numpy as np
import os
JAMPATH = os.environ.get('JAMPATH')
if JAMPATH is None:
    raise RuntimeError('Enviroment variable JAMPAHT must be set')
mge1d = np.genfromtxt('{}/utils/mge_gas_1d.dat'.format(JAMPATH))
mge1d_gas = np.genfromtxt('{}/utils/mge_gas_1d_exp.dat'.format(JAMPATH))


def Mstellar2Mgas_Alpha(logMstellar):
    '''
    input: log stellar mass
    return: log gas mass
    '''
    return 0.276*logMstellar + 7.042


def Mstellar2Mgas_Dutton(logMstellar):
    '''
    input: log stellar mass
    return: log gas mass
    '''
    logfrac = -0.47*(logMstellar - 10) - 0.27
    logMgas = np.log10(10**logfrac * 10**logMstellar)
    return logMgas


def Mgas2R1(Mgas):
    '''
    Mgas [M_solar]
    return R1 [pc]
    '''
    return np.sqrt(Mgas/12.88)  # J. Wang 2014 Eq.(3)


class gas(object):
    def __init__(self, R1, q=0.1):
        '''
        R1: HI character radius in pc (See J. Wang 2014 for more details)
        mge3d: 3d mge [M_solar/pc^3] [pc] [none]
        '''

        self.mge1d = mge1d.copy()
        # convert to 3d MGE
        self.mge3d = np.zeros([self.mge1d.shape[0], 3])
        self.mge3d[:, 0] = self.mge1d[:, 0] / (2.0 * np.pi * R1 *
                                               (self.mge1d[:, 1])**2 * q)
        self.mge3d[:, 1] = self.mge1d[:, 1] * R1
        self.mge3d[:, 2] = q
        # calculate gas mass
        self.Mgas = np.sum(self.mge3d[:, 0] *
                           (np.sqrt(2.0*np.pi) * self.mge3d[:, 1])**3 *
                           self.mge3d[:, 2])


class gas_exp(object):
    def __init__(self, Mgas, q=0.1):
        '''
        Mgas: gas mass in M_solar
        mge3d: 3d mge [M_solar/pc^3] [pc] [none]
        '''

        self.mge1d = mge1d_gas.copy()
        # convert to 3d MGE
        self.mge3d = np.zeros([self.mge1d.shape[0], 3])
        self.mge3d[:, 0] = \
            self.mge1d[:, 0] / (2.0*np.pi*1e3 *
                                (self.mge1d[:, 1])**2*q) * Mgas/1e10
        self.mge3d[:, 1] = self.mge1d[:, 1] * 1e3
        self.mge3d[:, 2] = q
        # calculate gas mass
        self.Mgas = np.sum(self.mge3d[:, 0] *
                           (np.sqrt(2.0*np.pi) * self.mge3d[:, 1])**3 *
                           self.mge3d[:, 2])
