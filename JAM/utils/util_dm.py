import numpy as np
from mge1d_fit import mge1d_fit
from scipy.integrate import quad

TWO_PI = 2.0 * np.pi
FOUR_PI = 4.0 * np.pi
SQRT_TOW_PI = np.sqrt(2.0*np.pi)


def _mge1dfit(r, rho, **kwargs):
    '''
    fit a 1D profile (e.g. a spherical dark halo) with MGEs and return the
      3D-deprojected mge coefficients
    important kwargs
      imax - maximum iteration steps (5-10 is enough)
      ngauss - number of Gaussians used in fitting (~10)
      rbound - range of the MGE sigmas (usually do not need to provide)
    return
      mge3d - N*3 3d gaussian coefficients
    '''
    err = r*0.0 + np.mean(rho) * 0.1
    mge = mge1d_fit(r, rho, err, **kwargs)
    mge3d = np.zeros([mge.shape[0], 3])
    mge3d[:, 0] = mge[:, 0]/SQRT_TOW_PI/mge[:, 1]
    mge3d[:, 1] = mge[:, 1]
    mge3d[:, 2] = 0.999
    return mge3d


class gnfw1d:
    def __init__(self, rho_s, rs, gamma):
        '''
        unit:
          rho_s [M_solar/kpc^3]
          rs [kpc]
          gamma [none], usually between [-2.0, 0], for NFW, gamma = -1.0
        '''
        self.rho_s = rho_s
        self.rs = rs
        self.gamma = gamma

    def densityProfile(self, r):
        '''
        Return the density values at given r
        r [kpc]
        densityProfile [M_solar/kpc^3]
        '''
        rdivrs = r / self.rs
        return self.rho_s*rdivrs**self.gamma *\
            (0.5+0.5*rdivrs)**(-self.gamma-3)

    def enclosedMass(self, R, **kwargs):
        '''
        Return the enlosed mass within R
        R [kpc]
        enclosedMass [M_solar]
        '''
        def _integrate(r):
            rdivrs = r / self.rs
            rst = self.rho_s*rdivrs**self.gamma *\
                (0.5+0.5*rdivrs)**(-self.gamma-3) * FOUR_PI * r * r
            return rst
        return quad(_integrate, 0.0, R, **kwargs)[0]

    def mge3d(self, rrange=[0.1, 200], ngauss=10, imax=5, npoints=200):
        '''
        rrange [kpc]
        mge3d [M_solar/pc^3]  [pc]  [none]
        '''
        r = np.logspace(np.log10(rrange[0]), np.log10(rrange[1]), npoints)
        rho = self.densityProfile(r)
        return _mge1dfit(r*1e3, rho/1e9, imax=imax, ngauss=ngauss)

    def mge2d(self, rrange=[0.1, 200], ngauss=10, imax=5, npoints=200):
        '''
        fit this halo profile using MGE method and return the MGE coefficents
        rrange [kpc], within which profile is fitted.
        mge2d [M_solar/pc^2]  [pc]  [none]
        '''
        mge3d = self.mge3d(rrange=rrange, ngauss=ngauss,
                           imax=imax, npoints=npoints)
        mge3d[:, 0] *= mge3d[:, 1]*SQRT_TOW_PI
        return mge3d


# do not use this
def gnfw(r, rho_s, rs, gamma):
    rdivrs = r / rs
    return rho_s*rdivrs**gamma * (0.5+0.5*rdivrs)**(-gamma-3)
