import numpy as np
from scipy.integrate import quad, dblquad
from scipy.misc import derivative

TWO_PI = 2.0 * np.pi
FOUR_PI = 4.0 * np.pi
SQRT_TOW_PI = np.sqrt(2.0*np.pi)


def _inte_2dencloseM(r, theta,
                     surf, twoSigma2, q2):
    cosTheta2 = np.cos(theta)**2
    rst = 0.0
    for i in range(len(surf)):
        rst += surf[i] * np.exp(- r*r / twoSigma2[i] *
                                (cosTheta2 + (1-cosTheta2)/q2[i])) * r
    return rst


def _inte_3dencloseM(r, u,
                     dens, twoSigma2, q2):
    rst = 0.0
    for i in range(len(twoSigma2)):
        rst += dens[i] * np.exp(- r*r / twoSigma2[i] *
                                (u*u + (1 - u*u) / q2[i])) * r*r
    return FOUR_PI * rst


def _Hu_inte(u,
             R, z,
             M, sigma, q2):
    rst = 0.0
    twoSigma2 = 2.0 * sigma * sigma
    for i in range(len(M)):
        rst += M[i] * np.exp(-u*u / twoSigma2[i] *
                             (R*R + z*z / (1.0 - (1.0 - q2[i]) * u*u))) /\
            np.sqrt(1.0 - (1.0 - q2[i]) * u*u) / sigma[i]
    return rst


class mge:
    def __init__(self, mge2d, inc, shape='oblate', dist=None):
        '''
        The default units are [L_solar/pc^2]  [pc]  [none]
        '''
        if dist is not None:
            pc = dist * np.pi / 0.648
            mge2d[:, 1] *= pc
        self.mge2d = mge2d
        self.inc = inc
        self.shape = shape
        self.ngauss = mge2d.shape[0]

    def deprojection(self):
        '''
        Return the 3D deprojected MGE coefficients
        '''
        mge3d = np.zeros_like(self.mge2d)
        if self.shape == 'oblate':
            qintr = self.mge2d[:, 2]**2 - np.cos(self.inc)**2
            if np.any(qintr <= 0):
                raise RuntimeError('Inclination too low q < 0')
            qintr = np.sqrt(qintr)/np.sin(self.inc)
            if np.any(qintr < 0.05):
                raise RuntimeError('q < 0.05 components')
            dens = self.mge2d[:, 0]*self.mge2d[:, 2] /\
                (self.mge2d[:, 1]*qintr*SQRT_TOW_PI)
            mge3d[:, 0] = dens
            mge3d[:, 1] = self.mge2d[:, 1]
            mge3d[:, 2] = qintr
        elif self.shape == 'prolate':
            qintr = np.sqrt(1.0/self.mge2d[:, 2]**2 -
                            np.cos(self.inc)**2)/np.sin(self.inc)
            if np.any(qintr > 10):
                raise RuntimeError('q > 10.0 conponents')
            sigmaintr = self.mge2d[:, 1]*self.mge2d[:, 2]
            dens = self.mge2d[:, 0] / (SQRT_TOW_PI*self.mge2d[:, 1] *
                                       self.mge2d[:, 2]**2*qintr)
            mge3d[:, 0] = dens
            mge3d[:, 1] = sigmaintr
            mge3d[:, 2] = qintr
        return mge3d

    def luminosity(self):
        '''
        Return the total luminosity of all the Gaussians (in L_solar)
        '''
        return (self.mge2d[:, 0]*TWO_PI*self.mge2d[:, 1]**2 *
                self.mge2d[:, 2]).sum()

    def luminosityDensity(self, R, z):
        '''
        Return the luminosity density at coordinate R, z (in L_solar/pc^3)
        '''
        rst = 0.0
        mge3d = self.deprojection()
        for i in range(self.ngauss):
            rst += mge3d[i, 0] * np.exp(-0.5/mge3d[i, 1]**2 *
                                        (R**2 + (z/mge3d[i, 2])**2))
        return rst

    def surfaceBrightness(self, x, y):
        '''
        Return the surface brightness at coordinate x, y (in L_solar/pc^2)
        x, y must have the same unit as mge2d (i.e. default as pc)
        x is the major axis for oblate and minor axis for prolate
        '''
        rst = 0.0
        if self.shape == 'prolate':
            for i in range(self.ngauss):
                rst += self.mge2d[i, 0] * np.exp(-0.5/self.mge2d[i, 1]**2 *
                                                 (y**2+(x/self.mge2d[i, 2])**2))
        elif self.shape == 'oblate':
                rst += self.mge2d[i, 0] * np.exp(-0.5/self.mge2d[i, 1]**2 *
                                                 (x**2+(y/self.mge2d[i, 2])**2))
        return rst

    def enclosed3Dluminosity(self, r):
        '''
        Return the 3D enclosed luminosity within a sphere r (in L_solar)
        input r should be in Kpc
        '''
        r *= 1e3
        mge3d = self.deprojection()
        dens = mge3d[:, 0]
        twoSigma2 = 2.0 * mge3d[:, 1]**2
        q2 = mge3d[:, 2]**2
        I = dblquad(_inte_3dencloseM, 0.0, 1.0, lambda x: 0.0,
                    lambda x: r, args=(dens, twoSigma2, q2))
        return I[0]

    def enclosed2Dluminosity(self, R):
        '''
        Return the 2D enclosed luminosity within a circular aperture R
          (in L_solar)
        input R should be in Kpc
        '''
        R *= 1e3  # convert kpc to pc
        surf = self.mge2d[:, 0]
        twoSigma2 = 2.0 * self.mge2d[:, 1]**2
        q2 = self.mge2d[:, 2]**2
        I = dblquad(_inte_2dencloseM, 0.0, 0.5*np.pi, lambda x: 0.0,
                    lambda x: R, args=(surf, twoSigma2, q2))
        return I[0] * 4.0

    def Phi(self, R, z):
        '''
        Return the gravitational potential at R, z
        '''
        G = 0.00430237
        mge3d = self.deprojection()
        sigma = mge3d[:, 1]
        q2 = mge3d[:, 2]**2
        M = mge3d[:, 0] * (SQRT_TOW_PI*sigma)**3 * mge3d[:, 2]
        I = quad(_Hu_inte, 0, 1, args=(R, z, M, sigma, q2))
        return -np.sqrt(2.0/np.pi)*G*I[0]

    def Vc(self, R, ml=1.0):
        '''
        Return the rotational velocity at R on the equatorial plane (in km/s)
        R in kpc
        ml mass-to-light ratio of these Gaussians
        '''
        R *= 1e3
        Force = derivative(self.Phi, R, args=([0]))
        return np.sqrt(ml*Force*R)
