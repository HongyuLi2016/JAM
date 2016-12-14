#!/usr/bin/env python
'''
#############################################################################
Author: Hongyu Li
E-mail: hyli@nao.cas.cn

Revised version of JAM model from Cappellari (2008), with prolate capability

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.

#############################################################################
'''
import numpy as np
from scipy import special, signal, ndimage
from cap_quadva import quadva
from time import time, localtime, strftime
from warnings import simplefilter

simplefilter('ignore', RuntimeWarning)


def bilinear_interpolate(xv, yv, im, xout, yout, fill_value=0):
    """
    The input array has size im[ny,nx] as in the output
    of im = f(meshgrid(xv, yv))
    xv and yv are vectors of size nx and ny respectively.
    map_coordinates is equivalent to IDL's INTERPOLATE.

    """
    ny, nx = np.shape(im)
    if (nx, ny) != (xv.size, yv.size):
        raise ValueError("Input arrays dimensions do not match")

    xi = (nx-1.)/(xv[-1] - xv[0]) * (xout - xv[0])
    yi = (ny-1.)/(yv[-1] - yv[0]) * (yout - yv[0])

    return ndimage.map_coordinates(im.T, [xi, yi], cval=fill_value, order=1)


def _mge_surf(mge2d, x, y, shape='oblate'):
    rst = 0.0
    if shape == 'oblate':
        for i in range(mge2d.shape[0]):
            rst += mge2d[i, 0] * np.exp(-0.5/mge2d[i, 1]**2 *
                                        (x**2 + (y/mge2d[i, 2])**2))
    elif shape == 'prolate':
        for i in range(mge2d.shape[0]):
            rst += mge2d[i, 0] * np.exp(-0.5/mge2d[i, 1]**2 *
                                        ((-y)**2 + (x/mge2d[i, 2])**2))
    return rst


def _powspace(xmin, xmax, num, index=0.5):
    '''
    linear sample in power law space, with power low index specified by
    index keyworks
    '''
    return np.linspace(xmin**index, xmax**index, num)**(1.0/index)


def check_invalid_values(x):
    return np.isnan(x).sum() + np.isinf(x).sum()


def _integrand(u1,
               dens_lum, sigma_lum, q_lum,
               dens_pot, sigma_pot, q_pot,
               x1, y1, inc, beta, tensor):
    """
    This routine computes the integrand of Eq.(28) of Cappellari (2008; C08)
    for a model with constant anisotropy sigma_R**2 = b*sigma_z**2 and
    <V_R*V_z> = 0. The components of the proper motion dispersions tensor are
    calculated as described in note 5 of C08.
    See Cappellari (2012; C12 http://arxiv.org/abs/1211.7009)
    for explicit formulas for the proper motion tensor.

    """
    dens_lum = dens_lum[:, None, None]
    sigma_lum = sigma_lum[:, None, None]
    q_lum = q_lum[:, None, None]
    beta = beta[:, None, None]

    dens_pot = dens_pot[None, :, None]
    sigma_pot = sigma_pot[None, :, None]
    q_pot = q_pot[None, :, None]

    u = u1[None, None, :]

    kani = 1./(1 - beta)  # Anisotropy ratio b = (sig_R/sig_z)**2
    ci = np.cos(inc)
    si = np.sin(inc)
    si2 = si**2
    ci2 = ci**2
    x2 = x1**2
    y2 = y1**2
    u2 = u**2

    s2_lum = sigma_lum**2
    q2_lum = q_lum**2
    e2_lum = 1 - q2_lum
    s2q2_lum = s2_lum*q2_lum

    s2_pot = sigma_pot**2
    e2_pot = 1 - q_pot**2

    # Double summation over (j, k) of eq.(28) for all values of integration
    #   variable u.
    # The triple loop in (j, k, u) is replaced by broadcast Numpy array
    #   operations.
    # The j-index refers to the Gaussians describing the total mass,
    # from which the potential is derived, while the k-index is used
    # for the MGE components describing the galaxy stellar luminosity.

    e2u2_pot = e2_pot*u2
    a = 0.5*(u2/s2_pot + 1/s2_lum)               # equation (29) in C08
    b = 0.5*(e2u2_pot*u2/(s2_pot*(1 - e2u2_pot)) +
             e2_lum/s2q2_lum)  # equation (30) in C08
    c = e2_pot - s2q2_lum/s2_pot                  # equation (22) in C08
    d = 1 - kani*q2_lum - ((1 - kani)*c +
                           e2_pot*kani)*u2  # equation (23) in C08
    e = a + b*ci2
    if tensor == 'xx':
        f = kani*s2q2_lum + d*((y1*ci*(a+b)/e)**2 +
                               si2/(2*e))  # equation (4) in C12
    elif tensor == 'yy':
        f = s2q2_lum*(si2 + kani*ci2) + d*x2*ci2  # equation (5) in C12
    elif tensor == 'zz':
        f = s2q2_lum*(ci2 + kani*si2) + d*x2*si2  # z' LOS equation (28) in C08
    elif tensor == 'xy':
        f = -d*np.abs(x1*y1)*ci2*(a+b)/e          # equation (6) in C12
    elif tensor == 'xz':
        f = d*np.abs(x1*y1)*si*ci*(a+b)/e         # equation (7) in C12
    elif tensor == 'yz':
        f = si*ci*(s2q2_lum*(1 - kani) - d*x2)   # equation (8) in C12
    else:
        raise RuntimeError('Incorrect TENSOR string')

    # arr has the dimensions (q_lum.size, q_pot.size, u.size)

    arr = q_pot*dens_pot*dens_lum*u2*f*np.exp(-a*(x2 + y2*(a + b)/e)) \
        / ((1. - c*u2)*np.sqrt((1. - e2u2_pot)*e))

    G = 0.00430237    # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA-10)]
    # lhy: This function has c version.
    return 4*np.pi**1.5*G*np.sum(arr, axis=(0, 1))


def _wvrms2(x_pc, y_pc, inc, lum3d_pc, pot3d_pc, beta, tensor):
    '''
    Calculate the surface brightness weighted vrms2 for all the points x, y
    --input parameters
    x_pc, y_pc: 1d array for x, y position in pc
    inc: inclination in rad
    lum3d_pc, pot3d_pc: N*3 array for mge coefficients, L in L_solar/pc^2,
                        sigma in pc
    beta: Anisotropy paramter vactor for each luminous Gaussian component
    tensor: moments to be calculated, usually zz (LOS)
    --output paramters:
    wvrms2: surface brightness weighted vrms2 (1d array with the same
            length as x_pc)
    '''
    wvrms2 = np.empty_like(x_pc)
    for i in range(len(x_pc)):
        '''
        lhy: Change _integrand function to enable scipy integration!
        wvrms2[i] = quad(_integrand, 0.0, 1.0,
                         args=(lum3d_pc[:, 0], lum3d_pc[:, 1],
                               lum3d_pc[:, 2], pot3d_pc[:, 0],
                               pot3d_pc[:, 1], pot3d_pc[:, 2],
                               x_pc[i], y_pc[i], inc, beta, tensor))
        '''
        wvrms2[i] = quadva(_integrand, [0., 1.],
                           args=(lum3d_pc[:, 0], lum3d_pc[:, 1], lum3d_pc[:, 2],
                                 pot3d_pc[:, 0], pot3d_pc[:, 1], pot3d_pc[:, 2],
                                 x_pc[i], y_pc[i], inc, beta, tensor))[0]
    # lhy This could be translated to c
    return wvrms2


def _psf(xbin, ybin, pixSize, sigmaPsf, step):
    '''
    calculate kernal and grid for psf, see equation (A6) of Cappellari (2008)
    --input parameters
    xbin, ybin: x, y position in arcsec for the interpolation grid, on which
                wvrms2 are calculated
    pixSize: Fiber size for IFU data
    sigmaPsf: Gaussian sigma for psf
    --output paramters
    x1(y1): nx(ny) array
    xCar(yCar): grid positon in arcsec for psf (ny*nx)
    kernel: psf kernel
    '''

    if step < 0.1:
        print 'Warning - too small step ({:.3f}) for psf grid,'\
            ' change to 0.1 arcsec'.format(step)
        step = 0.1

    mx = 3*np.max(sigmaPsf) + pixSize/np.sqrt(2)
    xmax = max(abs(xbin)) + mx
    ymax = max(abs(ybin)) + mx

    nx = np.ceil(xmax/step)
    ny = np.ceil(ymax/step)
    x1 = np.linspace(-nx, nx, 2*nx+1)*step
    y1 = np.linspace(-ny, ny, 2*ny+1)*step
    xCar, yCar = np.meshgrid(x1, y1, indexing='xy')

    nk = np.ceil(mx/step)
    kgrid = np.linspace(-nk, nk, 2*nk+1)*step
    xgrid, ygrid = np.meshgrid(kgrid, kgrid)
    dx = pixSize/2.0
    sp = np.sqrt(2.0)*sigmaPsf
    kernel = (special.erf((dx-xgrid)/sp) + special.erf((dx+xgrid)/sp)) \
        * (special.erf((dx-ygrid)/sp) + special.erf((dx+ygrid)/sp))
    kernel /= np.sum(kernel)
    return x1, y1, xCar, yCar, kernel, mx


def _deprojection(mge2d, inc, shape):
    '''
    deproject the 2d mges to 3d, using different fomula for different shape.
    mge2d: 2d mges
    inc: inclination in rad
    '''
    mge3d = np.zeros_like(mge2d)
    if shape == 'oblate':
        qintr = mge2d[:, 2]**2 - np.cos(inc)**2
        if np.any(qintr <= 0):
            raise RuntimeError('Inclination too low q < 0')
        qintr = np.sqrt(qintr)/np.sin(inc)
        if np.any(qintr < 0.05):
            raise RuntimeError('q < 0.05 components')
        dens = mge2d[:, 0]*mge2d[:, 2] /\
            (mge2d[:, 1]*qintr*np.sqrt(2*np.pi))
        mge3d[:, 0] = dens
        mge3d[:, 1] = mge2d[:, 1]
        mge3d[:, 2] = qintr
    elif shape == 'prolate':
        qintr = np.sqrt(1.0/mge2d[:, 2]**2 - np.cos(inc)**2)/np.sin(inc)
        if np.any(qintr > 10):
            raise RuntimeError('q > 10.0 conponents')
        sigmaintr = mge2d[:, 1]*mge2d[:, 2]
        dens = mge2d[:, 0] / (np.sqrt(2*np.pi)*mge2d[:, 1]*mge2d[:, 2]**2*qintr)
        mge3d[:, 0] = dens
        mge3d[:, 1] = sigmaintr
        mge3d[:, 2] = qintr
    return mge3d


class jam:
    '''
    Main calss for JAM modelling
    lum, pot: N*3 mge coefficients arrays for tracer density and luminous
              matter potential (dark matter potential should be provided
              separately when calling the run method). Usually lum=pot
              if there is no stellar mass-to-light ratio gradient.
    distance: distance in Mpc
    xbin, ybin: coordinates in arcsec at which one wants to compute the
                model predictions (x must be aligned with the major axis
                for both oblate and prolate model)
    mbh: blackhole mass in M_solar
    rms: observed root-mean-squared velocities
    erms: error of rms, if None, 5% constant error will be assumed.
    goodbins: good bins used in calculate chi2, bool array
    sigmapsf: gaussian psf in arcsec (MaNGA ~ 1.0 - 1.5 )
    pixsize: IFU fibersize (MaNGA - 2.0 arcsec)
    step: kernel and psf grid step
    nrad: number of grid in x axis
    nang: not used in this version
    rbh: blackhole mge sigma (do not change this parameter if unnecessary)
    tensor: moments to be calculated, zz for LOS
    index: _powspace parameter, see _powspace() function
    shape: luminous matter shape
    '''
    def __init__(self, lum, pot, distance, xbin, ybin, mbh=None, rms=None,
                 erms=None, goodbins=None, sigmapsf=0.0, pixsize=0.0,
                 step=None, nrad=25, nang=10, rbh=0.01, tensor='zz',
                 index=0.5, shape='oblate', quiet=False, **kwargs):
        # check invalid numbers in the input arrays
        if check_invalid_values(lum) > 0:
            print lum
            raise ValueError("Invalid number in lum_mge")
        if check_invalid_values(pot) > 0:
            print pot
            raise ValueError("Invalid number in pot_mge")
        if check_invalid_values(xbin) > 0:
            print xbin
            raise ValueError("Invalid number in xbin")
        if check_invalid_values(ybin) > 0:
            print ybin
            raise ValueError("Invalid number in ybin")
        if xbin.size != ybin.size:
            raise ValueError("xbin and ybin do not match")

        if rms is not None:
            if check_invalid_values(rms) > 0:
                print rms
                raise ValueError("Invalid number in rms")
            if erms is None:
                # Constant ~5% errors
                erms = np.full_like(rms, np.median(rms)*0.05)
            else:
                if check_invalid_values(erms) > 0:
                    print erms
                    raise ValueError("Invalid number in erms")
            if goodbins is None:
                goodbins = np.ones_like(rms, dtype=bool)
            elif goodbins.dtype != bool:
                raise ValueError("goodbins must be a boolean vector")
            if not (xbin.size == rms.size == erms.size == goodbins.size):
                raise ValueError("(rms, erms, goodbins) and (xbin, ybin)"
                                 " do not match")

        if step is None:
            step = max(pixsize/2.0, np.min(sigmapsf))/4.0
        # Characteristic MGE axial ratio in observed range
        w = lum[:, 2] < np.max(np.abs(xbin))
        if w.sum() < 3:
            qmed = np.median(lum[:, 2])
        else:
            qmed = np.median(lum[w, 2])
        nx = nrad
        ny = int(nx*qmed) if int(nx*qmed) > 10 else 10
        self.psfConvolution = (sigmapsf > 0.0) and (pixsize > 0.0)
        self.interpolation = self.psfConvolution or (len(xbin) > nx*ny)

        if shape not in ['oblate', 'prolate']:
            raise ValueError("Shape({}) must be oblate or prolate"
                             .format(shape))

        self.Index = index
        self.tensor = tensor
        # Constant factor to convert arcsec --> pc
        self.pc = distance*np.pi/0.648
        self.lum = lum
        self.pot = pot
        self.lum[:, 2] = self.lum[:, 2]
        self.pot[:, 2] = self.pot[:, 2]
        self.lum_pc = lum.copy()
        self.lum_pc[:, 1] *= self.pc
        self.pot_pc = pot.copy()
        self.pot_pc[:, 1] *= self.pc
        # self.lum3d_pc = np.zeros_likd(self.lum_pc)
        # self.pot3d_pc = np.zeros_likd(self.pot_pc)
        self.xbin = xbin
        self.ybin = ybin
        self.goodbins = goodbins
        self.rms = rms
        self.erms = erms

        if shape == 'oblate':
            self.xbin_pc = xbin*self.pc
            self.ybin_pc = ybin*self.pc
        elif shape == 'prolate':
            self.xbin_pc = ybin*self.pc  # rotate 90 degree if prolate
            self.ybin_pc = -xbin*self.pc
            temn = nx
            nx = ny
            ny = temn
        self.xbin_pcIndex = abs(self.xbin_pc)**self.Index
        self.ybin_pcIndex = abs(self.ybin_pc)**self.Index
        self.shape = shape
        # black hole mge
        if mbh is not None:
            sigmaBH_pc = rbh*self.pc
            densBH_pc = mbh/(np.sqrt(2*np.pi)*sigmaBH_pc)**3
            self.mge_bh = np.array([densBH_pc, sigmaBH_pc, 0.999]).reshape(1, 3)
        else:
            self.mge_bh = None

        if self.psfConvolution:
            self.xcar, self.ycar, self.xCar, self.yCar,\
                self.kernel, self.mx = _psf(self.xbin_pc/self.pc,
                                            self.ybin_pc/self.pc,
                                            pixsize, sigmapsf, step)
            self.xCar *= self.pc
            self.yCar *= self.pc
            self.xcar *= self.pc
            self.ycar *= self.pc
            self.xCarIndex = abs(self.xCar)**self.Index
            self.yCarIndex = abs(self.yCar)**self.Index
        else:
            self.mx = 0.0

        if self.interpolation:
            xmax = (abs(self.xbin_pc).max() + self.mx) * 1.01
            ymax = (abs(self.ybin_pc).max() + self.mx) * 1.01

            self.xgrid = _powspace(0.0, xmax, nx, index=index)
            self.ygrid = _powspace(0.0, ymax, ny, index=index)
            self.xgridIndex = self.xgrid**self.Index
            self.ygridIndex = self.ygrid**self.Index
            self.xGrid, self.yGrid = \
                map(np.ravel, np.meshgrid(self.xgrid, self.ygrid,
                                          indexing='xy'))
            self.xGirdIndex = self.xGrid**self.Index
            self.yGirdIndex = self.yGrid**self.Index
        if not quiet:
            print 'Initialize success!'
            print 'Model shape: {}'.format(self.shape)
            print 'Interpolation: {}'.format(self.interpolation)
            if self.interpolation:
                print 'Interpolation grid: {}*{}'.format(nx, ny)
                print 'Interpolation size: {:.1f}*{:.1f}'\
                    .format(xmax/self.pc, ymax/self.pc)
            print 'psfConvolution: {}'.format(self.psfConvolution)
            if self.psfConvolution:
                print 'sigmaPSF, pixSize: {:.2f}, {:.2f}'.format(sigmapsf,
                                                                 pixsize)

    def run(self, inc, beta, mge_dh=None, ml=1.0):
        '''
        inc: inclination in radian
        beta: anisotropy parameter, lenth n array (n=Number of luminous
              Gaussian)
        mge_dh: dark halo mge, N*3 array  [M_solar/pc^3]  [pc]  [none]
        ml: stellar mass-to-light ratio. self.pot will be scaled by this factor,
            but mge_dh and black halo mge will not be scaled!
        '''
        # scale the stellar potential by ml
        pot_pc = self.pot_pc.copy()
        pot_pc[:, 0] *= ml
        # deprojection
        self.lum3d_pc = _deprojection(self.lum_pc, inc, self.shape)
        self.pot3d_pc = _deprojection(pot_pc, inc, self.shape)

        if self.mge_bh is not None:
            self.pot3d_pc = np.append(self.mge_bh, self.pot3d_pc, axis=0)
        if mge_dh is not None:
            self.pot3d_pc = np.append(self.pot3d_pc, mge_dh, axis=0)
        if not self.interpolation:
            wvrms2 = _wvrms2(self.xbin_pc, self.ybin_pc, inc, self.lum3d_pc,
                             self.pot3d_pc, beta, self.tensor)
            surf = _mge_surf(self.lum_pc, self.xbin_pc,
                             self.ybin_pc, shape=self.shape)
            self.flux = surf
            self.rmsModel = np.sqrt(wvrms2/surf)
            if self.tensor in ('xy', 'xz'):
                self.rmsModel *= np.sign(self.xbin_pc*self.ybin_pc)
            return self.rmsModel
        else:
            wvrms2 = _wvrms2(self.xGrid, self.yGrid, inc, self.lum3d_pc,
                             self.pot3d_pc, beta, self.tensor)
            surf = _mge_surf(self.lum_pc, self.xGrid,
                             self.yGrid, shape=self.shape)
            self.flux = _mge_surf(self.lum_pc, self.xbin_pc,
                                  self.ybin_pc, shape=self.shape)
            if not self.psfConvolution:
                # interpolate to the input xbin, ybin
                tem = np.sqrt((wvrms2/surf).reshape(len(self.ygrid),
                                                    len(self.xgrid)))
                self.rmsModel =\
                    bilinear_interpolate(self.xgridIndex, self.ygridIndex, tem,
                                         self.xbin_pcIndex, self.ybin_pcIndex)
                if self.tensor in ('xy', 'xz'):
                    self.rmsModel *= np.sign(self.xbin_pc*self.ybin_pc)
                return self.rmsModel
            else:
                tem_wvrms2 = wvrms2.reshape(len(self.ygrid), len(self.xgrid))
                tem_surf = surf.reshape(len(self.ygrid), len(self.xgrid))
                wvrms2Car = \
                    bilinear_interpolate(self.xgridIndex, self.ygridIndex,
                                         tem_wvrms2, self.xCarIndex,
                                         self.yCarIndex)
                surfCar = bilinear_interpolate(self.xgridIndex, self.ygridIndex,
                                               tem_surf, self.xCarIndex,
                                               self.yCarIndex)
                wvrms2Car_psf = signal.fftconvolve(wvrms2Car, self.kernel,
                                                   mode='same')
                surfCar_psf = signal.fftconvolve(surfCar, self.kernel,
                                                 mode='same')
                tem = np.sqrt(wvrms2Car_psf/surfCar_psf)
                self.rmsModel =\
                    bilinear_interpolate(self.xcar, self.ycar, tem,
                                         self.xbin_pc, self.ybin_pc)
                if self.tensor in ('xy', 'xz'):
                    self.rmsModel *= np.sign(self.xbin_pc*self.ybin_pc)
                return self.rmsModel
