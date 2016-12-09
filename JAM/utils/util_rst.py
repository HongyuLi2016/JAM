#!/usr/bin/env python
import numpy as np
import pickle
from scipy.stats import gaussian_kde
from scipy import stats
import util_mge
import util_dm
import corner_plot
from cap_symmetrize_velfield import symmetrize_velfield
from velocity_plot import velocity_plot
from matplotlib import tri, colors
from matplotlib.patches import Circle
import matplotlib.font_manager
ticks_font =\
    matplotlib.font_manager.FontProperties(family='times new roman',
                                           style='normal', size=10,
                                           weight='bold', stretch='normal')
text_font =\
    matplotlib.font_manager.FontProperties(family='times new roman',
                                           style='normal', size=15,
                                           weight='bold', stretch='normal')
ticks_font1 =\
    matplotlib.font_manager.FontProperties(family='times new roman',
                                           style='normal', size=8,
                                           weight='bold', stretch='normal')
label_font =\
    matplotlib.font_manager.FontProperties(family='times new roman',
                                           style='normal', size=15,
                                           weight='bold', stretch='normal')


def set_labels(ax, rotate=False, font=ticks_font):
    for l in ax.get_xticklabels():
        if rotate:
            l.set_rotation(60)
        l.set_fontproperties(font)
    for l in ax.get_yticklabels():
        # if rotate:
        #    l.set_rotation(0)
        l.set_fontproperties(font)


def set_lim(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_aspect(1, anchor='C')
    if xlim[1]-xlim[0] > ylim[1]-ylim[0]:
        ax.set_xlim(xlim[::-1])
        ax.set_ylim(xlim)
    else:
        ax.set_xlim(ylim[::-1])
        ax.set_ylim(ylim)


def printParameters(names, values):
    temp = ['{}: {:.2f}  '.format(names[i], values[i])
            for i in range(len(names))]
    print ''.join(temp)


def printModelInfo(model):
    print '--------------------------------------------------'
    print 'Model Info'
    print 'pyJAM model run at {}'.format(model['date'])
    print 'Galaxy name: {}'.format(model.get('name', None))
    print 'Model type: {}'.format(model.get('type', None))
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


class modelRst:
    def __init__(self, name, path='.', burnin=0, best='median'):
        self.data = load(name, path=path)
        self.ndim = self.data['ndim']
        self.nwalkers = self.data['nwalkers']
        self.chain = self.data['rst']['chain']
        self.goodchains = self.data['rst']['goodchains']
        if self.goodchains.sum()/float(self.nwalkers) < 0.6:
            self.goodchains = np.ones_like(self.goodchains, dtype=bool)
            print 'Warning - goodchain fraction less than 0.6'
            print 'Acceptace fraction:'
            print self.data['rst']['acceptance_fraction']
        self.lnprob = self.data['rst']['lnprobability']
        self.flatchain = self.chain[self.goodchains,
                                    burnin:, :].reshape(-1, self.ndim)
        self.flatlnprob = self.lnprob[self.goodchains, burnin:].reshape(-1)
        self.medianPars = estimatePrameters(self.flatchain,
                                            flatlnprob=self.flatlnprob)
        self.meanPars = estimatePrameters(self.flatchain,
                                          flatlnprob=self.flatlnprob,
                                          method='mean')
        self.peakPars = estimatePrameters(self.flatchain,
                                          flatlnprob=self.flatlnprob,
                                          method='peak')
        self.maxPars = estimatePrameters(self.flatchain,
                                         flatlnprob=self.flatlnprob,
                                         method='max')
        switch = {'median': self.medianPars, 'mean': self.meanPars,
                  'peak': self.peakPars, 'max': self.maxPars}
        bestPars = switch[best]
        self.lum2d = self.data['lum2d']
        self.pot2d = self.data['pot2d']
        self.xbin = self.data['xbin']
        self.ybin = self.data['ybin']
        self.rms = self.data['rms'].clip(0.0, 330.0)
        self.goodbins = self.data['goodbins']
        self.symetrizedRms = self.rms.copy()
        self.symetrizedRms[self.goodbins] = \
            symmetrize_velfield(self.xbin[self.goodbins],
                                self.ybin[self.goodbins],
                                self.rms[self.goodbins])

        JAMmodel = self.data['JAM']

        if self.data['type'] == 'massFollowLight':
            inc = np.arccos(bestPars[0])
            beta = np.zeros(self.lum2d.shape[0]) + bestPars[1]
            ml = bestPars[2]
            self.rmsModel = JAMmodel.run(inc, beta, ml=ml)
            self.flux = JAMmodel.flux
            self.labels = [r'$\mathbf{cosi}$', r'$\mathbf{\beta}$',
                           r'$\mathbf{M/L}$']

    def printInfo(self):
        printModelInfo(self.data)

    def enclosed2DMass(self):
        pass

    def enclosed3DMass(self):
        pass

    def cornerPlot(self, figname='mcmc.png', outpath='.',
                   clevel=[0.683, 0.95, 0.997], truths='max', true=None,
                   hbins=30, color=[0.8936, 0.5106, 0.2553], vmap='dots',
                   xpos=0.65, ypos=0.58, size=0.2, symetrize=False,
                   residual=True, **kwargs):
        switch = {'median': self.medianPars, 'mean': self.meanPars,
                  'peak': self.peakPars, 'max': self.maxPars,
                  'true': true}
        truthsValue = switch[truths]
        kwargs['labels'] = kwargs.get('labels', self.labels)
        fig = corner_plot.corner(self.flatchain, clevel=clevel, hbins=hbins,
                                 truths=truthsValue, color=color,
                                 quantiles=[0.16, 0.5, 0.84], **kwargs)
        # plot velocity map
        if vmap in ['dots', 'map']:
            rDot = kwargs.get('rDot', 0.24)
            axes0a = fig.add_axes([xpos, ypos, size, size])
            axes0b = fig.add_axes([xpos, ypos+size, size, size])
            axes0b.set_yticklabels([])
            axes0b.set_xticklabels([])
            set_labels(axes0a)
            if residual:
                axes0c = fig.add_axes([xpos-size*1.3, ypos+size, size, size])
                axes0c.set_yticklabels([])
                axes0c.set_xticklabels([])

            if symetrize:
                rms = self.symetrizedRms
            else:
                rms = self.rms
            vmin, vmax = stats.scoreatpercentile(rms[self.goodbins],
                                                 [0.5, 99.5])
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            # plot mge contours
            dither = 1e-3 * np.random.random(len(self.xbin))
            triangles = tri.Triangulation(self.xbin+dither, self.ybin)
            axes0b.tricontour(triangles,
                              -2.5*np.log10(self.flux/np.max(self.flux)),
                              levels=(np.arange(0, 10)), colors='k')
            axes0a.tricontour(triangles,
                              -2.5*np.log10(self.flux/np.max(self.flux)),
                              levels=(np.arange(0, 10)), colors='k')
            # mark badbins
            for i in range(self.xbin[~self.goodbins].size):
                circle = Circle(xy=(self.xbin[~self.goodbins][i],
                                    self.ybin[~self.goodbins][i]),
                                fc='w', radius=rDot*0.8, zorder=10, lw=0.)
                axes0b.add_artist(circle)

            velocity_plot(self.xbin, self.ybin, rms, ax=axes0b,
                          text='$\mathbf{V_{rms}: Obs}$', size=rDot,
                          norm=norm, bar=False, vmap=vmap)
            velocity_plot(self.xbin, self.ybin, self.rmsModel,
                          ax=axes0a, text='$\mathbf{V_{rms}: JAM}$',
                          size=rDot, norm=norm, vmap=vmap)
            if residual:
                residualValue = self.rmsModel - rms
                vmax = \
                    stats.scoreatpercentile(abs(residualValue[self.goodbins])
                                            .clip(-100, 100.), 99.5)
                norm_residual = colors.Normalize(vmin=-vmax, vmax=vmax)
                velocity_plot(self.xbin, self.ybin, residualValue, ax=axes0c,
                              text='$\mathbf{Residual}$', size=rDot,
                              norm=norm_residual, vmap=vmap)
        else:
            raise ValueError('vmap {} not supported'.format(vmap))
        fig.savefig('{}/{}'.format(outpath, figname), dpi=300)

    def profiles(self):
        pass
