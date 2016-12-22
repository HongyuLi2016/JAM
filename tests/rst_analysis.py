#!/usr/bin/env python
from contex import JAM
import JAM.utils.util_rst as util_rst


if __name__ == '__main__':
    # change dump.dat to whatever you get from mcmc_pyjam.py
    # model = util_rst.modelRst('dump.dat', path='data', best='median')
    # model = util_rst.modelRst('mock_massFollowLight_out.dat', path='data',
    #                          best='median')
    model = util_rst.modelRst('mock_gNFW_out.dat', path='data', best='median')
    # model.printInfo()
    # model.printPrior()
    # model.cornerPlot(linewidths=2., xpos=0.725, vmap='dots', outpath='data')
    # model.plotChain(outpath='data')
    # model.plotVrms(outpath='data')
    model.dump(outpath='data')
    # print model.meanDisp(1.0)
