#!/usr/bin/env python
from contex import JAM
import JAM.utils.util_rst as util_rst


if __name__ == '__main__':
    # change dump.dat to whatever you get from mcmc_pyjam.py
    model = util_rst.modelRst('dump.dat', path='data', best='median')
    model.printInfo()
    model.cornerPlot(linewidths=2., xpos=0.725, vmap='map', outpath='data')
