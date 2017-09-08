#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : analysis_fitTP.py
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
from contex import JAM
from optparse import OptionParser


def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()
    if len(args) != 1:
        print('Error - please provide a folder name')
        exit(1)
    path = args[0]
    modelRst = \
        JAM.utils.util_rst_fitTP.modelRst('mcmc_tprofile.dat', path=path)
    modelRst.plotChain(outpath=path)
    modelRst.plotProfile(outpath=path)
    modelRst.cornerPlot(outpath=path, resample=False)


if __name__ == '__main__':
    main()
