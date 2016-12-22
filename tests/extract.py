#!/usr/bin/env python
import numpy as np
from contex import JAM

lhy = JAM.utils.util_extract.extract_basic('data', rstname='rst.dat',
                                           profilename='profiles.dat',
                                           modelfile='mock_gNFW_out.dat')
print lhy.rst['chi2dof']
print lhy.Re_kpc
stellarMass = lhy.stellarMass(lhy.Re_kpc)
darkMass = lhy.darkMass(lhy.Re_kpc)
totalMass = lhy.totalMass(lhy.Re_kpc)
fdm = lhy.fdm(lhy.Re_kpc)
print stellarMass
print darkMass
print totalMass
print fdm
