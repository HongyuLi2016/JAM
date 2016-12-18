#!/usr/bin/env python
import numpy as np
from contex import JAM
# import matplotlib.pyplot as plt
Mgas = 5e9  # [M_solar]
print 'Input gas mass: {:.3e} [M_solar]'.format(Mgas)
R1 = JAM.utils.util_gas.Mgas2R1(Mgas)  # convert to R1 [pc]
print 'R1: {:.2f}'.format(R1)
lhy = JAM.utils.util_gas.gas(R1)
print '3D mge for gas mass distribution'
print lhy.mge3d
print 'Gas mass predicted by this profile: {:.3e} [M_solar]'.format(lhy.Mgas)

gas2d = JAM.utils.util_mge.projection(lhy.mge3d, np.radians(85.0))
mge_gas = JAM.utils.util_mge.mge(gas2d, np.radians(85.0))
print mge_gas.enclosed3Dluminosity(R1*0.2)/1e10
'''
logMstar = np.linspace(9, 12, 10)
logMgas_dutton = JAM.utils.util_gas.Mstellar2Mgas_Dutton(logMstar)
logMgas_alpha = JAM.utils.util_gas.Mstellar2Mgas_Alpha(logMstar)
plt.plot(logMstar, logMgas_alpha, 'r')
plt.plot(logMstar, logMgas_dutton, 'b')
plt.show()
'''
