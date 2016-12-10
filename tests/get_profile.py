#!/usr/bin/env python
from contex import JAM
import JAM.utils.util_profile as util_profile
import numpy as np
# import matplotlib.pyplot as plt
import JAM.utils.util_mge as util_mge
import JAM.utils.util_dm as util_dm
# import cProfile, StringIO, pstats


def main():
    '--------------------------------------------------'
    # create the true density profiles
    r = np.logspace(np.log10(0.6), np.log10(10.0), 20)
    mge2d, dist, xbin, ybin, rms, erms = np.load('data/mock_sgnfw_data.npy')
    mgeStellar = util_mge.mge(mge2d, inc=np.radians(85.0), dist=dist)
    profileStellar = np.zeros_like(r)
    for i in range(len(r)):
        profileStellar[i] = (3.8 * mgeStellar.meanDensity(r[i]*1e3)) * 1e9
    logrho_s = 5.8
    rs = 40.0
    gamma = -1.2
    dh = util_dm.gnfw1d(10**logrho_s, rs, gamma)
    profileDark = dh.densityProfile(r)
    profileTotal = profileDark + profileStellar
    true = {'stellarDens': np.log10(profileStellar),
            'darkDens': np.log10(profileDark),
            'totalDens': np.log10(profileTotal), 'r': r}
    '--------------------------------------------------'
    profile = util_profile.profile('mock_gNFW_out.dat', path='data', nlines=200)
    profile.plotProfiles(true=true)
    # print lines


if __name__ == '__main__':
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'cumulative'
    # sortby = 'tottime'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()
