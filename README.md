# JAM(Jeans Anisotropic MGE model)
=====
Version 0.0 

revised JAM modelling code, including prolate capability.
For the original version, please see [here](http://www-astro.physics.ox.ac.uk/~mxc/software/)

=====
References
-----
Please see the following papers for more describtion about the method:
 * Cappellari M. 2008, MNRAS, 390, 71
 * Li H. et al 2016, MNARS, 455, 3680

Installation
----------
If you have git installed, JAM can be obtained with the following commands:
```
cd /path/to/desired/location/
git clone https://github.com/HongyuLi2016/JAM
```

Contents
--------
Below is a brief description of the contents of the directories in the
 root directory:
 
 * `JAM`:  Main package for JAM modelling.

 * `JAM/cpyjam`: Contains the model souce codes (cython, to be developed)

 * `JAM/pyjam`: Contains the model souce codes (python)
    * `axi_rms.py`: main JAM modelling routine
    * `cap_quadva.py`: integrator
  
 * `JAM/mcmc`: Contiains MCMC wrappers for JAM. 
    * `mcmc_pyjam.py`: Run emcee for a mass-follow-light model or spherical gNFW model and output
    mcmc chains. (Currently only oblate JAM is tested, see test/mcmc_example_*.py for instructions)
    
 * `JAM/utils`: Contains some useful utilities (e.g. mge, dark halo class)
    * `util_dm.py`: Class for dark matter halos. Including density profile, enclosed mass calculation and MGE
    approaximation.
    * `util_mge.py`: Class for MGE density profiles. Including surface birghtness, luminosity density, total
    luminosity calculation, 2D to 3D deprojection and 2D, 3D enclosed mass calculation.
    * `corner_plot.py`: Make corner plots.
    * `velocity_plot.py`: Plot velocity maps. (Both map or dotted plots are supported)
    * `vprofile.py`: Plot velocity profiles alone some position angle.
    * `util_fig.py`: Some useful functions for figure plotting.
    * `util_rst.py`: Analyze the output chains from MCMC.

--------
Some other packages and test scripts
* `mge1d`: Cython module for 1D mge fitting. See mge1d/install.txt for more information  about installation.

* `tests`: Contains the test scripts
    * `oblate_test.py`: Comparison between the oblate model here and the original version (i.e. from Cappellari)
    * `prolate_test_spherical.py`: Comparison between oblate and prolate model in spherical case (q=1)
    * `mcmc_example_massFollowLight.py`: Fit a mass-follow-light JAM model to mock data with emcee.
    * `mcmc_example_gnfw.py`: Fit a spherical gNFW JAM model to mock data with emcee.
    * `rst_analysis.py`: Use util_rst.py to plot some figures and analyze results.
    * `mge_example.py`: Simple examples about how to use mge/dm utils.
