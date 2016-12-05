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
JAM root directory:

 * `cpyjam`: Contains the model souce codes (cython, to be developed)

 * `pyjam`: Contains the model souce codes (python)
    * `axi_rms.py`: main JAM modelling routine
    * `cap_quadva.py`: integrator

 * `tests`: Contains the test scripts
    * `oblate_test.py`: Comparison between the oblate model here and the original version (i.e. from Cappellari)
    * `prolate_test_spherical.py`: Comparison between oblate and prolate model in spherical case (q=1)
    
 * `utils`: Contains some useful utilities (e.g. mge, dark halo class)
    * `util_dm.py`: Class for dark matter halos. Including density profile, enclosed mass calculation and MGE
    approaximation.
    * `util_mge.py`: Class for MGE density profiles. Including surface birghtness, luminosity density, total
    luminosity calculation, 2D to 3D deprojection and 2D, 3D enclosed mass calculation.

* `mge1d`: Cython module for 1D mge fitting. See mge1d/install.txt for more information  about how
installation.
