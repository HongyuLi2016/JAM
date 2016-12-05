from distutils.core import setup, Extension
from Cython.Build import cythonize

ext=Extension("mge1d_fit",
              sources=["mge1d_fit.pyx"],
              library_dirs=['.'],
              libraries=['mge1d_mpfit']
             )
setup(name='mge1d_fit',
      ext_modules=cythonize([ext]))
