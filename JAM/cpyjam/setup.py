from distutils.core import setup, Extension
from Cython.Build import cythonize

ext_axi_rms = Extension("axi_rms",
                        sources=["axi_rms.pyx"],
                        library_dirs=['clib'],
                        libraries=['cjam'])

setup(name='Cpyjam',
      version='0.0',
      description='Python interface of cJAM',
      author='Hongyu Li',
      author_email='hyli@nao.cas.cn',
      ext_modules=cythonize([ext_axi_rms
                             ]))
