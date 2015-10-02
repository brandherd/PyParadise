#!/usr/bin/env python

deps = ['numpy','scipy','pymc','matplotlib']
try:
    import pyfits
    deps.append('pyfits')
except ImportError:
    deps.append('astropy')

from distutils.core import setup
setup(name='Paradise',
      version='0.2',
      description="Modelling of spectra with superposition of stellar population templates and fitting of emission lines",
      author='Bernd Husemann/Jakob Walcher',
      packages=['Paradise','Paradise/lib'],
      package_dir={'Paradise' : './', 'Paradise/lib' : 'lib'},
      requires=deps,
      scripts=['bin/ParadiseApp.py'])
