#!/usr/bin/env python

from distutils.core import setup
setup(name='Paradise',
      version='0.2',
      description="Modelling of spectra with superposition of stellar population templates and fitting of emission lines",
      author='Bernd Husemann/Jakob Walcher',
      packages=['Paradise','Paradise/lib'],
      package_dir={'Paradise' : './', 'Paradise/lib' : 'lib'},
      requires=['pyfits','numpy','scipy','pymc','matplotlib'],
      scripts=['bin/ParadiseApp.py'])
