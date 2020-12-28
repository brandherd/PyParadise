#!/usr/bin/env python3

from distutils.core import setup
setup(name='Paradise',
      version='0.2',
      description="Modelling of galaxy spectra as a superposition of stellar population templates and Gaussian components for emission lines",
      author='Bernd Husemann, Jakob Walcher',
      author_email='berndhusemann@gmx.de, jwalcher@aip.de',
      license='MIT',
      url='https://github.com/brandherd/PyParadise',
      packages=['PyParadise'],
      install_requires=['numpy',
                        'scipy',
                        'astropy',
                        'pymc',
                        'emcee',
                        'matplotlib'],
      scripts=['bin/ParadiseApp.py','bin/ParadisePlot.py'])
