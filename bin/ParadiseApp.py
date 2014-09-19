#!/usr/bin/env python

__version__ = "0.1"

import sys
import argparse
import time
import os
import pyfits
from Paradise import *
import copy_reg


class ParadiseApp(object):
    """The class `ParadiseApp` handles the fitting of the spectrum with
    a linear combination of template spectra and/or emission lines. For the
    the fitting of the template spectra, velocity and the velocity
    dispersion are obtained by Monte Carlo Markov Chains and the linear
    combination of template spectra with a non-negative least-squares
    algorithm.

    The emission line fitting is a separate function, and fits a provided
    set of emission lines.

    The bootstrapping fitting function can be used to obtain errors on the
    parameters for the template spectral fitting and errors on the
    parameters of the emission lines.
    """
    def __init__(self, input_file, outprefix, instrFWHM):
        """
        Parameters
        ----------
        input_file : str
            The fits file containing the spectrum for which the fitting will be
            performed
        outprefix : str
            The prefix that will be prepended to the files that are written by
            the fitting routines.
        instrFWHM : float
            The instrumental resolution of the data in `input_file`.
        """
        self.__inputData = loadCube(input_file)
        if self.__inputData._datatype == 'CUBE':
            self.__inputData.correctError()
            self.__datatype = 'CUBE'
        elif self.__inputData._datatype == 'RSS':
            self.__inputData = loadRSS(input_file)
            self.__inputData.correctError()
            self.__datatype = 'RSS'
        self.__outPrefix = outprefix
        self.__instrFWHM = instrFWHM

    def run_SSP_fit(self, parfile, parallel, verbose):
        """This functions fits a linear combination of template spectra to the
        input spectra to obtain the best fit.

        Parameters
        ----------
        parfile : str
            The parameter file containing the constraints under which the input
            spectra will be fitted.
        parallel : {'auto', int}, optional
            If parallel is not equal to one, the python multiprocessing routine
            shall be used to run parts of the code in parallel. With the option
            `auto`, it adjusts the number of parallel processes to the number
            of cpu-cores/threads that are available.
        verbose : bool, optional
            Produces screen output, such that the user can follow which spectrum
            is currently fitted and what the results are for each spectrum.

        Notes
        -----
        The function does not return anything, but writes the results of the fit
        to disk. There are three files written to disk, all prepended by the
        `outprefix` supplied to this class at creation. The three files are:
        - `outprefix`.cont_model.fits which contains the spectra corresponding
          to the best linear combination of spectra.
        - `outprefix`.cont_res.fits contains the residual between the input fits
          file and the best fitted model spectra.
        - `outprefix`.stellar_table.fits contains the parameters of the best
          fit, like velocity, velocity dispersion, the luminosity-weighted and
          mass-weighted parameters.
        """
        ## Read in parameter file and get values
        parList = ParameterList(parfile)
        nlib_guess = int(parList['tmplinitspec'].getValue())
        vel_guess = float(parList['vel_guess'].getValue())
        vel_min = float(parList['vel_min'].getValue())
        vel_max = float(parList['vel_max'].getValue())
        disp_min = float(parList['disp_min'].getValue())
        disp_max = float(parList['disp_max'].getValue())
        kin_fix = bool(float(parList['kin_fix'].getValue()))
        nwidth_norm = int(parList['nwidth_norm'].getValue())
        iterations = int(parList['iterations'].getValue())
        samples = int(parList['samples'].getValue())
        burn = int(parList['burn'].getValue())
        thin = int(parList['thin'].getValue())
        start_wave = float(parList['start_wave'].getValue())
        end_wave = float(parList['end_wave'].getValue())
        excl_fit = CustomMasks(parList['excl_fit'].getValue())
        excl_cont = CustomMasks(parList['excl_cont'].getValue())
        min_x = int(parList['min_x'].getValue())
        max_x = int(parList['max_x'].getValue())
        min_y = int(parList['min_y'].getValue())
        max_y = int(parList['max_y'].getValue())

        if verbose:
            print "The stellar population library is being prepared."
        lib = SSPlibrary(filename=parList['tmpldir'].getValue() + '/' + parList['tmplfile'].getValue())
        if kin_fix is True:
            hdu = pyfits.open(self.__outPrefix + '.kin_table.fits')
            tab = hdu[1].data
            vel_fit = tab.field('vel_fit')
            disp_fit = tab.field('disp_fit')
            vel_fit_err = tab.field('vel_fit_err')
            disp_fit_err = tab.field('disp_fit_err')
            vel_min = numpy.min(vel_fit)
            vel_max = numpy.max(vel_fit)
        min_wave = (self.__inputData.getWave()[0] / (1 + (vel_min - 2000) / 300000.0))
        max_wave = (self.__inputData.getWave()[-1] / (1 + (vel_max + 2000) / 300000.0))

        if nlib_guess < 0:
            select = numpy.arange(lib.getBaseNumber()) == nlib_guess * -1 - 1
            lib = lib.subLibrary(select)
        lib = lib.subWaveLibrary(min_wave=min_wave, max_wave=max_wave)
        lib = lib.matchInstFWHM(self.__instrFWHM, vel_guess)
        lib = lib.resampleWaveStepLinear(self.__inputData.getWaveStep(), vel_guess / 300000.0)
        lib_norm = lib.normalizeBase(nwidth_norm, excl_cont, vel_guess / 300000.0)
        lib_rebin = lib_norm.rebinLogarithmic()

        if verbose:
            print "The input cube is being normalized."
        normData = self.__inputData.normalizeSpec(nwidth_norm, excl_cont.maskPixelsObserved(self.__inputData.getWave(),
             vel_guess / 300000.0))
        normDataSub = normData.subWaveLimits(start_wave, end_wave)

        if verbose:
            print "The stellar population modelling has been started."
        if self.__datatype == 'CUBE':
            if kin_fix:
                (fitted, coeff, chi2, x_pix, y_pix, cube_model) = normDataSub.fit_Lib_fixed_kin(lib_rebin, vel_fit,
                disp_fit, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, mask_fit=excl_fit.maskPixelsObserved(
                normDataSub.getWave(), vel_guess / 300000.0), iterations=iterations, burn=burn, samples=samples, thin=thin,
                verbose=verbose, parallel=parallel)
            else:
                (vel_fit, vel_fit_err, disp_fit, disp_fit_err, fitted, coeff, chi2, x_pix, y_pix,
                cube_model) = normDataSub.fit_Kin_Lib_simple(lib_rebin, nlib_guess, vel_min, vel_max, disp_min, disp_max,
                min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, mask_fit=excl_fit.maskPixelsObserved(
                normDataSub.getWave(), vel_guess / 300000.0), iterations=iterations, burn=burn, samples=samples, thin=thin,
                verbose=verbose, parallel=parallel)
        elif self.__datatype == 'RSS':
            if kin_fix:
                fibers = tab.field('fiber')
                (fitted, coeff, chi2, fiber, rss_model) = normDataSub.fit_Lib_fixed_kin(lib_rebin, vel_fit, disp_fit,
                fibers, min_y=min_y, max_y=max_y, mask_fit=excl_fit.maskPixelsObserved(normDataSub.getWave(),
                vel_guess / 300000.0), verbose=verbose, parallel=parallel)
            else:
                (vel_fit, vel_fit_err, disp_fit, disp_fit_err, fitted, coeff, chi2, fiber,
                rss_model) = normDataSub.fit_Kin_Lib_simple(lib_rebin, nlib_guess, vel_min, vel_max, disp_min, disp_max,
                min_y=min_y, max_y=max_y, mask_fit=excl_fit.maskPixelsObserved(normDataSub.getWave(),
                vel_guess / 300000.0), iterations=iterations, burn=burn, samples=samples, thin=thin,
                verbose=verbose, parallel=parallel)

        if verbose:
                print "Storing the results to %s (model), %s (residual) and %s (parameters)." % (
                    self.__outPrefix + '.cont_model.fits', self.__outPrefix + '.cont_res.fits',
                    self.__outPrefix + '.stellar_table.fits')
        if self.__datatype == 'RSS':
            model_out = RSS(wave=self.__inputData.subWaveLimits(start_wave, end_wave)._wave, data=rss_model,
                header=self.__inputData.getHeader())
            res_out = RSS(wave=self.__inputData.subWaveLimits(start_wave, end_wave)._wave,
                data=self.__inputData.subWaveLimits(start_wave, end_wave)._data - rss_model, header=self.__inputData.getHeader())
        elif self.__datatype == 'CUBE':
            model_out = Cube(wave=self.__inputData.subWaveLimits(start_wave, end_wave)._wave, data=cube_model,
                header=self.__inputData.getHeader())
            res_out = Cube(wave=self.__inputData.subWaveLimits(start_wave, end_wave)._wave,
                data=self.__inputData.subWaveLimits(start_wave, end_wave)._data - cube_model, header=self.__inputData.getHeader())
        model_out.writeFitsData(self.__outPrefix + '.cont_model.fits')
        res_out.writeFitsData(self.__outPrefix + '.cont_res.fits')

        mass_weighted_pars = numpy.zeros((len(fitted), 5), dtype=numpy.float32)
        lum_weighted_pars = numpy.zeros((len(fitted), 5), dtype=numpy.float32)
        for i in range(len(fitted)):
            if fitted[i]:
                try:
                    mass_weighted_pars[i, :] = lib_norm.massWeightedPars(coeff[i, :])
                except:
                    mass_weighted_pars[i, :] = numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan])
                try:
                    lum_weighted_pars[i, :] = lib_norm.lumWeightedPars(coeff[i, :])
                except:
                    lum_weighted_pars[i, :] = numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan])

        columns = []
        if self.__datatype == 'CUBE':
            columns.append(pyfits.Column(name='x_cor', format='J', array=x_pix[fitted]))
            columns.append(pyfits.Column(name='y_cor', format='J', array=y_pix[fitted]))
        else:
            columns.append(pyfits.Column(name='fiber', format='J', array=fiber[fitted]))
        columns.append(pyfits.Column(name='vel_fit', format='E', unit='km/s', array=vel_fit[fitted]))
        columns.append(pyfits.Column(name='vel_fit_err', format='E', unit='km/s', array=vel_fit_err[fitted]))
        columns.append(pyfits.Column(name='disp_fit', format='E', unit='km/s', array=disp_fit[fitted]))
        columns.append(pyfits.Column(name='disp_fit_err', format='E', unit='km/s', array=disp_fit_err[fitted]))
        columns.append(pyfits.Column(name='chi2', format='E', array=chi2[fitted]))
        if lib.getBaseNumber() > 1:
            columns.append(pyfits.Column(name='base_coeff', format='%dE' % (lib.getBaseNumber()), array=coeff[fitted, :]))
        else:
            columns.append(pyfits.Column(name='base_coeff', format='E', array=coeff[fitted].flatten()))
        columns.append(pyfits.Column(name='lum_coeff_frac_total', format='E', array=lum_weighted_pars[fitted, 0]))
        columns.append(pyfits.Column(name='lum_age_total', format='E', array=lum_weighted_pars[fitted, 1]))
        columns.append(pyfits.Column(name='lum_M/L_total', format='E', array=lum_weighted_pars[fitted, 2]))
        columns.append(pyfits.Column(name='lum_[Fe/H]_total', format='E', array=lum_weighted_pars[fitted, 3]))
        columns.append(pyfits.Column(name='lum_[A/Fe]_total', format='E', array=lum_weighted_pars[fitted, 4]))
        columns.append(pyfits.Column(name='mass_coeff_frac_total', format='E', array=mass_weighted_pars[fitted, 0]))
        columns.append(pyfits.Column(name='mass_age_total', format='E', array=mass_weighted_pars[fitted, 1]))
        columns.append(pyfits.Column(name='mass_M/L_total', format='E', array=mass_weighted_pars[fitted, 2]))
        columns.append(pyfits.Column(name='mass_[Fe/H]_total', format='E', array=mass_weighted_pars[fitted, 3]))
        columns.append(pyfits.Column(name='mass_[A/Fe]_total', format='E', array=mass_weighted_pars[fitted, 4]))

        table_out = pyfits.new_table(columns)
        table_out.writeto(self.__outPrefix + '.stellar_table.fits', clobber=True)

    def run_eline_fit(self, parfile, parallel, verbose):
        """This functions fits a a set of emission lines to the spectra.

        Parameters
        ----------
        parfile : str
            The parameter file containing the constraints under which the
            emission lines will be fitted to the input spectra.
        parallel : {'auto', int}, optional
            If parallel is not equal to one, the python multiprocessing routine
            shall be used to run parts of the code in parallel. With the option
            `auto`, it adjusts the number of parallel processes to the number
            of cpu-cores/threads that are available.
        verbose : bool, optional
            Produces screen output, such that the user can follow which spectrum
            is currently fitted and what the results are for each spectrum.

        Notes
        -----
        The function does not return anything, but writes the results of the fit
        to disk. There are three files written to disk, all prepended by the
        `outprefix` supplied to this class at creation. The three files are:
        - `outprefix`.eline_model.fits which contains the best fitted emission
          line spectra.
        - `outprefix`.eline_res.fits contains the residual between the input
          fits file and the best fitted emission line model spectra.
        - `outprefix`.eline_table.fits contains the parameters of the best
          fit, like flux, velocity and the FWHM.
        """
        parList = ParameterList(parfile)
        eCompFile = parList['eCompFile'].getValue()
        vel_guess = float(parList['vel_guess'].getValue())
        line_fit = CustomMasks(parList['line_fit_region'].getValue())
        efit_method = parList['efit_method'].getValue()
        efit_ftol = float(parList['efit_ftol'].getValue())
        efit_xtol = float(parList['efit_xtol'].getValue())
        guess_window = int(parList['eguess_window'].getValue())
        min_x = float(parList['min_x'].getValue())
        max_x = float(parList['max_x'].getValue())
        min_y = float(parList['min_y'].getValue())
        max_y = float(parList['max_y'].getValue())

        line_par = fit_profile.parFile(eCompFile, self.__instrFWHM / 2.354)
        if self.__datatype == 'CUBE':
            res_out = loadCube(self.__outPrefix + '.cont_res.fits')
        elif self.__datatype == 'RSS':
            res_out = loadRSS(self.__outPrefix + '.cont_res.fits')
        res_wave_start = res_out._wave[0]
        res_wave_end = res_out._wave[-1]
        res_out._error = self.__inputData.subWaveLimits(res_wave_start, res_wave_end)._error
        res_out._mask = self.__inputData.subWaveLimits(res_wave_start, res_wave_end)._mask
        if self.__datatype == 'CUBE':
            out_lines = res_out.fitELines(line_par, line_fit.maskPixelsObserved(res_out.getWave(),
                vel_guess / 300000.0), min_x, max_x, min_y, max_y, method=efit_method, guess_window=guess_window,
                spectral_res=self.__instrFWHM, ftol=efit_ftol, xtol=efit_xtol, verbose=0, parallel=parallel)
            model_line = Cube(wave=cube_res_out._wave, data=out_lines[4], header=self.__inputData.getHeader())
            line_res = Cube(wave=cube_res_out._wave, data=res_out._data - model_line._data,
                header=self.__inputData.getHeader())
        elif self.__datatype == 'RSS':
            out_lines = res_out.fitELines(line_par, line_fit.maskPixelsObserved(res_out.getWave(),
                vel_guess / 300000.0), min_y, max_y, method=efit_method, guess_window=guess_window,
                spectral_res=self.__instrFWHM, ftol=efit_ftol, xtol=efit_xtol, verbose=0, parallel=parallel)
            model_line = RSS(wave=cube_res_out._wave, data=out_lines[4], header=self.__inputData.getHeader())
            line_res = RSS(wave=cube_res_out._wave, data=res_out._data - model_line._data,
                header=self.__inputData.getHeader())
        model_line.writeFitsData(self.__outPrefix + '.eline_model.fits')
        line_res.writeFitsData(self.__outPrefix + '.eline_res.fits')

        columns = []
        if self.__datatype == 'CUBE':
            columns.append(pyfits.Column(name='x_cor', format='J', array=out_lines[2][out_lines[1]]))
            columns.append(pyfits.Column(name='y_cor', format='J', array=out_lines[3][out_lines[1]]))
        elif self.__datatype == 'RSS':
            columns.append(pyfits.Column(name='fiber', format='J', array=out_lines[2][out_lines[1]]))
        for n in line_par._names:
            if line_par._profile_type[n] == 'Gauss':
                columns.append(pyfits.Column(name='%s_flux' % (n), format='E', array=out_lines[0][n]['flux'][out_lines[1]]))
                columns.append(pyfits.Column(name='%s_vel' % (n), format='E', unit='km/s',
                    array=out_lines[0][n]['vel'][out_lines[1]]))
                columns.append(pyfits.Column(name='%s_fwhm' % (n), format='E', unit='km/s',
                        array=out_lines[0][n]['fwhm'][out_lines[1]]))
        table_out = pyfits.new_table(columns)
        table_out.writeto(self.__outPrefix + '.eline_table.fits', clobber=True)

    def run_bootstrap(self, stellar_parfile, eline_parfile, bootstraps, modkeep, parallel, verbose):
        """The bootstrap functions performs a bootstrap on the data in order to
        obtain errors. The errors for the template fitting are determined by
        refitting with a subset of the templates (with fixed velocity and
        velocity dispersion) and looking at the spread of the determined
        parameters to obtain a bootstrapped error estimate. Each time the
        templates are fitted, the emission lines are also fitted and this then
        gives an estimate of the error in the emission line parameters.

        Parameters
        ----------
        stellar_parfile : str
            The parameter file containing the constraints under which the input
            template spectra are fitted.
        eline_par_file : str
            The parameter file containing the constraints under which the
            emission lines are fitted to the input spectra.
        bootstraps : int
            The number of bootstraps run to each spectra.
        modkeep : float
            The percentage of template spectra that will be keeped under each
            bootstrap run.
        parallel : {'auto', int}, optional
            If parallel is not equal to one, the python multiprocessing routine
            shall be used to run parts of the code in parallel. With the option
            `auto`, it adjusts the number of parallel processes to the number
            of cpu-cores/threads that are available.
        verbose : bool, optional
            Produces screen output, such that the user can follow which spectrum
            is currently fitted and what the results are for each spectrum.

        Notes
        -----
        The function does not return anything, but overwrites the stellar table
        and emission-line table fits files created by the fitting functions
        `run_SSP_fit` and `run_eline_fit`. The files
        `outprefix`.stellar_table.fits and `outprefix`.eline_table.fits are
        appended with information on the errors on the fitted parameters.
        """
        ## Read in parameter file and get values
        parList = ParameterList(stellar_parfile)
        tmpldir = parList['tmpldir'].getValue()
        tmplfile = parList['tmplfile'].getValue()
        vel_guess = float(parList['vel_guess'].getValue())
        vel_min = float(parList['vel_min'].getValue())
        vel_max = float(parList['vel_max'].getValue())
        nwidth_norm = int(parList['nwidth_norm'].getValue())
        start_wave = float(parList['start_wave'].getValue())
        end_wave = float(parList['end_wave'].getValue())
        excl_fit = CustomMasks(parList['excl_fit'].getValue())
        excl_cont = CustomMasks(parList['excl_cont'].getValue())
        nlib_guess = int(parList['tmplinitspec'].getValue())
        if nlib_guess < 0:
            modkeep = 100.0

        hdu = pyfits.open(self.__outPrefix + '.stellar_table.fits')
        stellar_table = hdu[1].data
        if self.__datatype == 'CUBE':
            x_cor = stellar_table.field('x_cor')
            y_cor = stellar_table.field('y_cor')
        elif self.__datatype == 'RSS':
            fiber = stellar_table.field('fiber')
        vel = stellar_table.field('vel_fit')
        disp = stellar_table.field('disp_fit')

        if eline_parfile is not None:
            parList = ParameterList(eline_parfile)
            eCompFile = parList['eCompFile'].getValue()
            vel_guess = float(parList['vel_guess'].getValue())
            line_fit = CustomMasks(parList['line_fit_region'].getValue())
            efit_method = parList['efit_method'].getValue()
            efit_ftol = float(parList['efit_ftol'].getValue())
            efit_xtol = float(parList['efit_xtol'].getValue())
            guess_window = int(parList['eguess_window'].getValue())

            line_par = fit_profile.parFile(eCompFile, self.__instrFWHM / 2.354)

            hdu = pyfits.open(self.__outPrefix + '.eline_table.fits')
            eline_table = hdu[1].data

        if verbose:
            print "The stellar population library is being prepared."
        lib = SSPlibrary(filename=tmpldir + '/' + tmplfile)
        min_wave = (self.__inputData.getWave()[0] / (1 + (vel_min - 2000) / 300000.0))
        max_wave = (self.__inputData.getWave()[-1] / (1 + (vel_max + 2000) / 300000.0))


        if nlib_guess < 0:
            select = numpy.arange(lib.getBaseNumber()) == nlib_guess * -1 - 1
            lib = lib.subLibrary(select)
        lib = lib.subWaveLibrary(min_wave=min_wave, max_wave=max_wave)
        lib = lib.matchInstFWHM(self.__instrFWHM, vel_guess)
        lib = lib.resampleWaveStepLinear(self.__inputData.getWaveStep(), vel_guess / 300000.0)
        lib_norm = lib.normalizeBase(nwidth_norm, excl_cont, vel_guess / 300000.0)
        lib_rebin = lib_norm.rebinLogarithmic()

        if verbose:
            print "The input data is being normalized."
        normData = self.__inputData.normalizeSpec(nwidth_norm, excl_cont.maskPixelsObserved(self.__inputData.getWave(),
             vel_guess / 300000.0))
        normDataSub = normData.subWaveLimits(start_wave, end_wave)
        excl_fit = excl_fit.maskPixelsObserved(normDataSub.getWave(), vel_guess / 300000.0)
        if eline_parfile is None:
            if self.__datatype == 'CUBE':
                (mass_weighted_pars_err, lum_weighted_pars_err, maps) = normDataSub.fit_Lib_Boots(lib_rebin,
                    x_cor, y_cor, vel, disp, mask_fit=excl_fit, bootstraps=bootstraps, modkeep=modkeep, parallel=parallel,
                    verbose=verbose)
            elif self.__datatype == 'RSS':
                (mass_weighted_pars_err, lum_weighted_pars_err, maps) = normDataSub.fit_Lib_Boots(lib_rebin,
                    fiber, vel, disp, mask_fit=excl_fit, bootstraps=bootstraps, modkeep=modkeep, parallel=parallel,
                    verbose=verbose)
        else:
            if self.__datatype == 'CUBE':
                (mass_weighted_pars_err, lum_weighted_pars_err, maps) = normDataSub.fit_Lib_Boots(lib_rebin,
                    x_cor, y_cor, vel, disp, bootstraps=bootstraps, par_eline=line_par,
                    select_wave_eline=line_fit.maskPixelsObserved(normDataSub.getWave(), vel_guess / 300000.0),
                    method_eline=efit_method, mask_fit=excl_fit, guess_window=guess_window, spectral_res=self.__instrFWHM,
                    ftol=efit_ftol, xtol=efit_xtol, modkeep=modkeep, parallel=parallel, verbose=verbose)
            elif self.__datatype == 'RSS':
                (mass_weighted_pars_err, lum_weighted_pars_err, maps) = normDataSub.fit_Lib_Boots(lib_rebin,
                    fiber, vel, disp, bootstraps=bootstraps, par_eline=line_par,
                    select_wave_eline=line_fit.maskPixelsObserved(normDataSub.getWave(), vel_guess / 300000.0),
                    mask_fit=excl_fit, method_eline=efit_method, guess_window=guess_window, spectral_res=self.__instrFWHM,
                    ftol=efit_ftol, xtol=efit_xtol, modkeep=modkeep, parallel=parallel, verbose=verbose)
        columns_stellar = []
        columns_stellar.append(pyfits.Column(name='lum_coeff_frac_total_err', format='E', array=lum_weighted_pars_err[:, 0]))
        columns_stellar.append(pyfits.Column(name='lum_age_total_err', format='E', array=lum_weighted_pars_err[:, 1]))
        columns_stellar.append(pyfits.Column(name='lum_M/L_total_err', format='E', array=lum_weighted_pars_err[:, 2]))
        columns_stellar.append(pyfits.Column(name='lum_[Fe/H]_total_err', format='E', array=lum_weighted_pars_err[:, 3]))
        columns_stellar.append(pyfits.Column(name='lum_[A/Fe]_total_err', format='E', array=lum_weighted_pars_err[:, 4]))
        columns_stellar.append(pyfits.Column(name='mass_coeff_frac_total_err', format='E', array=mass_weighted_pars_err[:, 0]))
        columns_stellar.append(pyfits.Column(name='mass_age_total_err', format='E', array=mass_weighted_pars_err[:, 1]))
        columns_stellar.append(pyfits.Column(name='mass_M/L_total_err', format='E', array=mass_weighted_pars_err[:, 2]))
        columns_stellar.append(pyfits.Column(name='mass_[Fe/H]_total_err', format='E', array=mass_weighted_pars_err[:, 3]))
        columns_stellar.append(pyfits.Column(name='mass_[A/Fe]_total_err', format='E', array=mass_weighted_pars_err[:, 4]))

        hdu = pyfits.new_table(stellar_table.columns[:17] + pyfits.new_table(columns_stellar).columns)
        hdu.writeto(self.__outPrefix + '.stellar_table.fits', clobber=True)

        if maps is not None:
            columns_eline = []
            for n in line_par._names:
                if line_par._profile_type[n] == 'Gauss':
                    columns_eline.append(pyfits.Column(name='%s_flux_err' % (n), format='E', array=maps[n]['flux_err']))
                    columns_eline.append(pyfits.Column(name='%s_vel_err' % (n), format='E', unit='km/s',
                        array=maps[n]['vel_err']))
                    columns_eline.append(pyfits.Column(name='%s_fwhm_err' % (n), format='E', unit='km/s',
                            array=maps[n]['fwhm_err']))

            hdu = pyfits.new_table(eline_table.columns[:len(columns_eline) + 2] + pyfits.new_table(columns_eline).columns)
            hdu.writeto(self.__outPrefix + '.eline_table.fits', clobber=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
Program to model the stellar population model from spectrosopic data stored either in RSS or datacube format.
Estimated parameters are velocity, velocity disperion, the best fit continuum model, and the star formation history.
Additionally the program allows to model emission lines in the residual spectra and infer the errors using a bootstrap
Monte Carlo simulation taking systematic uncertainties of the continuum model estimation into account.""",
formatter_class=argparse.ArgumentDefaultsHelpFormatter, prog='Paradise', version='Paradise version %s' % (__version__))

    parser.add_argument("input", type=str, help="""File name of the input datacube or RSS file. Please have a look at the
        documentation for the correct format of each file.""")
    parser.add_argument("outprefix", type=str, help="""Prefix used for nameing all the output file names.""")
    parser.add_argument("instrFWHM", type=float, help="""Instrumental spectral resolution of the input spectra.""")
    parser.add_argument("--SSP_par", type=str, default=None, help="""File name of the parameter file that controls the fitting procedure""")
    parser.add_argument("--line_par", type=str, default=None, help="""File name of the parameter file that controls the fitting procedure""")
    parser.add_argument("--bootstraps", type=int, default=None, help="""Number of bootstraps iterations per spectrum""")
    parser.add_argument("--modkeep", type=float, default=80, help="""Fraction of random SSP models used for each bootstrap.""")
    parser.add_argument("--parallel", type=str, default="auto", help="""Options are: 'auto' - using all CPUs available on the
    machine, an integer number specifying in the number of CPUs. An integer of 1 means no parrell processing.""")
    parser.add_argument("--verbose", action="store_true", default=False, help="""Flag to print some progress information on
    the screen. The default is False.""")

    args = parser.parse_args()
    app = ParadiseApp(args.input, args.outprefix, args.instrFWHM)
    if args.SSP_par is not None and args.bootstraps is None:
        app.run_SSP_fit(args.SSP_par, args.parallel, args.verbose)
    if args.line_par is not None and args.bootstraps is None:
        app.run_eline_fit(args.line_par, args.parallel, args.verbose)
    if args.bootstraps is not None:
        app.run_bootstrap(args.SSP_par, args.line_par, args.bootstraps, args.modkeep, args.parallel, args.verbose)
