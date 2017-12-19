#!/usr/bin/env python
import argparse
from Paradise import *

__version__ = "0.2"


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

        self.__inputData = loadSpectrum(input_file)
        if self.__inputData._datatype == 'CUBE':
            self.__inputData = loadCube(input_file)
            self.__inputData.correctError()
            self.__datatype = 'CUBE'
        elif self.__inputData._datatype == 'RSS':
            self.__inputData = loadRSS(input_file)
            self.__inputData.correctError()
            self.__datatype = 'RSS'
        elif self.__inputData._datatype == 'Spectrum1D':
            data = loadSpectrum(input_file)
            data.correctError()
            self.__datatype = 'RSS'
            if data._error is not None:
                err = numpy.array([data._error])
            else:
                err = None
            if data._mask is not None:
                m = numpy.array([data._mask])
            else:
                m = None
            self.__inputData = RSS(wave=data._wave, data=numpy.array([data._data]), error=err, mask=m)
        self.__outPrefix = outprefix
        try:
            self.__instrFWHM = SpectralResolution(res=float(instrFWHM))
        except ValueError:
            try:
                self.__instrFWHM=SpectralResolution()
                self.__instrFWHM.readFile(instrFWHM)
            except IOError:
                print("Wrong input for spectral resolution. Specify either a float number or the path to an ASCII file with columns wavelengt and spectral resolution as content.")
                
           
            

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
        kin_fix = bool(int(parList['kin_fix'].getValue()))
        mcmc_code = parList['mcmc_code'].getValue()
        nwidth_norm = int(parList['nwidth_norm'].getValue())
        iterations = int(parList['iterations'].getValue())
        samples = int(parList['samples'].getValue())
        walkers = int(parList['walkers'].getValue())
        burn = int(parList['burn'].getValue())
        thin = int(parList['thin'].getValue())
        store_chain = bool(int(parList['store_chain'].getValue()))
        start_wave = float(parList['start_wave'].getValue())
        end_wave = float(parList['end_wave'].getValue())
        excl_fit = CustomMasks(parList['excl_fit'].getValue())
        excl_cont = CustomMasks(parList['excl_cont'].getValue())
        min_x = int(parList['min_x'].getValue())
        max_x = int(parList['max_x'].getValue())
        min_y = int(parList['min_y'].getValue())
        max_y = int(parList['max_y'].getValue())
        bins = [[None, None]]
        
        try:
            f = open(parList['agebinfile'].getValue())
            try:
                for l in f.readlines():
                    bins.append([float(l.split()[0]), float(l.split()[1])])
            finally:
                f.close()
        except (IOError, ValueError):
            if verbose:
                print('agebinfile is non-existent or ignored.')
                print('Optional stellar population binning will not occur.')

        if verbose:
            print("The stellar population library is being prepared.")
        lib = SSPlibrary(filename=parList['tmpldir'].getValue() + '/' + parList['tmplfile'].getValue())
        if kin_fix is True:
            hdu = pyfits.open(self.__outPrefix + '.kin_table.fits')
            tab = hdu[1].data
            vel_fit = tab.field('vel_fit')
            disp_fit = tab.field('disp_fit')
            vel_fit_err = tab.field('vel_fit_err')
            disp_fit_err = tab.field('disp_fit_err')
            Rvel = -numpy.ones(vel_fit.shape)
            Rdisp = -numpy.ones(disp_fit.shape)
            if self.__datatype == 'CUBE':
                x_pixels = tab.field('x_cor')
                y_pixels = tab.field('y_cor')
            elif self.__datatype == 'RSS':
                fibers = tab.field('fiber')
            vel_min = numpy.min(vel_fit)
            vel_max = numpy.max(vel_fit)
            
        min_wave = (start_wave / (1 + (vel_max + 2000) / 300000.0))
        max_wave = (end_wave / (1 + (vel_min - 2000) / 300000.0))
        if nlib_guess < 0:
            select = numpy.arange(lib.getBaseNumber()) == nlib_guess * -1 - 1
            lib = lib.subLibrary(select)
        lib = lib.subWaveLibrary(min_wave=min_wave, max_wave=max_wave)
        lib = lib.matchInstFWHM(self.__instrFWHM, vel_guess)
        lib = lib.resampleWaveStepLinear(self.__inputData.getWaveStep(), vel_guess / 300000.0)
        lib_norm = lib.normalizeBase(nwidth_norm, excl_cont, vel_guess / 300000.0)
        lib_rebin = lib_norm.rebinLogarithmic()

        if verbose:
            print("The input cube is being normalized.")

        normData = self.__inputData.normalizeSpec(nwidth_norm, excl_cont.maskPixelsObserved(self.__inputData.getWave(),
             vel_guess / 300000.0))
        normDataSub = normData.subWaveLimits(start_wave, end_wave)
        
        #pylab.plot(self.__inputData._data[0,:],'-k')
        #pylab.plot(normData._normalization[0,:],'-g')
        #pylab.plot(normData._data[0,:],'-r')
        #pylab.show()
        if verbose:
            print("The stellar population modelling has been started.")
        if self.__datatype == 'CUBE':
            if kin_fix:
                (fitted, coeff, chi2, x_pix, y_pix, cube_model, mask) = normDataSub.fit_Lib_fixed_kin(lib_rebin, nlib_guess,
                vel_fit, disp_fit, x_pixels, y_pixels, min_x, max_x, min_y, max_y,
                excl_fit, verbose, parallel)
            else:
                (vel_fit, vel_fit_err, Rvel, disp_fit, disp_fit_err, Rdisp, fitted, coeff, chi2, x_pix, y_pix,
                cube_model, mask) = normDataSub.fit_Kin_Lib_simple(lib_rebin, nlib_guess, vel_min, vel_max, disp_min, disp_max,
                min_x, max_x, min_y, max_y, excl_fit, iterations, mcmc_code,
                walkers, burn, samples, thin, verbose, parallel)
        elif self.__datatype == 'RSS':
            if kin_fix:
                (fitted, coeff, chi2, fiber, rss_model, mask) = normDataSub.fit_Lib_fixed_kin(lib_rebin, nlib_guess, vel_fit, disp_fit, fibers, min_y, max_y, excl_fit, verbose, parallel)
            else:
                
                (vel_fit, vel_fit_err, Rvel, disp_fit, disp_fit_err, Rdisp, fitted, coeff, chi2, fiber, rss_model, mask, vel_trace, disp_trace) = normDataSub.fit_Kin_Lib_simple(lib_rebin, nlib_guess,vel_min, vel_max, disp_min, disp_max, min_y, max_y, excl_fit, iterations, mcmc_code, walkers, burn, samples, thin, verbose, store_chain, parallel)
        if verbose:
                print("Storing the results to %s (model), %s (residual) and %s (parameters)." % (
                    self.__outPrefix + '.cont_model.fits', self.__outPrefix + '.cont_res.fits',
                    self.__outPrefix + '.stellar_table.fits'))
        # pylab.plot(rss_model[0,:],'-g')
        # pylab.show()
        if self.__datatype == 'RSS':
            model_out = RSS(wave=normDataSub.getWave(), data=rss_model, mask=mask,
                header=self.__inputData.getHeader(), normalization=normDataSub.getNormalization())
            res_out = RSS(wave=normDataSub.getWave(),
                data=self.__inputData.subWaveLimits(start_wave, end_wave).getData() - rss_model,
                header=self.__inputData.getHeader())
        elif self.__datatype == 'CUBE':
            model_out = Cube(wave=normDataSub.getWave(), data=cube_model, mask=mask,
                header=self.__inputData.getHeader(), normalization=normDataSub.getNormalization())
            res_out = Cube(wave=normDataSub.getWave(),
                data=self.__inputData.subWaveLimits(start_wave, end_wave).getData() - cube_model,
                header=self.__inputData.getHeader())
        if numpy.max(self.__inputData.getWave()[1:] - self.__inputData.getWave()[:-1]) - numpy.min(
            self.__inputData.getWave()[1:] - self.__inputData.getWave()[:-1]) < 0.01:
            model_out.writeFitsData(self.__outPrefix + '.cont_model.fits')
            res_out.writeFitsData(self.__outPrefix + '.cont_res.fits')
        else:
            model_out.writeFitsData(self.__outPrefix + '.cont_model.fits', store_wave=True)
            res_out.writeFitsData(self.__outPrefix + '.cont_res.fits', store_wave=True)

        mass_weighted_pars = numpy.zeros((len(fitted), 5, len(bins) + 1), dtype=numpy.float32)
        lum_weighted_pars = numpy.zeros((len(fitted), 5, len(bins) + 1), dtype=numpy.float32)
        for i in range(len(fitted)):
            if fitted[i]:
                for j in range(len(bins)):
                    try:
                        mass_weighted_pars[i, :, j] = lib_norm.massWeightedPars(coeff[i, :], bins[j][0], bins[j][1])
                    except:
                        mass_weighted_pars[i, :, j] = numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan])
                    try:
                        lum_weighted_pars[i, :, j] = lib_norm.lumWeightedPars(coeff[i, :], bins[j][0], bins[j][1])
                    except:
                        lum_weighted_pars[i, :, j] = numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan])

        columns = []
        if self.__datatype == 'CUBE':
            columns.append(pyfits.Column(name='x_cor', format='J', array=x_pix[fitted]))
            columns.append(pyfits.Column(name='y_cor', format='J', array=y_pix[fitted]))
        elif self.__datatype == 'RSS':
            columns.append(pyfits.Column(name='fiber', format='J', array=fiber[fitted]))
        columns.append(pyfits.Column(name='vel_fit', format='E', unit='km/s', array=vel_fit[fitted]))
        columns.append(pyfits.Column(name='vel_fit_err', format='E', unit='km/s', array=vel_fit_err[fitted]))
        if store_chain and self.__datatype == 'RSS':
            columns.append(pyfits.Column(name='vel_trace', format='%dE'%(vel_trace.shape[1]), unit='km/s', array=vel_trace[fitted,:]))
        columns.append(pyfits.Column(name='Rvel', format='E', unit='km/s', array=Rvel[fitted]))
        columns.append(pyfits.Column(name='disp_fit', format='E', unit='km/s', array=disp_fit[fitted]))
        columns.append(pyfits.Column(name='disp_fit_err', format='E', unit='km/s', array=disp_fit_err[fitted]))
        if store_chain and self.__datatype == 'RSS':
            columns.append(pyfits.Column(name='disp_trace', format='%dE'%(disp_trace.shape[1]), unit='km/s', array=disp_trace[fitted,:]))
        columns.append(pyfits.Column(name='Rdisp', format='E', unit='km/s', array=Rdisp[fitted]))
        columns.append(pyfits.Column(name='chi2', format='E', array=chi2[fitted]))
        if lib.getBaseNumber() > 1:
            columns.append(pyfits.Column(name='base_coeff', format='%dE' % (lib.getBaseNumber()), array=coeff[fitted, :]))
        else:
            columns.append(pyfits.Column(name='base_coeff', format='E', array=coeff[fitted].flatten()))
        for i, postfix in enumerate(['total'] + ['bin{}'.format(i) for i in range(1, len(bins))]):
            columns.append(pyfits.Column(name='lum_coeff_frac_' + postfix, format='E', array=lum_weighted_pars[fitted, 0, i]))
            columns.append(pyfits.Column(name='lum_age_' + postfix, format='E', array=lum_weighted_pars[fitted, 1, i]))
            columns.append(pyfits.Column(name='lum_M/L_' + postfix, format='E', array=lum_weighted_pars[fitted, 2, i]))
            columns.append(pyfits.Column(name='lum_[Fe/H]_' + postfix, format='E', array=lum_weighted_pars[fitted, 3, i]))
            columns.append(pyfits.Column(name='lum_[A/Fe]_' + postfix, format='E', array=lum_weighted_pars[fitted, 4, i]))
            columns.append(pyfits.Column(name='mass_coeff_frac_' + postfix, format='E', array=mass_weighted_pars[fitted, 0, i]))
            columns.append(pyfits.Column(name='mass_age_' + postfix, format='E', array=mass_weighted_pars[fitted, 1, i]))
            columns.append(pyfits.Column(name='mass_M/L_' + postfix, format='E', array=mass_weighted_pars[fitted, 2, i]))
            columns.append(pyfits.Column(name='mass_[Fe/H]_' + postfix, format='E', array=mass_weighted_pars[fitted, 3, i]))
            columns.append(pyfits.Column(name='mass_[A/Fe]_' + postfix, format='E', array=mass_weighted_pars[fitted, 4, i]))

        try:
            table_out = pyfits.BinTableHDU.from_columns(columns)
        except:
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

        hdu = pyfits.open(self.__outPrefix + '.stellar_table.fits')
        stellar_table = hdu[1].data
        if self.__datatype == 'CUBE':
            x_cor = stellar_table.field('x_cor')
            y_cor = stellar_table.field('y_cor')
        elif self.__datatype == 'RSS':
            fiber = stellar_table.field('fiber')

        line_par = fit_profile.parFile(eCompFile, self.__instrFWHM)
        if self.__datatype == 'CUBE':
            res_out = loadCube(self.__outPrefix + '.cont_res.fits')
        elif self.__datatype == 'RSS':
            res_out = loadRSS(self.__outPrefix + '.cont_res.fits')
        disp1 = res_out._wave[1]-res_out._wave[0]
        res_wave_start = res_out._wave[0]-disp1/2.0
        disp2 = res_out._wave[-1]-res_out._wave[-2]
        res_wave_end = res_out._wave[-1]+disp2/2.0
        res_out._error = self.__inputData.subWaveLimits(res_wave_start, res_wave_end)._error
        res_out._mask = self.__inputData.subWaveLimits(res_wave_start, res_wave_end)._mask
        if self.__datatype == 'CUBE':
            out_lines = res_out.fitELines(line_par, line_fit.maskPixelsObserved(res_out.getWave(),
                vel_guess / 300000.0), min_x, max_x, min_y, max_y, method=efit_method, guess_window=guess_window,
                spectral_res=self.__instrFWHM, ftol=efit_ftol, xtol=efit_xtol, verbose=verbose, parallel=parallel)
            model_line = Cube(wave=res_out.getWave(), data=out_lines[4], header=self.__inputData.getHeader())
            line_res = Cube(wave=res_out.getWave(), data=res_out.getData() - model_line.getData(),
                header=self.__inputData.getHeader())
        elif self.__datatype == 'RSS':
            out_lines = res_out.fitELines(line_par, line_fit.maskPixelsObserved(res_out.getWave(),
                vel_guess / 300000.0), min_y, max_y, method=efit_method, guess_window=guess_window,
                spectral_res=self.__instrFWHM, ftol=efit_ftol, xtol=efit_xtol, verbose=verbose, parallel=parallel)
            model_line = RSS(wave=res_out.getWave(), data=out_lines[3], header=self.__inputData.getHeader())
            line_res = RSS(wave=res_out.getWave(), data=res_out.getData() - model_line.getData(),
                header=self.__inputData.getHeader())
        if numpy.max(self.__inputData.getWave()[1:] - self.__inputData.getWave()[:-1]) - numpy.min(
                     self.__inputData.getWave()[1:] - self.__inputData.getWave()[:-1]) < 0.01:
            model_line.writeFitsData(self.__outPrefix + '.eline_model.fits')
            line_res.writeFitsData(self.__outPrefix + '.eline_res.fits')
        else:
            model_line.writeFitsData(self.__outPrefix + '.eline_model.fits',store_wave=True)
            line_res.writeFitsData(self.__outPrefix + '.eline_res.fits',store_wave=True)
        indices = numpy.arange(len(out_lines[2]))
        valid = numpy.zeros(len(out_lines[2]),dtype="bool")
        for i in range(len(out_lines[2])):
            if self.__datatype == 'CUBE':
                select_pos = (x_cor==out_lines[2][i]) & (y_cor==out_lines[3][i])
            elif self.__datatype == 'RSS':
                select_pos= fiber == out_lines[2][i]

            if numpy.sum(select_pos)>0:
                valid[i]=True
        columns = []
        if self.__datatype == 'CUBE':
            columns.append(pyfits.Column(name='x_cor', format='J', array=out_lines[2][valid]))
            columns.append(pyfits.Column(name='y_cor', format='J', array=out_lines[3][valid]))
        elif self.__datatype == 'RSS':
            columns.append(pyfits.Column(name='fiber', format='J', array=out_lines[2][valid]))
        for n in line_par._names:
            if line_par._profile_type[n] == 'Gauss':
                columns.append(pyfits.Column(name='%s_flux' % (n), format='E', array=out_lines[0][n]['flux'][valid]))
                columns.append(pyfits.Column(name='%s_vel' % (n), format='E', unit='km/s',
                    array=out_lines[0][n]['vel'][valid]))
                columns.append(pyfits.Column(name='%s_fwhm' % (n), format='E', unit='km/s',
                        array=out_lines[0][n]['fwhm'][valid]))

        try:
            table_out = pyfits.BinTableHDU.from_columns(columns)
        except:
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
        kin_bootstrap = bool(int(parList['kin_bootstrap'].getValue()))

        bins = [[None, None]]
        try:
            f = open(parList['agebinfile'].getValue())
            try:
                for l in f.readlines():
                    bins.append([float(l.split()[0]), float(l.split()[1])])
            finally:
                f.close()
        except (IOError, ValueError):
            if verbose:
                print('agebinfile is non-existent or ignored.')
                print('Optional stellar population binning will not occur.')

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
        if kin_bootstrap:
            vel_err = stellar_table.field('vel_fit_err')
            disp_err = stellar_table.field('disp_fit_err')
        else:
            vel_err = None
            disp_err = None

        if eline_parfile is not None:
            parList = ParameterList(eline_parfile)
            eCompFile = parList['eCompFile'].getValue()
            vel_guess = float(parList['vel_guess'].getValue())
            line_fit = CustomMasks(parList['line_fit_region'].getValue())
            efit_method = parList['efit_method'].getValue()
            efit_ftol = float(parList['efit_ftol'].getValue())
            efit_xtol = float(parList['efit_xtol'].getValue())
            guess_window = int(parList['eguess_window'].getValue())

            line_par = fit_profile.parFile(eCompFile, self.__instrFWHM)

            hdu = pyfits.open(self.__outPrefix + '.eline_table.fits')
            eline_table = hdu[1].data
            if self.__datatype == 'CUBE':
                x_eline = eline_table.field('x_cor')
                y_eline = eline_table.field('y_cor')
            if self.__datatype == 'RSS':
                fiber_eline = eline_table.field('fiber')

        if verbose:
            print("The stellar population library is being prepared.")
        lib = SSPlibrary(filename=tmpldir + '/' + tmplfile)
        min_wave = (start_wave / (1 + (vel_max + 2000) / 300000.0))
        max_wave = (end_wave / (1 + (vel_min - 2000) / 300000.0))


        if nlib_guess < 0:
            select = numpy.arange(lib.getBaseNumber()) == nlib_guess * -1 - 1
            lib = lib.subLibrary(select)
        lib = lib.subWaveLibrary(min_wave=min_wave, max_wave=max_wave)
        lib = lib.matchInstFWHM(self.__instrFWHM, vel_guess)
        lib = lib.resampleWaveStepLinear(self.__inputData.getWaveStep(), vel_guess / 300000.0)
        lib_norm = lib.normalizeBase(nwidth_norm, excl_cont, vel_guess / 300000.0)
        lib_rebin = lib_norm.rebinLogarithmic()

        if verbose:
            print("The input data is being normalized.")
        normData = self.__inputData.normalizeSpec(nwidth_norm, excl_cont.maskPixelsObserved(self.__inputData.getWave(),
             vel_guess / 300000.0))
        #normData.writeFitsData('test.fits')
        normDataSub = normData.subWaveLimits(start_wave, end_wave)
        excl_fit = excl_fit.maskPixelsObserved(normDataSub.getWave(), vel_guess / 300000.0)
        if eline_parfile is None:
            if self.__datatype == 'CUBE':
                (coeffs, maps) = normDataSub.fit_Lib_Boots(
                    lib_rebin, x_cor, y_cor, vel, disp, vel_err, disp_err,
                    mask_fit=excl_fit, bootstraps=bootstraps, modkeep=modkeep,
                    parallel=parallel, verbose=verbose)
            elif self.__datatype == 'RSS':
                (coeffs, maps) = normDataSub.fit_Lib_Boots(
                    lib_rebin, fiber, vel, disp, vel_err, disp_err,
                    mask_fit=excl_fit, bootstraps=bootstraps, modkeep=modkeep,
                    parallel=parallel, verbose=verbose)
        else:
            select_wave_eline = line_fit.maskPixelsObserved(normDataSub.getWave(), vel_guess / 300000.0)
            if self.__datatype == 'CUBE':
                (coeffs, maps) = normDataSub.fit_Lib_Boots(
                    lib_rebin, x_cor, y_cor, vel, disp, vel_err, disp_err,
                    line_par, select_wave_eline, excl_fit, efit_method,
                    guess_window, self.__instrFWHM, efit_ftol, efit_xtol,
                    bootstraps, modkeep, parallel, verbose)
            elif self.__datatype == 'RSS':
                (coeffs, maps) = \
                    normDataSub.fit_Lib_Boots(lib_rebin, fiber, vel, disp, vel_err, disp_err, bootstraps=bootstraps, par_eline=line_par,
                    select_wave_eline=select_wave_eline, mask_fit=excl_fit, method_eline=efit_method,
                    guess_window=guess_window, spectral_res=self.__instrFWHM, ftol=efit_ftol, xtol=efit_xtol,
                    modkeep=modkeep, parallel=parallel, verbose=verbose)
        mass_weighted_pars_full = numpy.zeros((len(vel), bootstraps, 5, len(bins) + 1), dtype=numpy.float32)
        lum_weighted_pars_full = numpy.zeros((len(vel), bootstraps, 5, len(bins) + 1), dtype=numpy.float32)
        for i in range(len(vel)):
            for j in range(len(bins)):
                for m in range(bootstraps):
                    try:
                        mass_weighted_pars_full[i, m, :, j] = lib_norm.massWeightedPars(coeffs[i, m, :], min_age=bins[j][0], max_age=bins[j][1])
                    except:
                        mass_weighted_pars_full[i, m, :, j] = numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan])
                    try:
                        lum_weighted_pars_full[i, m, :, j] = lib_norm.lumWeightedPars(coeffs[i, m, :], min_age=bins[j][0], max_age=bins[j][1])
                    except:
                        lum_weighted_pars_full[i, m, :, j] = numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan])
        mass_weighted_pars_mean = numpy.nanmean(mass_weighted_pars_full, axis=1)
        mass_weighted_pars_err = numpy.nanstd(mass_weighted_pars_full, axis=1)
        lum_weighted_pars_mean = numpy.nanmean(lum_weighted_pars_full, axis=1)
        lum_weighted_pars_err = numpy.nanstd(lum_weighted_pars_full, axis=1)
        columns_stellar = []
        for i, postfix in enumerate(['total'] + ['bin{}'.format(i) for i in range(1, len(bins))]):
            columns_stellar.append(pyfits.Column(name='lum_coeff_frac_' + postfix + '_btmean', format='E', array=lum_weighted_pars_mean[:, 0, i]))
            columns_stellar.append(pyfits.Column(name='lum_coeff_frac_' + postfix + '_err', format='E', array=lum_weighted_pars_err[:, 0, i]))
            columns_stellar.append(pyfits.Column(name='lum_age_' + postfix + '_btmean', format='E', array=lum_weighted_pars_mean[:, 1, i]))
            columns_stellar.append(pyfits.Column(name='lum_age_' + postfix + '_err', format='E', array=lum_weighted_pars_err[:, 1, i]))
            columns_stellar.append(pyfits.Column(name='lum_M/L_' + postfix + '_btmean', format='E', array=lum_weighted_pars_mean[:, 2, i]))
            columns_stellar.append(pyfits.Column(name='lum_M/L_' + postfix + '_err', format='E', array=lum_weighted_pars_err[:, 2, i]))
            columns_stellar.append(pyfits.Column(name='lum_[Fe/H]_' + postfix + '_btmean', format='E', array=lum_weighted_pars_mean[:, 3, i]))
            columns_stellar.append(pyfits.Column(name='lum_[Fe/H]_' + postfix + '_err', format='E', array=lum_weighted_pars_err[:, 3, i]))
            columns_stellar.append(pyfits.Column(name='lum_[A/Fe]_' + postfix + '_btmean', format='E', array=lum_weighted_pars_mean[:, 4, i]))
            columns_stellar.append(pyfits.Column(name='lum_[A/Fe]_' + postfix + '_err', format='E', array=lum_weighted_pars_err[:, 4, i]))
            columns_stellar.append(pyfits.Column(name='mass_coeff_frac_' + postfix + '_btmean', format='E', array=mass_weighted_pars_mean[:, 0, i]))
            columns_stellar.append(pyfits.Column(name='mass_coeff_frac_' + postfix + '_err', format='E', array=mass_weighted_pars_err[:, 0, i]))
            columns_stellar.append(pyfits.Column(name='mass_age_' + postfix + '_btmean', format='E', array=mass_weighted_pars_mean[:, 1, i]))
            columns_stellar.append(pyfits.Column(name='mass_age_' + postfix + '_err', format='E', array=mass_weighted_pars_err[:, 1, i]))
            columns_stellar.append(pyfits.Column(name='mass_M/L_' + postfix + '_btmean', format='E', array=mass_weighted_pars_mean[:, 2, i]))
            columns_stellar.append(pyfits.Column(name='mass_M/L_' + postfix + '_err', format='E', array=mass_weighted_pars_err[:, 2, i]))
            columns_stellar.append(pyfits.Column(name='mass_[Fe/H]_' + postfix + '_btmean', format='E', array=mass_weighted_pars_mean[:, 3, i]))
            columns_stellar.append(pyfits.Column(name='mass_[Fe/H]_' + postfix + '_err', format='E', array=mass_weighted_pars_err[:, 3, i]))
            columns_stellar.append(pyfits.Column(name='mass_[A/Fe]_' + postfix + '_btmean', format='E', array=mass_weighted_pars_mean[:, 4, i]))
            columns_stellar.append(pyfits.Column(name='mass_[A/Fe]_' + postfix + '_err', format='E', array=mass_weighted_pars_err[:, 4, i]))

        columns_bootstrap = None
        try:
            if bool(int(parList['bootstrap_verb'].getValue())) != 0:
                columns_bootstrap = []
                for m in range(bootstraps):
                    if lib.getBaseNumber() > 1:
                        columns_bootstrap.append(pyfits.Column(
                            name='bootstrap_coeff_{}'.format(m),
                            format='%dE' % (lib.getBaseNumber()),
                            array=coeffs[:, m]))
                    else:
                        columns_bootstrap.append(pyfits.Column(
                            name='bootstrap_coeff_{}'.format(m),
                            format='E', array=coeffs[:, m].flatten()))
        except KeyError:  # in case bootstrap_verb is not in the parameters-file
            pass

        tbl_size = 8 + 2 if self.__datatype == 'CUBE' else 8 + 1
        tbl_size += 10 * len(bins)
        try:
            hdu = pyfits.BinTableHDU.from_columns(stellar_table.columns[:tbl_size] + pyfits.ColDefs(columns_stellar))
            if columns_bootstrap is not None:
                btunit = pyfits.BinTableHDU.from_columns(pyfits.ColDefs(columns_bootstrap))
        except:
            hdu = pyfits.new_table(stellar_table.columns[:tbl_size] + pyfits.new_table(columns_stellar).columns)
            if columns_bootstrap is not None:
                btunit = pyfits.new_table(columns_bootstrap)
        if columns_bootstrap is None:
            hdu = pyfits.HDUList([pyfits.PrimaryHDU([]), hdu])
        else:
            hdu = pyfits.HDUList([pyfits.PrimaryHDU([]), hdu, btunit])
        hdu.writeto(self.__outPrefix + '.stellar_table.fits', clobber=True)

        if self.__datatype == 'CUBE' and eline_parfile is not None:
            mapping=numpy.zeros(len(x_eline),dtype=numpy.int16)
            indices = numpy.arange(len(x_cor))
            valid = numpy.zeros(len(x_eline),dtype="bool")
            for i in range(len(x_eline)):
                select_pos = (x_cor==x_eline[i]) & (y_cor==y_eline[i])
                if numpy.sum(select_pos)>0:
                    valid[i]=True
                    mapping[i]=indices[select_pos][0]
                else:
                    mapping[i]=-1
        elif self.__datatype == 'RSS' and eline_parfile is not None:
            mapping=numpy.zeros(len(fiber_eline),dtype=numpy.int16)
            indices = numpy.arange(len(fiber))
            valid = numpy.zeros(len(fiber_eline),dtype="bool")
            for i in range(len(fiber_eline)):
                select_pos = (fiber==fiber_eline[i])
                if numpy.sum(select_pos)>0:
                    valid[i]=True
                    mapping[i]=indices[select_pos][0]
                else:
                    mapping[i]=-1


        if maps is not None:
            columns_eline = []
            for n in line_par._names:
                if line_par._profile_type[n] == 'Gauss':
                    columns_eline.append(pyfits.Column(name='%s_flux_err' % (n), format='E', array=maps[n]['flux_err'][valid][mapping[valid]]))
                    columns_eline.append(pyfits.Column(name='%s_vel_err' % (n), format='E', unit='km/s',
                        array=maps[n]['vel_err'][valid][mapping[valid]]))
                    columns_eline.append(pyfits.Column(name='%s_fwhm_err' % (n), format='E', unit='km/s',
                            array=maps[n]['fwhm_err'][valid][mapping[valid]]))

            try:
                hdu = pyfits.BinTableHDU.from_columns(eline_table.columns[:len(columns_eline) + 2] + pyfits.ColDefs(columns_eline))
            except:
                hdu = pyfits.new_table(eline_table.columns[:len(columns_eline) + 2] + pyfits.new_table(columns_eline).columns)
            hdu.writeto(self.__outPrefix + '.eline_table.fits', clobber=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
Program to model the stellar population model from spectrosopic data stored either in RSS or datacube format.
Estimated parameters are velocity, velocity disperion, the best fit continuum model, and the star formation history.
Additionally the program allows to model emission lines in the residual spectra and infer the errors using a bootstrap
Monte Carlo simulation taking systematic uncertainties of the continuum model estimation into account.""",
formatter_class=argparse.ArgumentDefaultsHelpFormatter, prog='Paradise')
    parser.add_argument('--version', action='version', version='Paradise version %s' % (__version__))

    parser.add_argument("input", type=str, help="""File name of the input datacube or RSS file. Please have a look at the
        documentation for the correct format of each file.""")
    parser.add_argument("outprefix", type=str, help="""Prefix used for nameing all the output file names.""")
    parser.add_argument("instrFWHM", type=str, help="""Instrumental spectral resolution of the input spectra.""")
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
