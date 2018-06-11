from Paradise.lib.data import Data
from Paradise.lib.spectrum1d import Spectrum1D
import numpy
from multiprocessing import cpu_count
from multiprocessing import Pool
from time import sleep
from functools import partial
import signal


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class RSS(Data):
    """A class representing 2D spectra

    `RSS` is a subclass of Data which allows for handling and organizing a
    two-dimensional spectra. The class supports reading and writing FITS
    files, resampling and rebinning, velocity shifting and broadening, the
    application of extinction corrections and various advanced fitting
    functions.

    Parameters
    ----------
    data : `numpy.ndarray`
        The spectra as a 2D numpy array structured such that the different
        spectra are located along the first dimension.
    wave : `numpy.ndarray`
        The wavelength elements corresponding to the different data points
        along the first dimension of the `data`.
    error : `numpy.ndarray`, optional
        The error spectrum, should be of the same shape as `data`.
        If `error` equals None, it is assumed the error spectrum is not
        known.
    mask : `numpy.ndarray`
        A boolean array where True represents a masked (invalid) data point
        and False a good data point. Should be of the same shape as `data`.
    normalization : `numpy.ndarray`
        An array which is used to normalize the data/error; both data and
        error are divided by `normalization`. Should be of the same shape
        as `data`.
    inst_fwhm : float
        The instrumental FWHM in the same units as `wavelength`.
    header : Header, optional
        Contains information for reading and writing data to and from Fits
        files.
    """
    def __init__(self, data=None, wave=None, error=None, mask=None, error_weight=None, normalization=None, inst_fwhm=None,
    header=None):
        Data.__init__(self, wave=wave, data=data, error=error, mask=mask, error_weight=error_weight,
        normalization=normalization, inst_fwhm=inst_fwhm, header=header)

    def getSpec(self, i):
        """Get a single spectrum from the RSS instance.

        Parameters
        ----------
        i : int
            The index of the spectrum which will be returned.

        Returns
        -------
        spec : Spectrum1D
            The object containg the i-th spectrum, together with the proper
            error, mask and normalization.
        """
        wave = self._wave
        data = self._data[i, :]
        if self._error is not None:
            error = self._error[i, :]
        else:
            error = None
        if self._mask is not None:
            mask = self._mask[i, :]
        else:
            mask = None
        if self._normalization is not None:
            normalization = self._normalization[i, :]
        else:
            normalization = None
        spec = Spectrum1D(wave=wave, data=data, error=error, mask=mask, normalization=normalization)
        return spec

    def fit_Kin_Lib_simple(self, SSPLib, nlib_guess, vel_min, vel_max, disp_min, disp_max, min_y, max_y, mask_fit,
        iterations=2, mcmc_code='emcee', walkers=50, burn=1500, samples=4000, thin=2, verbose=False,sample_out=False, parallel='auto'):
        """Fits template spectra according to Markov chain Monte Carlo
        algorithm. This uses the PyMC library. The MCMC code is runned to
        determine the velocity and the velocity dispersion while the
        coefficients for the best combination of stellar templates are
        determined by iteration and non-negative least sqaures fitting.

        Notes
        -----
        The output sample might be smaller than the input sample, due to any
        limits imposed by `min_y` and `max_y`.

        Parameters
        ----------
        SSPLib : SSPlibrary
            The library containing the template spectra.
        nlib_guess : int
            The initial guess for the best fitting template spectrum.
        vel_min : float
            The minimum velocity in km/s used in the MCMC for each spectrum.
        vel_max : float
            The maximum velocity in km/s used in the MCMC for each spectrum.
        disp_min : float
            The minimum velocity dispersion in km/s used in the MCMC.
        disp_max : float
            The maximum velocity dispersion in km/s used in the MCMC.
        min_y : int
            The lowest index from the spectral data set that will be used in the
            fitting.
        max_y : int
            The highest index from the spectral data set that will be used in
            the fitting.
        mask_fit : `numpy.ndarray`
            A 1D boolean array representing any wavelength regions which are
            masked out during the fitting of each spectrum with the
            (normalized) stellar templates.
        iterations : int
            The number of iterations applied to determine the coefficients for
            the set of template spectra.
        burn : int, optional
            The burn-in parameter that is often applied in MCMC implementations.
            The first `burn` samples will be discarded in the further analysis.
        samples : int, optional
            The number of iterations runned by PyMC.
        thin : int, optional
            Only keeps every `thin`th sample, this argument should circumvent
            any possible autocorrelation among the samples.
        verbose : bool, optional
            Produces screen output, such that the user can follow which spectrum
            is currently fitted and what the results are for each spectrum.
        parallel : {'auto', int}, optional
            If parallel is not equal to one, the python multiprocessing routine
            shall be used to run parts of the code in parallel. With the option
            `auto`, it adjusts the number of parallel processes to the number
            of cpu-cores available.

        Returns
        -------
        vel_fit : `numpy.ndarray`
            The average velocity determined from the MCMC fitting for each
            spectrum.
        vel_fit_err : `numpy.ndarray`
            The standard deviation in the velocity determined from the MCMC
            fitting for each spectrum.
        disp_fit : `numpy.ndarray`
            The average velocity dispersion determined from the MCMC fitting for
            each spectrum.
        disp_fit_err : `numpy.ndarray`
            The standard devation in the velocity dispersion determined from the
            MCMC fitting for each spectrum.
        fitted : `numpy.ndarray`
            A 1D boolean array representing whether each spectrum is fitted
            correctly.
        coeff : `numpy.ndarray`
            The coefficients of the SSP library which will produce the best fit
            to the data. The variable is a 2D numpy array where the first
            dimension represents the different spectra and the second dimension
            the coefficients for each template in `SSPLib`.
        chi2 : `numpy.ndarray`
            The chi^2 value between `bestfit_spec` and `data`.
        fiber : `numpy.ndarray`
            A 1D numpy array containing the indices of each output spectrum
            pointing to each input `RSS`.
        rss_model : `numpy.ndarray`
            The best fitted spectrum obtained from the linear
            combination of template spectra.
        """
        rss_model = numpy.zeros(self.getShape(), dtype=numpy.float32)
        mask = numpy.zeros(self.getShape(), dtype=numpy.bool)
        vel_fit = numpy.zeros(self._fibers, dtype=numpy.float32)
        vel_fit_err = numpy.zeros(self._fibers, dtype=numpy.float32)
        Rvel = numpy.zeros(self._fibers, dtype=numpy.float32)
        disp_fit = numpy.zeros(self._fibers, dtype=numpy.float32)
        disp_fit_err = numpy.zeros(self._fibers, dtype=numpy.float32)
        Rdisp = numpy.zeros(self._fibers, dtype=numpy.float32)
        chi2 = numpy.zeros(self._fibers, dtype=numpy.float32)
        fiber = numpy.zeros(self._fibers, dtype=numpy.int16)
        fitted = numpy.zeros(self._fibers, dtype="bool")
        coeff = numpy.zeros((self._fibers, SSPLib.getBaseNumber()), dtype=numpy.float32)
        if sample_out:
            print(samples,burn, (samples-burn)//2)
            vel_trace = numpy.zeros((self._fibers,(samples-burn)//2), dtype=numpy.float32)
            disp_trace = numpy.zeros((self._fibers,(samples-burn)//2), dtype=numpy.float32)
        else:
            vel_trace = None
            disp_trace = None

        def extract_result(result, i,sample_out):
            vel_fit[i] = result[0]
            vel_fit_err[i] = result[1]
            Rvel[i] = result[2]
            disp_fit[i] = result[3]
            disp_fit_err[i] = result[4]
            Rdisp[i] = result[5]
            fiber[i] = i
            fitted[i] = True
            coeff[i, :] = result[7]
            chi2[i] = result[8]
            if sample_out:
                vel_trace[i,:]= result[9]
                disp_trace[i,:]= result[10]
            rss_model[i, :] = result[6].unnormalizedSpec().getData()
            mask[i, :] = result[6].getMask()
            if verbose:
                print("Fitting of SSP(s) to fiber %d finished." %i)
                print("vel_fit: %.3f  disp_fit: %.3f chi2: %.2f" % (vel_fit[i], disp_fit[i], chi2[i]))

        if parallel == 'auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus, maxtasksperchild=200, initializer=init_worker)
            results = []
        for m in range(self._fibers):
            spec = self.getSpec(m)
            if spec.hasData() and m >= (min_y - 1) and m <= (max_y - 1):
                args = (SSPLib, nlib_guess, vel_min, vel_max, disp_min,
                        disp_max, mask_fit, iterations, mcmc_code,
                        walkers, burn, samples, thin, sample_out)
                if cpus > 1:
                    results.append([m, pool.apply_async(spec.fit_Kin_Lib_simple, args, callback=partial(extract_result,i=m,sample_out=sample_out))])
                    sleep(0.01)
                else:
                    try:
                        result = spec.fit_Kin_Lib_simple(*args)
                        extract_result(result, m, sample_out)
                    except (ValueError, IndexError) as e:
                        print("Fitting of spectrum %d failed: %s" %(m, e.message))

        if cpus > 1:
            pool.close()
            pool.join()
            for i, result in results:
                try:
                    result.get()
                except (ValueError, IndexError) as e:
                    print("Fitting of spectrum %d failed: %s" % (i, e.message))

        return vel_fit, vel_fit_err, Rvel, disp_fit, disp_fit_err, Rdisp, fitted, coeff, chi2, fiber, rss_model, mask, vel_trace, disp_trace

    def fit_Lib_fixed_kin(self, SSPLib, nlib_guess, vel, vel_disp, fibers, min_y, max_y, mask_fit,
        verbose=False, parallel='auto'):
        """Fits template spectra with fixed kinematics with non-negative least
        squares fitting to determine the best combination of template spectra.

        Notes
        -----
        The output sample might be smaller than the input sample, due to any
        limits imposed by `min_y` and `max_y`.

        Parameters
        ----------
        SSPLib : SSPlibrary
            The library containing the template spectra.
        vel : `numpy.ndarray`
            The velocity in km/s to which each input spectra is resampled.
        vel_disp : `numpy.ndarray`
            The velocity dispersion in km/s to which each input spectra is
            broadened.
        fibers : `numpy.ndarray`
            The index of each spectrum used to read in the corresponding values
            for `vel and `vel_disp`. The number of elements should equal the
            value (`max_y` - `min_y`).
        min_y : int
            The lowest index from the spectral data set that will be used in the
            fitting.
        max_y : int
            The highest index from the spectral data set that will be used in
            the fitting.
        mask_fit : `numpy.ndarray`
            A 1D boolean array representing any wavelength regions which are
            masked out during the fitting of each spectrum with the
            (normalized) stellar templates.
        verbose : bool, optional
            Produces screen output, such that the user can follow which spectrum
            is currently fitted and what the results are for each spectrum.
        parallel : {'auto', int}, optional
            If parallel is not equal to one, the python multiprocessing routine
            shall be used to run parts of the code in parallel. With the option
            `auto`, it adjusts the number of parallel processes to the number
            of cpu-cores available.

        Returns
        -------
        fitted : `numpy.ndarray`
            A 1D boolean array representing whether each spectrum is fitted
            correctly.
        coeff : `numpy.ndarray`
            The coefficients of the SSP library which will produce the best fit
            to the data. The variable is a 2D numpy array where the first
            dimension represents the different spectra and the second dimension
            the coefficients for each template in `SSPLib`.
        chi2 : `numpy.ndarray`
            The chi^2 value between `bestfit_spec` and `data`.
        fiber : `numpy.ndarray`
            A 1D numpy array containing the indices of each output spectrum
            pointing to each input `RSS`.
        rss_model : `numpy.ndarray`
            The best fitted spectrum obtained from the linear
            combination of template spectra.
        """
        rss_model = numpy.zeros(self.getShape(), dtype=numpy.float32)
        mask = numpy.zeros(self.getShape(), dtype=numpy.bool)
        chi2 = numpy.zeros(self._fibers, dtype=numpy.float32)
        fiber = numpy.zeros(self._fibers, dtype=numpy.int16)
        fitted = numpy.zeros(self._fibers, dtype="bool")
        coeff = numpy.zeros((self._fibers, SSPLib.getBaseNumber()), dtype=numpy.float32)

        def extract_result(result, i):
            fiber[i] = i
            fitted[i] = True
            coeff[i, :] = result[0]
            chi2[i] = result[2]
            rss_model[i, :] = result[1].unnormalizedSpec().getData()
            mask[i, :] = result[1].getMask()
            if verbose:
                print("Fitting of SSP(s) to fiber %d finished." %i)
                print("chi2: %.2f" % (chi2[i]))

        if parallel == 'auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus, maxtasksperchild=200, initializer=init_worker)
            results = []
        for m in range(self._fibers):
            spec = self.getSpec(m)
            if spec.hasData() and m >= (min_y - 1) and m <= (max_y - 1):
                args = (SSPLib, nlib_guess, vel[fibers[m]], vel_disp[fibers[m]],
                        mask_fit.maskPixelsObserved(spec.getWave(), vel[m] / 300000.0))
                if cpus > 1:
                    results.append([m, pool.apply_async(spec.fitSuperposition, args, callback=partial(extract_result, i=m))])
                    sleep(0.01)
                else:
                    try:
                        result = spec.fitSuperposition(*args)
                        extract_result(result, m)
                    except (ValueError, IndexError):
                        print("Fitting of spectrum %d failed." % m)

        if cpus > 1:
            pool.close()
            pool.join()
            for i, result in results:
                try:
                    result.get()
                except (ValueError, IndexError):
                    print("Fitting of spectrum %d failed." % i)

        return fitted, coeff, chi2, fiber, rss_model, mask

    def fitELines(self, par, select_wave, min_y, max_y, method='leastsq', guess_window=0.0, spectral_res=0.0,
    ftol=1e-4, xtol=1e-4, verbose=1, parallel='auto'):
        """This routine fits a set of emission lines to each spectrum.

        Parameters
        ----------
        par : parFile
            The object containing all the constraints on the parameters.
        select_wave : numpy.ndarray
            A 1D boolean array where the True value represents which elements
            in the wave, data, error, mask and normalization.
        min_y : int
            The lowest index from the spectral data set that will be used in the
            fitting.
        max_y : int
            The highest index from the spectral data set that will be used in
            the fitting.
        method : {'leastsq', 'simplex'}, optional
            This argument specifies if ordinary least squares fitting
            (`leastsq`) should be applied, or if a downhill simplex algorithm
            (`simplex`) should be used.
        guess_window : float, optional
            The wavelength region in which the emission line will be fitted.
        spectral_res : float, optional
            The spectral resolution of the line.
        ftol : float, optional
            The maximum acceptable error for fit convergence.
        xtol : float, optional
            The relative acceptable error for fit convergence.
        verbose : bool, optional
            Produces screen output, such that the user can follow which spectrum
            is currently fitted and what the results are for each spectrum.
        parallel : {'auto', int}, optional
            If parallel is not equal to one, the python multiprocessing routine
            shall be used to run parts of the code in parallel. With the option
            `auto`, it adjusts the number of parallel processes to the number
            of cpu-cores available.

        Returns
        -------
        maps : dictionary
            The flux and the velocity shift of each line
        fitted : `numpy.ndarray`
            A 1D numpy array representing the best fitted values at the
            wavelength range `select_wave`.
        fiber : `numpy.ndarray`
            A 1D numpy array containing the indices of each output spectrum
            pointing to each input `RSS`.
        rss_model : `numpy.ndarray`
            A 1D numpy array representing the residuals at the wavelength range
            `select_wave`.
        """
        rss_model = numpy.zeros(self.getShape(), dtype=numpy.float32)
        fitted = numpy.zeros(self._fibers, dtype="bool")
        fiber = numpy.zeros(self._fibers, dtype=numpy.int16)

        maps = {}
        for n in par._names:
            model = {}
            if par._profile_type[n] == 'Gauss':
                model['flux'] = numpy.zeros((self._fibers), dtype=numpy.float32)
                model['vel'] = numpy.zeros((self._fibers), dtype=numpy.float32)
                model['fwhm'] = numpy.zeros((self._fibers), dtype=numpy.float32)
            maps[n] = model

        def extract_result(result, i):
            fitted[i] = True
            fiber[i] = i
            rss_model[i, :] = result[1]
            if verbose:
                print("Fitting of emission lines to fiber %d finished." %i)
            for n in par._names:
                if par._profile_type[n] == 'Gauss':
                    maps[n]['flux'][i] = result[0][n][0]
                    maps[n]['vel'][i] = result[0][n][1]
                    maps[n]['fwhm'][i] = result[0][n][2]

        if parallel == 'auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus, maxtasksperchild=200, initializer=init_worker)
        for m in range(self._fibers):
            spec = self.getSpec(m)
            if spec.hasData() and m >= (min_y - 1) and m <= (max_y - 1):
                args = (par, select_wave, method, guess_window, spectral_res, ftol, xtol, 1)
                if cpus > 1:
                    pool.apply_async(spec.fitELines, args, callback=partial(extract_result, i=m))
                    sleep(0.01)
                else:
                    try:
                        result = spec.fitELines(*args)
                        extract_result(result, m)
                    except (ValueError, IndexError):
                        print("Fitting of spectrum %d failed." % m)

        if cpus > 1:
            pool.close()
            pool.join()

        return maps, fitted, fiber, rss_model

    def fit_Lib_Boots(self, lib_SSP, fiber, vel, disp, vel_err=None, disp_err=None, par_eline=None, select_wave_eline=None,
        mask_fit=None, method_eline='leastsq', guess_window=10.0, spectral_res=0.0, ftol=1e-4, xtol=1e-4, bootstraps=100,
        modkeep=80, parallel=1, verbose=False):
        """Bootstrap the spectra, while fixing the velocity and velocity
        dispersion, to determine the errors on the mass-weighted and luminosity
        weighted parameters. The parameters are age, mass-to-light ratio, [Fe/H]
        and [alpha/Fe]. If emission lines are fitted, the fit parameters on the
        emission lines will be bootstrapped too.

        Parameters
        ----------
        lib_SSP : SSPlibrary
            The library containing the template spectra.
        fiber : `numpy.ndarray`
            A 1D numpy array containing the indices of each spectrum of the
            `RSS` used in the bootstrapping.
        vel : `numpy.ndarray`
            The velocity in km/s to which each input spectra will be resampled.
        vel_disp : `numpy.ndarray`
            The velocity dispersion in km/s to which each input spectra will be
            broadened.
        vel_err : `numpy.ndarray`, optional
            ???Unused???
        disp_err : `numpy.ndarray`, optional
            ???Unused???
        par_eline : parFile, optional
            A list of emission lines and their parameters that will be included
            in the bootstrap fitting procedure.
        mask_fit : `numpy.ndarray`, optional
            A 1D boolean array representing any wavelength regions which are
            masked out during the bootstrapping of each spectrum with the
            (normalized) stellar templates.
        select_wave_eline : `numpy.ndarray`, optional
            A 1D boolean array where the True value represents which wavelength
            elements will be used in the emission line fitting. This argument is
            mandatory in case `par_eline` is not None.
        method_eline : {'leastsq', 'simplex'}, optional
            This argument specifies if ordinary least squares fitting
            (`leastsq`) should be applied, or if a downhill simplex algorithm
            (`simplex`) should be used.
        guess_window : float, optional
            The wavelength region in which the emission line will be fitted.
        spectral_res : float, optional
            The spectral resolution of the line.
        ftol : float, optional
            The maximum acceptable error for fit convergence.
        xtol : float, optional
            The relative acceptable error for fit convergence.
        bootstraps : int, optional
            The number of bootstraps runs that will be performed.
        modkeep : float
            The number of template spectra that will be keeped at each bootstrap
            run.
        parallel : {'auto', int}, optional
            If parallel is not equal to one, the python multiprocessing routine
            shall be used to run parts of the code in parallel. With the option
            `auto`, it adjusts the number of parallel processes to the number
            of cpu-cores available.

        Returns
        -------
        mass_weighted_pars_err : `numpy.ndarray`
            Mass-weighted averages stored in a 1D numpy array. The first
            value is age, the second value the mass-to-light ratio, the third
            value [Fe/H], and the fourth value [alpha/Fe], and the fifth value a
            zero.
        lum_weighted_pars_err : `numpy.ndarray`
            Luminosity-weighted averages stored in a 1D numpy array. The first
            value is age, the second value the mass-to-light ratio, the third
            value [Fe/H], and the fourth value [alpha/Fe], and the fifth value a
            zero.
        maps : list
            A dictionary for each emission line containing a dictionary for each
            parameter of that emission line with the results of the bootstrap.
        """
        coeffs = numpy.zeros((len(fiber), bootstraps, lib_SSP.getBaseNumber()), dtype=numpy.float32)

        if par_eline is not None:
            maps = {}
            for n in par_eline._names:
                model = {}
                if par_eline._profile_type[n] == 'Gauss':
                    model['flux_err'] = numpy.zeros(len(fiber), dtype=numpy.float32)
                    model['vel_err'] = numpy.zeros(len(fiber), dtype=numpy.float32)
                    model['fwhm_err'] = numpy.zeros(len(fiber), dtype=numpy.float32)
                maps[n] = model

        def extract_result(result, i):
            coeffs[i, :] = result[0]
            if verbose:
                print("Bootstrapping fiber %d finished." %i)
            if par_eline is not None:
                for n in par_eline._names:
                    if par_eline._profile_type[n] == 'Gauss':
                        maps[n]['flux_err'][i] = result[1][n]['flux']
                        maps[n]['vel_err'][i] = result[1][n]['vel']
                        maps[n]['fwhm_err'][i] = result[1][n]['fwhm']

        if parallel == 'auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus, maxtasksperchild=200, initializer=init_worker)
        for m in range(len(fiber)):
            spec = self.getSpec(fiber[m])
            vel_err_m = None if vel_err is None else vel_err[m]
            disp_err_m = None if disp_err is None else disp_err[m]
            args = (lib_SSP, vel[m], disp[m], vel_err_m, disp_err_m, par_eline,
                    select_wave_eline, mask_fit, method_eline, guess_window,
                    spectral_res, ftol, xtol, bootstraps, modkeep, 1)
            if cpus > 1:
                pool.apply_async(spec.fit_Lib_Boots, args, callback=partial(extract_result, i=m))
                sleep(0.01)
            else:
                result = spec.fit_Lib_Boots(*args)
                extract_result(result, m)

        if cpus > 1:
            pool.close()
            pool.join()

        if par_eline is None:
            maps = None
        return coeffs, maps


def loadRSS(infile, extension_data=None, extension_mask=None, extension_error=None):

    rss = RSS()
    rss.loadFitsData(infile, extension_data=extension_data, extension_mask=extension_mask, extension_error=extension_error)

    return rss
