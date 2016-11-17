from Paradise.lib.data import *
from Paradise.lib.spectrum1d import Spectrum1D
import numpy
from multiprocessing import cpu_count
from multiprocessing import Pool
from time import sleep
from functools import partial
import signal


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class Cube(Data):
    """A class representing 3D spectra.

    `Cube` is a subclass of Data which allows for handling and organizing a
    three-dimensional spectrum. The class supports reading and writing FITS
    files, resampling and rebinning, velocity shifting and broadening, the
    application of extinction corrections and various advanced fitting
    functions.

    Parameters
    ----------
    data : `numpy.ndarray`
        The spectra as a 2D numpy array structured such that the different
        spectra are located along the second and third dimension.
    wave : `numpy.ndarray`
        The wavelength elements corresponding to the different data points
        along the third dimension of the `data`.
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

    def getSpec(self, x, y):
        """Get a single spectrum from the RSS instance.

        Parameters
        ----------
        x : int
            The index of the spectrum along the first dimension.
        y : int
            The index of the spectrum along the second dimension.

        Returns
        -------
        spec : Spectrum1D
            The object containg the (x,y)-th spectrum, together with the proper
            error, mask and normalization.
        """

        wave = self._wave
        data = self._data[:, y, x]
        if self._error is not None:
            error = self._error[:, y, x]
        else:
            error = None
        if self._mask is not None:
            mask = self._mask[:, y, x]
        else:
            mask = None
        if self._normalization is not None:
            normalization = self._normalization[:, y, x]
        else:
            normalization = None
        spec = Spectrum1D(wave=wave, data=data, error=error, mask=mask, normalization=normalization)
        return spec

    def fit_Kin_Lib_simple(self, SSPLib, nlib_guess, vel_min, vel_max, disp_min, disp_max, min_x, max_x, min_y, max_y, mask_fit,
        iterations=2, mcmc_code='emcee', walkers=50, burn=1500, samples=4000, thin=2, verbose=False, parallel='auto'):
        """Fits template spectra according to Markov chain Monte Carlo
        algorithm. This uses the PyMC library. The MCMC code is runned to
        determine the velocity and the velocity dispersion while the
        coefficients for the best combination of stellar templates are
        determined by iteration and non-negative least sqaures fitting.

        Notes
        -----
        The output sample might be smaller than the input sample, due to any
        limits imposed by `min_x`, `max_x`, `min_y` and `max_y`.

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
        min_x : int
            The lowest index from the spectral data set along the third
            dimension that will be used in the fitting.
        max_x : int
            The highest index from the spectral data set along the third
            dimension that will be used in the fitting.
        min_y : int
            The lowest index from the spectral data set along the second
            dimension that will be used in the fitting.
        max_y : int
            The highest index from the spectral data set along the second
            dimension that will be used in the fitting.
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
            to the data. The variable is a 3D numpy array where the last two
            dimension represents the different spectra and the first dimension
            the coefficients for each template in `SSPLib`.
        chi2 : `numpy.ndarray`
            The chi^2 value between `bestfit_spec` and `data`.
        fiber : `numpy.ndarray`
            A 1D numpy array containing the indices of each output spectrum
            pointing to each input `RSS`.
        cube_model : `numpy.ndarray`
            The best fitted spectrum obtained from the linear
            combination of template spectra.
        """
        cube_model = numpy.zeros(self.getShape(), dtype=numpy.float32)
        mask = numpy.zeros(self.getShape(), dtype=numpy.bool)
        vel_fit = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.float32)
        vel_fit_err = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.float32)
        Rvel = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.float32)
        disp_fit = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.float32)
        disp_fit_err = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.float32)
        Rdisp = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.float32)
        chi2 = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.float32)
        x_pix = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.int16)
        y_pix = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.int16)
        fitted = numpy.zeros(self._dim_y * self._dim_x, dtype="bool")
        coeff = numpy.zeros((self._dim_y * self._dim_x, SSPLib.getBaseNumber()), dtype=numpy.float32)

        def extract_result(result, i, x, y):
            vel_fit[i] = result[0]
            vel_fit_err[i] = result[1]
            Rvel[i] = result[2]
            disp_fit[i] = result[3]
            disp_fit_err[i] = result[4]
            Rdisp[i] = result[5]
            fitted[i] = True
            coeff[i, :] = result[7]
            chi2[i] = result[8]
            x_pix[i] = x
            y_pix[i] = y
            cube_model[:, y, x] = result[6].unnormalizedSpec().getData()
            mask[:, y, x] = result[6].getMask()
            if verbose:
                print("Fitting of SSP(s) to spectrum (y, x) = (%d, %d) finished." % (y, x))
                print("vel_fit: %.3f  disp_fit: %.3f chi2: %.2f" % (vel_fit[i], disp_fit[i], chi2[i]))

        if parallel == 'auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus, maxtasksperchild=200, initializer=init_worker)
            results = []
        m = 0
        for x in range(self._dim_x):
            for y in range(self._dim_y):
                spec = self.getSpec(x, y)
                if spec.hasData() and x >= (min_x - 1) and x <= (max_x - 1) and y >= (min_y - 1) and y <= (max_y - 1):
                    args = (SSPLib, nlib_guess, vel_min, vel_max, disp_min,
                            disp_max, mask_fit, iterations, mcmc_code,
                            walkers, burn, samples, thin)
                    if cpus > 1:
                        results.append([x, y, pool.apply_async(spec.fit_Kin_Lib_simple, args, callback=partial(extract_result, i=m, x=x, y=y))])
                        sleep(0.01)
                    else:
                        try:
                            result = spec.fit_Kin_Lib_simple(*args)
                            extract_result(result, m, x, y)
                        except (ValueError, IndexError):
                            print("Fitting of spectrum (y, x) = (%d, %d) failed." % (y, x))
                m += 1

        if cpus > 1:
            pool.close()
            pool.join()
            for x, y, result in results:
                try:
                    result.get()
                except (ValueError, IndexError):
                    print("Fitting of spectrum (y, x) = (%d, %d) failed." % (y, x))

        return vel_fit, vel_fit_err, Rvel, disp_fit, disp_fit_err, Rdisp, fitted, coeff, chi2, x_pix, y_pix, cube_model, mask

    def fit_Lib_fixed_kin(self, SSPLib, nlib_guess, vel, vel_disp, x_pos,y_pos, min_x, max_x, min_y, max_y, mask_fit,
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
            value (`max_y` - `min_y`) * (`max_x` - `min_x`).
        min_x : int
            The lowest index from the spectral data set along the third
            dimension that will be used in the fitting.
        max_x : int
            The highest index from the spectral data set along the third
            dimension that will be used in the fitting.
        min_y : int
            The lowest index from the spectral data set along the second
            dimension that will be used in the fitting.
        max_y : int
            The highest index from the spectral data set along the second
            dimension that will be used in the fitting.
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
        cube_model : `numpy.ndarray`
            The best fitted spectrum obtained from the linear
            combination of template spectra.
        """
        cube_model = numpy.zeros(self.getShape(), dtype=numpy.float32)
        mask = numpy.zeros(self.getShape(), dtype=numpy.bool)
        chi2 = numpy.zeros(len(x_pos), dtype=numpy.float32)
        x_pix = numpy.zeros(len(x_pos), dtype=numpy.int16)
        y_pix = numpy.zeros(len(x_pos), dtype=numpy.int16)
        fitted = numpy.zeros(len(x_pos), dtype="bool")
        coeff = numpy.zeros((len(x_pos), SSPLib.getBaseNumber()), dtype=numpy.float32)

        def extract_result(result, i, x, y):
            fitted[i] = True
            coeff[i, :] = result[0]
            chi2[i] = result[2]
            x_pix[i] = x
            y_pix[i] = y
            cube_model[:, y, x] = result[1].unnormalizedSpec().getData()
            mask[:, y, x] = result[1].getMask()
            if verbose:
                print("Fitting of SSP(s) to spectrum (y, x) = (%d, %d) finished." % (y, x))
                print("chi2: %.2f" % (chi2[i]))

        if parallel == 'auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus, maxtasksperchild=200, initializer=init_worker)
            results = []
        for m in range(len(x_pos)):
            x = x_pos[m]
            y = y_pos[m]
            spec = self.getSpec(x, y)
            if spec.hasData() and x >= (min_x - 1) and x <= (max_x - 1) and y >= (min_y - 1) and y <= (max_y - 1):
                args = (SSPLib, nlib_guess, vel[m], vel_disp[m], mask_fit.maskPixelsObserved(spec.getWave(), vel[m] / 300000.0))
                if cpus > 1:
                    results.append([m, pool.apply_async(spec.fitSuperposition, args, callback=partial(extract_result, i=m, x=x, y=y))])
                    sleep(0.01)
                else:
                    try:
                        result = spec.fitSuperposition(*args)
                        extract_result(result, m, x, y)
                    except (ValueError, IndexError):
                        print("Fitting of spectrum (y, x) = (%d, %d) failed." % (y, x))

        if cpus > 1:
            pool.close()
            pool.join()
            for m, result in results:
                try:
                    result.get()
                except (ValueError, IndexError):
                    print("Fitting of spectrum (y, x) = (%d, %d) failed." % (y_pos[m], x_pos[m]))

        return fitted, coeff, chi2, x_pix, y_pix, cube_model, mask

    def fitELines(self, par, select_wave, min_x, max_x, min_y, max_y, method='leastsq', guess_window=0.0, spectral_res=0.0,
    ftol=1e-4, xtol=1e-4, verbose=1, parallel='auto'):
        """This routine fits a set of emission lines to each spectrum.

        Parameters
        ----------
        par : parFile
            The object containing all the constraints on the parameters.
        select_wave : numpy.ndarray
            A 1D boolean array where the True value represents which elements
            in the wave, data, error, mask and normalization.
        min_x : int
            The lowest index from the spectral data set along the third
            dimension that will be used in the fitting.
        max_x : int
            The highest index from the spectral data set along the third
            dimension that will be used in the fitting.
        min_y : int
            The lowest index from the spectral data set along the second
            dimension that will be used in the fitting.
        max_y : int
            The highest index from the spectral data set along the second
            dimension that will be used in the fitting.
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
        cube_model = numpy.zeros(self.getShape(), dtype=numpy.float32)
        fitted = numpy.zeros(self._dim_y * self._dim_x, dtype="bool")
        x_pix = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.int16)
        y_pix = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.int16)

        maps = {}
        for n in par._names:
            model = {}
            if par._profile_type[n] == 'Gauss':
                model['flux'] = numpy.zeros((self._dim_y * self._dim_x), dtype=numpy.float32)
                model['vel'] = numpy.zeros((self._dim_y * self._dim_x), dtype=numpy.float32)
                model['fwhm'] = numpy.zeros((self._dim_y * self._dim_x), dtype=numpy.float32)
            maps[n] = model

        def extract_result(result, i, x, y):
            fitted[i] = True
            x_pix[i] = x
            y_pix[i] = y
            cube_model[:, y, x] = result[1]
            if verbose:
                print("Fitting of emission lines to spectrum (y, x) = (%d, %d) finished." % (y, x))
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
        m = 0
        for x in range(self._dim_x):
            for y in range(self._dim_y):
                spec = self.getSpec(x, y)
                if spec.hasData() and x >= (min_x - 1) and x <= (max_x - 1) and y >= (min_y - 1) and y <= (max_y - 1):
                    args = (par, select_wave, method, guess_window, spectral_res, ftol, xtol, 1)
                    if cpus > 1:
                        pool.apply_async(spec.fitELines, args, callback=partial(extract_result, i=m, x=x, y=y))
                        sleep(0.01)
                    else:
                        try:
                            result = spec.fitELines(*args)
                            extract_result(result, m, x, y)
                        except (ValueError, IndexError):
                            print("Fitting of spectrum (y, x) = (%d, %d) failed." % (y, x))
                m += 1

        if cpus > 1:
            pool.close()
            pool.join()

        return maps, fitted, x_pix, y_pix, cube_model

    def fit_Lib_Boots(self, lib_SSP, x_cor, y_cor, vel, disp, vel_err=None, disp_err=None, par_eline=None, select_wave_eline=None,
        mask_fit=None, method_eline='leastsq', guess_window=10.0, spectral_res=0.0, ftol=1e-4, xtol=1e-4, bootstraps=100,
        modkeep=80, parallel=1, verbose=False):

        coeffs = numpy.zeros((len(x_cor), bootstraps, lib_SSP.getBaseNumber()),
                             dtype=numpy.float32)

        if par_eline is not None:
            maps = {}
            for n in par_eline._names:
                model = {}
                if par_eline._profile_type[n] == 'Gauss':
                    model['flux_err'] = numpy.zeros(len(x_cor), dtype=numpy.float32)
                    model['vel_err'] = numpy.zeros(len(x_cor), dtype=numpy.float32)
                    model['fwhm_err'] = numpy.zeros(len(x_cor), dtype=numpy.float32)
                maps[n] = model

        def extract_result(result, i, x, y):
            coeffs[i, :] = result[0]
            if verbose:
                print("Bootstrapping spectrum (y, x) = (%d, %d) finished." % (y, x))
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
        for m in range(len(x_cor)):
            spec = self.getSpec(x_cor[m], y_cor[m])
            vel_err_m = None if vel_err is None else vel_err[m]
            disp_err_m = None if disp_err is None else disp_err[m]
            args = (lib_SSP, vel[m], disp[m], vel_err_m, disp_err_m, par_eline,
                    select_wave_eline, mask_fit, method_eline, guess_window,
                    spectral_res, ftol, xtol, bootstraps, modkeep, 1)
            if cpus > 1:
                pool.apply_async(spec.fit_Lib_Boots, args, callback=partial(extract_result, i=m, x=x_cor[m], y=y_cor[m]))
                sleep(0.01)
            else:
                result = spec.fit_Lib_Boots(*args)
                extract_result(result, m, x_cor[m], y_cor[m])

        if cpus > 1:
            pool.close()
            pool.join()

        if par_eline is None:
            maps = None
        return coeffs, maps


def loadCube(infile, extension_data=None, extension_mask=None, extension_error=None):

    cube = Cube()
    cube.loadFitsData(infile, extension_data=extension_data, extension_mask=extension_mask, extension_error=extension_error)

    return cube
