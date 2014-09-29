from Paradise.lib.data import Data
from Paradise.lib.spectrum1d import Spectrum1D
import numpy
import pylab
from multiprocessing import cpu_count
from multiprocessing import Pool


class RSS(Data):
    def __init__(self, data=None, wave=None, error=None, mask=None, normalization=None, inst_fwhm=None, header=None):
        Data.__init__(self, wave=wave, data=data, error=error, mask=mask, normalization=normalization, inst_fwhm=inst_fwhm,
        header=header)

    def getSpec(self, i):
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
        iterations=2, burn=1500, samples=4000, thin=2, verbose=False, parallel='auto'):

        rss_model = numpy.zeros(self.getShape(), dtype=numpy.float32)
        vel_fit = numpy.zeros(self._fibers, dtype=numpy.float32)
        vel_fit_err = numpy.zeros(self._fibers, dtype=numpy.float32)
        disp_fit = numpy.zeros(self._fibers, dtype=numpy.float32)
        disp_fit_err = numpy.zeros(self._fibers, dtype=numpy.float32)
        chi2 = numpy.zeros(self._fibers, dtype=numpy.float32)
        fiber = numpy.zeros(self._fibers, dtype=numpy.int16)
        fitted = numpy.zeros(self._fibers, dtype="bool")
        coeff = numpy.zeros((self._fibers, SSPLib.getBaseNumber()), dtype=numpy.float32)
        if parallel == 'auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus, maxtasksperchild=200)
            result_fit = []
            m = 0
            while m < self._fibers:
                spec = self.getSpec(m)
                if spec.hasData() and m >= (min_y - 1) and m <= (max_y - 1):
                    result_fit.append(pool.apply_async(spec.fit_Kin_Lib_simple, args=(SSPLib, nlib_guess, vel_min, vel_max,
                    disp_min, disp_max, mask_fit, iterations, burn, samples, thin)))
                else:
                    result_fit.append(None)
                fiber[m] = m
                m += 1

            pool.close()
            pool.join()
            for  m in range(len(result_fit)):
                if result_fit[m] is not None:
                    try:
                        result = result_fit[m].get()
                        vel_fit[m] = result[0]
                        vel_fit_err[m] = result[1]
                        disp_fit[m] = result[2]
                        disp_fit_err[m] = result[3]
                        fitted[m] = True
                        coeff[m, :] = result[5]
                        chi2[m] = result[6]
                        rss_model[m, :] = result[4].unnormalizedSpec().getData()
                    except ValueError:
                        print "Fitting failed because of bad spectrum."
        else:
            m = 0
            while m < self._fibers:
                spec = self.getSpec(m)
                if spec.hasData() and m >= (min_y - 1) and m <= (max_y - 1):
                    if verbose:
                        print "Fitting Spectrum (%d) of RSS" % (m + 1)
                    try:
                        result = spec.fit_Kin_Lib_simple(SSPLib, nlib_guess, vel_min, vel_max, disp_min, disp_max, mask_fit,
                    iterations=iterations, burn=burn, samples=samples, thin=thin)
                        vel_fit[m] = result[0]
                        vel_fit_err[m] = result[1]
                        disp_fit[m] = result[2]
                        disp_fit_err[m] = result[3]
                        fitted[m] = True
                        coeff[m, :] = result[5]
                        chi2[m] = result[6]
                        fiber[m] = m
                        rss_model[m, :] = result[4].unnormalizedSpec().getData()
                        if verbose:
                            print "vel_fit: %.3f  disp_fit: %.3f chi2: %.2f" % (vel_fit[m], disp_fit[m], chi2[m])
                    except (ValueError, IndexError):
                        print "Fitting failed because of bad spectrum."

                m += 1
        return vel_fit, vel_fit_err, disp_fit, disp_fit_err, fitted, coeff, chi2, fiber, rss_model

    def fit_Lib_fixed_kin(self, SSPLib, x_pos, y_pos, vel, vel_disp, fibers, min_y, max_y, mask_fit,
        verbose=False, parallel='auto'):

        rss_model = numpy.zeros(self.getShape(), dtype=numpy.float32)
        chi2 = numpy.zeros(self._fibers, dtype=numpy.float32)
        fiber = numpy.zeros(self._fibers, dtype=numpy.int16)
        fitted = numpy.zeros(self._fibers, dtype="bool")
        coeff = numpy.zeros((self._fibers, SSPLib.getBaseNumber()), dtype=numpy.float32)
        if parallel == 'auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus, maxtasksperchild=200)
            result_fit = []
            m = 0
            while m < self._fibers:
                spec = self.getSpec(m)
                if spec.hasData() and m >= (min_y - 1) and m <= (max_y - 1):
                    result_fit.append(pool.apply_async(spec.fitSuperposition, args=(SSPLib, vel[fibers[m]], vel_disp[fibers[m]])))
                else:
                    result_fit.append(None)
                fiber[m] = m
                m += 1

            pool.close()
            pool.join()
            for  m in range(len(result_fit)):
                if result_fit[m] is not None:
                    try:
                        result = result_fit[m].get()
                        fitted[m] = True
                        coeff[m, :] = result[0]
                        chi2[m] = result[2]
                        rss_model[m, :] = result[1].unnormalizedSpec().getData()
                    except ValueError:
                        print "Fitting failed because of bad spectrum."
        else:
            m = 0
            while m < self._fibers:
                spec = self.getSpec(m)
                if spec.hasData() and m >= (min_y - 1) and m <= (max_y - 1):
                    if verbose:
                        print "Fitting Spectrum (%d) of RSS" % (m + 1)
                    try:
                        result = spec.fitSuperposition(SSPLib, vel, vel_disp)
                        fitted[m] = True
                        coeff[m, :] = result[0]
                        chi2[m] = result[2]
                        fiber[m] = m
                        rss_model[m, :] = result[1].unnormalizedSpec().getData()
                        if verbose:
                            print "vel_fit: %.3f  disp_fit: %.3f chi2: %.2f" % (vel_fit[m], disp_fit[m], chi2[m])
                    except (ValueError, IndexError):
                        print "Fitting failed because of bad spectrum."
                m += 1
        return fitted, coeff, chi2, fiber, rss_model

    def fitELines(self, par, select_wave, min_y, max_y, method='leastsq', guess_window=0.0, spectral_res=0.0,
    ftol=1e-4, xtol=1e-4, verbose=1, parallel='auto'):

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

        if parallel == 'auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus, maxtasksperchild=200)
            result_fit = []
            m = 0
            while m < self._fibers:
                spec = self.getSpec(m)
                if spec.hasData() and m >= (min_y - 1) and m <= (max_y - 1):
                    result_fit.append(pool.apply_async(spec.fitELines, args=(par, select_wave, method, guess_window,
                    spectral_res, ftol, xtol, 1)))
                else:
                    result_fit.append(None)
                    fiber[m] = m
                m += 1
            pool.close()
            pool.join()
            for  m in range(len(result_fit)):
                if result_fit[m] is not None:
                    result = result_fit[m].get()
                    rss_model[m, :] = result[1]
                    fitted[m] = True

                    for n in par._names:
                        if par._profile_type[n] == 'Gauss':
                            maps[n]['flux'][m] = result[0][n][0]
                            maps[n]['vel'][m] = result[0][n][1]
                            maps[n]['fwhm'][m] = result[0][n][2]
        else:
            m = 0
            while m < self._fibers:
                spec = self.getSpec(m)

                if spec.hasData() and m >= (min_y - 1) and m <= (max_y - 1):
                    if verbose:
                        print "Fitting Spectrum (%d) of RSS" % (m + 1)
                    result = spec.fitELines(par, select_wave, method=method, guess_window=guess_window,
                        spectral_res=spectral_res, ftol=ftol, xtol=xtol, parallel=1)
                    fiber[m] = m
                    fitted[m] = True
                    rss_model[m, :] = result[1]
                    for n in par._names:
                        if par._profile_type[n] == 'Gauss':
                            maps[n]['flux'][m] = result[0][n][0]
                            maps[n]['vel'][m] = result[0][n][1]
                            maps[n]['fwhm'][m] = result[0][n][2]
                m += 1
        return maps, fitted, fiber, rss_model

    def fit_Lib_Boots(self, lib_SSP, fiber, vel, disp, vel_err=None, disp_err=None, par_eline=None, select_wave_eline=None,
        mask_fit=None, method_eline='leastsq', guess_window=10.0, spectral_res=0.0, ftol=1e-4, xtol=1e-4, bootstraps=100,
        modkeep=80, parallel=1, verbose=False):

        mass_weighted_pars_err = numpy.zeros((len(fiber), 5), dtype=numpy.float32)
        lum_weighted_pars_err = numpy.zeros((len(fiber), 5), dtype=numpy.float32)

        if par_eline is not None:
            maps = {}
            for n in par_eline._names:
                model = {}
                if par_eline._profile_type[n] == 'Gauss':
                    model['flux_err'] = numpy.zeros(len(fiber), dtype=numpy.float32)
                    model['vel_err'] = numpy.zeros(len(fiber), dtype=numpy.float32)
                    model['fwhm_err'] = numpy.zeros(len(fiber), dtype=numpy.float32)
                maps[n] = model

        if parallel == 'auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus, maxtasksperchild=200)
            result_fit = []
            for m in range(len(fiber)):
                spec = self.getSpec(fiber[m])
                result_fit.append(pool.apply_async(spec.fit_Lib_Boots, args=(lib_SSP, vel[m], disp[m], None, None, par_eline,
                         select_wave_eline, mask_fit, method_eline, guess_window, spectral_res, ftol, xtol, bootstraps, modkeep, 1)))
            pool.close()
            pool.join()
            for  m in range(len(result_fit)):
                if result_fit[m] is not None:
                    result = result_fit[m].get()
                    mass_weighted_pars_err[m, :] = result[0]
                    lum_weighted_pars_err[m, :] = result[1]
                    if par_eline is not None:
                        for n in par_eline._names:
                            if par_eline._profile_type[n] == 'Gauss':
                                maps[n]['flux_err'][m] = result[2][n]['flux']
                                maps[n]['vel_err'][m] = result[2][n]['vel']
                                maps[n]['fwhm_err'][m] = result[2][n]['fwhm']
        else:
            for m in range(len(fiber)):
                spec = self.getSpec(fiber[m])
                if verbose:
                    print "Fitting Spectrum (%d) of RSS" % (fiber[m] + 1)
                result = spec.fit_Lib_Boots(lib_SSP, vel[m], disp[m], None, None, par_eline,
                     select_wave_eline, mask_fit, method_eline, guess_window, spectral_res, ftol, xtol, bootstraps, modkeep, 1)

                mass_weighted_pars_err[m, :] = result[0]
                lum_weighted_pars_err[m, :] = result[1]
                if par_eline is not None:
                    for n in par_eline._names:
                        if par_eline._profile_type[n] == 'Gauss':
                            maps[n]['flux_err'][m] = result[2][n]['flux']
                            maps[n]['vel_err'][m] = result[2][n]['vel']
                            maps[n]['fwhm_err'][m] = result[2][n]['fwhm']
        if par_eline is None:
            maps = None
        return mass_weighted_pars_err, lum_weighted_pars_err, maps


def loadRSS(infile, extension_data=None, extension_mask=None, extension_error=None):

    rss = RSS()
    rss.loadFitsData(infile, extension_data=None, extension_mask=None, extension_error=None)

    return rss
