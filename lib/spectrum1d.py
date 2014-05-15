import numpy
import pyfits
import fit_profile
from scipy import optimize
from scipy import ndimage
from scipy import interpolate
import time
import pymc
import pylab
from copy import deepcopy
import copy_reg
from types import *


class Spectrum1D(object):
    """A class representing 1D spectrum.

    `Spectrum1D` is a class which allows for handling and organizing a one-
    dimensional spectrum.

    Parameters
    ----------
    wave : `numpy.ndarray`
        The wavelength at each point of the `data` array.
    data : `numpy.ndarray`
        The spectrum. Should be of the same size as `wave`.
    error : `numpy.ndarray`, optional
        The error spectrum.
        If `error` equals None, it is assumed the error spetrum is not
        known.
    mask : `numpy.ndarray`
        A boolean array where True represents a masked (invalid) data point
        and False a good data point.
    normalization : `numpy.ndarray`
        An array which is used to normalize the data/error; both data and
        error are divided by `normalization`.
    inst_fwhm : float
        The instrumental FWHM in the same units as `wavelength`.
    """
    def __init__(self, wave=None, data=None, error=None, mask=None, normalization=None, inst_fwhm=None):
        self.__wave = wave
        self.__data = data
        if data is not None:
            self.__dim = self.__data.shape[0]
            self.__pixels = numpy.arange(self.__dim)
        self.__error = error
        self.__mask = mask
        self.__normalization = normalization
        self.__inst_fwhm = inst_fwhm

    def getWave(self):
        return self.__wave

    def getData(self):
        return self.__data

    def setData(self, data):
        self.__data = data

    def getError(self):
        return self.__error

    def setError(self, error):
        self.__error = error

    def hasData(self, start_wave=None, end_wave=None):
        if start_wave is not None:
            self.__mask = numpy.logical_or(self.__mask, self.__wave<start_wave)
        if end_wave is not None:
            self.__mask = numpy.logical_or(self.__mask, self.__wave>end_wave)
        if numpy.sum(self.__data) != 0 and numpy.sum(self.__mask) != len(self.__data):
            return True
        else:
            return False

    def getMask(self):
        return self.__mask

    def setMask(self, mask):
        self.__mask = mask

    def getNormalization(self):
        return self.__normalization

    def setNormalization(self, normalization):
        self.__normalization = normalization

    def applyNormalization(self, normalization):
        if self.__normalization is None:
            self.__data = self.__data / normalization
            if self.__error is not None:
                self.__error = self.__error / normalization

    def subWave(self,select_wave):
        """Obtain a copy of Spectrum1D within a certain wavelength range.

        Parameters
        ----------
        select_wave : numpy.ndarray
            A 1D boolean array where the True value represents which elements
            in the `wave`, `data`, `error`, `mask` and `normalization`

        Returns
        -------
        spec : Spectrum1D
            A new `Spectrum1D` instance containing only the elements
            according to `select_wave`.
        """
        if self.getError() is not None:
            new_error = self.getError()[select_wave]
        else:
            new_error = None
        if self.getMask() is not None:
            new_mask = self.getMask()[select_wave]
        else:
            new_mask = None
        if self.getNormalization() is not None:
            new_normalization = self.getNormalization()[select_wave]
        else:
            new_normalization = None
        spec = Spectrum1D(wave=self.getWave()[select_wave], data=self.getData()[select_wave], error=new_error, mask=new_mask,
            normalization=new_normalization, inst_fwhm=self.getFWHM())
        return spec

    def getFWHM(self):
        return self.__inst_fwhm

    def setFWHM(self, FWHM):
        self.__inst_fwhm = FWHM

    def setVelSampling(self, vel_sampling):
        self.__vel_sampling = vel_sampling

    def getVelSampling(self):
        return self.__vel_sampling

    def resampleSpec(self, ref_wave, method='spline', err_sim=500, replace_error=1e10):
        """Returns a new Spectrum1D object resampled to a new wavelength grid

        The interpolation is perfomed based on the `method`-argument.

        Parameters
        ----------
        ref_wave : numpy.ndarray
            The new wavelength array to which the spectrum will be
            resampled.
        method : {'spline', 'linear', 'hann', 'lanczos2', 'lanczos3'}, optional
            Specifies the resampling method.
            'spline' : 3rd degree spline interpolation without smoothing
            'linear' : linear interpolation
            'hann' : *Not implemented*, uses a Hann window for resampling
            'lanczos2' : *Not implemented*, uses a Lanczos kernel for
            resampling with the window parameter set at a = 2.
            'lanczos3' : *Not implemented*, uses a Lanczos kernel for
            resampling with the window parameter set at a = 3. It is
            therefore broader than `lanczos2`-kernel.
        err_sim : *Not implemented* int
        replace_error : *Not implemented* int

        Returns
        -------
        spec : Spectrum1D
            A new `Spectrum1D` instance resampled to the new wavelength grid.
            Currently, the masks, errors and normalization are not
            propagated in `spec`.
        """
        # perform the interpolation on the data
        if method == 'spline':
            intp = interpolate.UnivariateSpline(self.__wave, self.__data, s=0, k=3)
            new_data = intp(ref_wave)
        elif method == 'linear':
            intp = interpolate.UnivariateSpline(self.__wave, self.__data, s=0, k=1)
            new_data = intp(ref_wave)
 #           elif method=='hann':

#            elif method=='lanczos2':

#            elif method=='lanczos3':

#        if self.__error!=None and err_sim>0:
#            sim  = numpy.zeros((err_sim, len(ref_wave)), dtype=numpy.float32)
#            data = numpy.zeros(len(self.__wave), dtype=numpy.float32)
#
#            for i in range(err_sim):
#                data = numpy.random.normal(self.__data, self.__error).astype(numpy.float32)
#                if method=='spline':
#                    intp = interpolate.UnivariateSpline(self.__wave, data, s=0)
#                    out =intp(ref_wave)
#                elif method=='linear':
#                    intp = interpolate.UnivariateSpline(self.__wave, data, k=1, s=0)
#                    out = intp(ref_wave)
#
#                sim[i, :] = out
#            new_error = numpy.std(sim, 0)
#
#
#        elif self._error==None or err_sim==0:
#            new_error=None
        spec_out = Spectrum1D(wave=ref_wave, data=new_data)
        return spec_out

    def rebinLogarithmic(self):
        """Rebin the spectrum from a linear wavelength grid to a logarithmic
        wavelength grid.

        Returns
        -------
        spec : Spectrum1D
            A new Spectrum1D object with the `data` in logarithmic base.
        """
        wave_log = 10 ** numpy.arange(numpy.log10(self.__wave[0]), numpy.log10(self.__wave[-1]), (numpy.log10(self.__wave[-1])
        - numpy.log10(self.__wave[0])) / len(self.__wave))
        new_spec = self.resampleSpec(wave_log)
        new_spec.setVelSampling((self.__wave[1] - self.__wave[0]) / self.__wave[0] * 300000.0)
        return new_spec

    def applyGaussianLOSVD(self, vel, disp_vel):
        """Returns a broadened copy of the spectrum.
        
        Parameters
        ----------
        vel : float
            The `wave` is redshifted according to `vel`, which should
            be specified in km/s.
        disp_vel : float
            The velocity dispersion value in km/s, which will be used
            for the Gaussian broadening of the data.

        Returns
        -------
        spec : Spectrum1D
            A new Spectrum1D object which is redshifted and broadened.

        Notes
        -----
        No correction is applied to `error`, `mask`, and
        `normalization`.
        """
        disp_pix = disp_vel / self.__vel_sampling
        new_data = ndimage.filters.gaussian_filter(self.__data, disp_pix, mode='constant')
        wave = self.__wave * (1 + vel / 300000.0)
        spec_out = Spectrum1D(wave=wave, data=new_data, error=None)
        try:
            spec_out.__vel_sampling = self.__vel_sampling
        except:
            pass
        return spec_out

    def correctExtinction(self, A_V, mode='correct', law='Cardelli', R_V=3.1):
        micron = self.__wave / 10000.0
        wave_number = 1.0 / micron
        y = wave_number - 1.82
        ax = 1 + (0.17699 * y) - (0.50447 * y ** 2) - (0.02427 * y ** 3) + (0.72085 * y ** 4) + (0.01979 * y ** 5)
        - (0.77530 * y ** 6) + (0.32999 * y ** 7)
        bx = (1.41338 * y) + (2.28305 * y ** 2) + (1.07233 * y ** 3) - (5.38434 * y ** 4) - (0.62251 * y ** 5)
        + (5.30260 * y ** 6) - (2.09002 * y ** 7)
        Arat = ax + (bx / R_V)
        Alambda = Arat * A_V
        cor_factor = 10 ** (Alambda / -2.5)
        if mode == 'correct':
            data = self.__data / cor_factor
            if self.__error is not None:
                error = self.__error / cor_factor
            else:
                error = None
        elif mode == 'apply':
            data = self.__data * cor_factor
            if self.__error is not None:
                error = self.__error * cor_factor
            else:
                error = None
        spec_out = Spectrum1D(data=data, wave=self.__wave, error=error)
        try:
            spec_out.__vel_sampling = self.__vel_sampling
        except:
            pass
        return spec_out

    def randomizeSpectrum(self):
        if self.__error is not None:
            data_new = numpy.random.Normal(self.__data, self.__error)
            spec_out = Spectrum1D(data=data_new, error=self.__error, mask=self.__mask)
            try:
                spec_out.__vel_sampling = self.__vel_sampling
            except:
                pass
        else:
            spec_out = self

    def applyKin(self, vel, disp_vel, wave):
        tempSpec = self.applyGaussianLOSVD(vel, disp_vel)
        tempSpec = tempSpec.resampleSpec(wave)
        return tempSpec

    def normalizeSpec(self, pixel_width, mask_norm=None):
        mean = numpy.zeros(len(self.__data), dtype=numpy.float32)
        data_temp = numpy.zeros(len(self.__data), dtype=numpy.float32)
        data_temp[:] = self.__data
        if self.__mask is not None and mask_norm is not None:
            mask = (self.__mask) & (mask_norm)
            select_bad = mask
        elif mask_norm is not None:
            mask = mask_norm
            select_bad = mask
        else:
            mask = numpy.zeros(self.getWave().shape,dtype='bool')
            select_bad = mask
        data_temp[select_bad] = 0.0
        uniform = ndimage.filters.convolve(data_temp, numpy.ones(pixel_width, dtype=numpy.int16), mode='nearest')
        summed = ndimage.filters.generic_filter(numpy.logical_not(mask).astype('int16'), numpy.sum, pixel_width, mode='nearest')
        select = summed > 0
        mean[select] = uniform[select] / summed[select].astype('float32')
        mean[numpy.logical_not(select)] = 1
        select_zero = mean == 0
        mean[select_zero] = 1
        new_data = self.__data / mean
        if self.__error is not None:
            new_error = self.__error / mean
        else:
            new_error = None
        spec_out = Spectrum1D(wave=self.__wave, data=new_data, error=new_error, mask=self.__mask, normalization=mean)
        return spec_out

    def unnormalizedSpec(self):
        data = self.__data * self.__normalization
        if self.__error is not None:
            error = self.__error * self.__normalization
        else:
            error = None
        spec_out = Spectrum1D(wave=self.__wave, data=data, error=error, mask=self.__mask, normalization=None)
        return spec_out

    def fitSuperposition(self, SSPLibrary, negative=False):
        valid_pix = numpy.logical_not(self.__mask)
        if self.__error is None:
            error = numpy.ones((self.__dim), dtype=numpy.float32)
        else:
            error = self.__error
        if len(SSPLibrary.getWave())!=len(self.__wave) or numpy.sum(SSPLibrary.getWave() - self.__wave)!=0.0:
            tempLib = SSPLibrary.resampleBase(self.__wave)
        else:
            tempLib = SSPLibrary
        libFit = fit_profile.fit_linearComb(tempLib.getBase())
        libFit.fit(self.__data, error, self.__mask, negative=negative)
        bestfit_spec = Spectrum1D(self.__wave, data=libFit())
        chi2 = libFit.chisq(self.__data, sigma=error, mask=self.__mask)
        return libFit.getCoeff(), bestfit_spec, chi2

    def fitKinMCMC_fixedpop(self, spec_model, vel_min, vel_max, vel_disp_min, vel_disp_max, burn=1000, samples=3000, thin=2):
        valid_pix = numpy.logical_not(self.__mask)
        wave = self.__wave[valid_pix]

        vel = pymc.Uniform('vel', lower=vel_min, upper=vel_max)
        disp = pymc.Uniform('disp', lower=vel_disp_min, upper=vel_disp_max)

        @pymc.deterministic(plot=False)
        def m(vel=vel, disp=disp):
            return spec_model.applyKin(vel, disp, wave).getData()
        d = pymc.Normal('d', mu=m, tau=self.__error[valid_pix] ** (-2), value=self.__data[valid_pix], observed=True)
        #d = pymc.Normal('d', mu=m, tau=0, value=self.__data[valid_pix], observed=True)
        M = pymc.MCMC([vel, disp, m, d])
        M.use_step_method(pymc.AdaptiveMetropolis, [vel, disp])
        #M.MAP(m)
        #M.fit()
        M.sample(burn=burn, iter=samples, thin=thin, progress_bar=False)
        return M

    def fit_Kin_Lib_simple(self, lib_SSP, nlib_guess, vel_min, vel_max, disp_min, disp_max, mask_fit=None,
         iterations=3, burn=50, samples=200, thin=1):
        spec_lib_guess = lib_SSP.getSpec(nlib_guess)
        if mask_fit is not None:
            self.setMask(numpy.logical_or(self.getMask(), mask_fit))
        for i in range(iterations):
            M = self.fitKinMCMC_fixedpop(spec_lib_guess, vel_min, vel_max, disp_min, disp_max, burn=burn, samples=samples, thin=thin)
            trace_vel = M.trace('vel', chain=None)[:]
            trace_disp = M.trace('disp', chain=None)[:]
            vel = numpy.mean(trace_vel)
            vel_err = numpy.std(trace_vel)
            disp = numpy.mean(trace_disp)
            disp_err = numpy.std(trace_disp)
            lib_vel = lib_SSP.applyGaussianLOSVD(vel, disp)
            (coeff, bestfit_spec, chi2) = self.fitSuperposition(lib_vel)
            spec_lib_guess = lib_SSP.compositeSpectrum(coeff)
        bestfit_spec.setNormalization(self.getNormalization())
        return vel, vel_err, disp, disp_err, bestfit_spec, coeff, chi2

    def fit_Lib_Boots(self, lib_SSP, vel, disp, vel_err=None, disp_err=None, par_eline=None, select_wave_eline=None,
        method_eline='leastsq', guess_window=10.0, spectral_res=0.0, ftol=1e-4, xtol=1e-4, bootstraps=100, modkeep=80,
        parallel=1):
        mass_weighted_pars = numpy.zeros((bootstraps, 5), dtype=numpy.float32)
        lum_weighted_pars = numpy.zeros((bootstraps, 5), dtype=numpy.float32)
        kin_SSP = lib_SSP.applyGaussianLOSVD(vel, disp).resampleBase(self.getWave())
        if par_eline is not None:
            line_models = {}
            for n in par_eline._names:
                model = {}
                if par_eline._profile_type[n] == 'Gauss':
                    model['flux'] = numpy.zeros(bootstraps, dtype=numpy.float32)
                    model['vel'] = numpy.zeros(bootstraps, dtype=numpy.float32)
                    model['fwhm'] = numpy.zeros(bootstraps, dtype=numpy.float32)
                line_models[n] = model
        m = 0
        try:
            while m < bootstraps:
                (sub_SSPlib, select) = kin_SSP.randomSubLibrary(float(modkeep / 100.0))
                select_bad = self.getError() <= 0
                self.__error[select_bad] = 1e+10
                self.__mask[select_bad] = True
                spec = Spectrum1D(wave=self.getWave(), data=numpy.random.normal(self.getData(), self.getError()),
                error=self.getError(), mask=self.getMask(), normalization=self.getNormalization())
                (coeff, bestfit_spec, chi2) = spec.fitSuperposition(sub_SSPlib)
                factors = numpy.zeros(kin_SSP.getBaseNumber(), dtype=numpy.float32)
                factors[select] = coeff
                if numpy.sum(coeff) > 0:
                    mass_weighted_pars[m, :] = kin_SSP.massWeightedPars(factors)
                    lum_weighted_pars[m, :] = kin_SSP.lumWeightedPars(factors)
                    if par_eline is not None:
                        res = Spectrum1D(wave=self.getWave(), data=(spec.getData() - bestfit_spec.getData()), error=spec.getError(),
                        mask=spec.getMask(), normalization=self.getNormalization()).unnormalizedSpec()
                        (out_model, best_fit, best_res) = res.fitELines(par_eline, select_wave_eline, method=method_eline,
                        guess_window=guess_window, spectral_res=spectral_res, ftol=ftol, xtol=xtol, parallel=parallel)
                        for n in par_eline._names:
                            if par_eline._profile_type[n] == 'Gauss':
                                line_models[n]['flux'][m] = out_model[n][0]
                                line_models[n]['vel'][m] = out_model[n][1]
                                line_models[n]['fwhm'][m] = out_model[n][2] * 2.354
                    else:
                        line_models = None
                    m += 1
                else:
                    if line_models is not None:
                        lines_error = {}
                        for n in par_eline._names:
                            error = {}
                            if par_eline._profile_type[n] == 'Gauss':
                                error['flux'] = numpy.nan
                                error['vel'] = numpy.nan
                                error['fwhm'] = numpy.nan
                            lines_error[n] = error
                    else:
                        lines_error = None
                    return numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan,numpy.nan]), numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan,numpy.nan]), lines_error
        except RuntimeError:
            if line_models is not None:
                lines_error = {}
                for n in par_eline._names:
                    error = {}
                    if par_eline._profile_type[n] == 'Gauss':
                        error['flux'] = numpy.nan
                        error['vel'] = numpy.nan
                        error['fwhm'] = numpy.nan
                    lines_error[n] = error
            else:
                lines_error = None
            return numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan,numpy.nan]), numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan,numpy.nan]), lines_error

        if line_models is not None:
            lines_error = {}
            for n in par_eline._names:
                error = {}
                if par_eline._profile_type[n] == 'Gauss':
                    error['flux'] = numpy.std(line_models[n]['flux'])
                    error['vel'] = numpy.std(line_models[n]['vel'])
                    error['fwhm'] = numpy.std(line_models[n]['fwhm'])
                lines_error[n] = error
        else:
            lines_error = None
        return numpy.std(mass_weighted_pars,0), numpy.std(lum_weighted_pars,0), lines_error

    def fitParFile(self, par, err_sim=0, ftol=1e-8, xtol=1e-8, method='leastsq', parallel='auto'):
        static_par = deepcopy(par)

        if self.getError() is not None:
            sigma = self.getError()
        else:
            sigma = 1.0
        par.fit(self.getWave(), self.getData(), sigma=sigma, err_sim=err_sim, maxfev=1000, method=method, ftol=ftol, xtol=xtol,
        parallel=parallel)
        par.restoreResult()
        if err_sim > 0 and self._error is not None:
            par_err = deepcopy(static_par)
            par_err._par = par._par_err
            par_err.restoreResult()
            par._parameters_err = par_err._parameters

    def fitELines(self, par, select_wave, method='leastsq', guess_window=0.0, spectral_res=0.0, ftol=1e-4,
    xtol=1e-4, parallel='auto'):
        spec_res = float(spectral_res / 2.354)
        #par = fit_profile.parFile(par_file, spec_res)
        fit_par = deepcopy(par)
        spec_fit = self.subWave(select_wave)
        if guess_window != 0.0:
            fit_par._guess_window = int(guess_window)
            fit_par.guessPar(self.getWave(), self.getData())

        spec_fit.fitParFile(fit_par, err_sim=0, method=method, ftol=ftol, xtol=xtol, parallel=parallel)
        best_fit = fit_par(self.getWave())
        best_res = self.getData() - best_fit
        out_model = {}
        for n in fit_par._names:
            if fit_par._profile_type[n] == 'Gauss':
                out_model[n] = (fit_par._parameters[n]['flux'], fit_par._parameters[n]['vel'],
                numpy.fabs(fit_par._parameters[n]['disp']) * 2.354)
        return out_model, best_fit, best_res

    def fit_AVgrid_Lib(self, lib_SSP, AV_start=0.0, AV_end=3.0, AV_step=0.1, mask_fit=None, bootstrap=0, prob=0.0):
        AVgrid = numpy.arange(AV_start, AV_end + AV_step, AV_step)
        chisq = numpy.zeros(len(AVgrid), dtype=numpy.float32)
        lib_weights = numpy.zeros((lib_SSP.getBaseNumber(), len(AVgrid)), dtype=numpy.float32)

        if mask_fit is not None:
            mask = numpy.logical_or(self.__mask, mask_fit)
        else:
            mask = self.__mask
        tempLib = lib_SSP.resampleBase(self.__wave)
        for i in range(len(AVgrid)):
            spec_cor = self.correctExtinction(AVgrid[i], mode='correct')
            spec_cor.__mask = mask
            (coeff, bestfit_spec, chi2) = spec_cor.fitSuperposition(tempLib)
            #pylab.plot(spec_cor.getWave(), spec_cor.getData(),'-k')
            #pylab.plot(bestfit_spec.getWave(), bestfit_spec.getData(),'-r')
            #pylab.show()
            lib_weights[:, i] = coeff
            chisq[i] = chi2
            if i>1 and chisq[i] > chisq[i-1]:
                break
        #pylab.plot(AVgrid,chisq,"ok")
        #pylab.show()
        select_best = numpy.min(chisq[chisq>0]) == chisq
        best_AV = AVgrid[select_best]
        best_chisq = chisq[select_best]
        best_coeff = lib_weights[:, select_best][:, 0]
        best_spec = (lib_SSP.compositeSpectrum(best_coeff)).correctExtinction(best_AV, mode='apply')

        return best_AV, best_coeff, best_spec, best_chisq




def _pickle_method(method):
      func_name = method.im_func.__name__
      obj = method.im_self
      cls = method.im_class
      if func_name.startswith('__') and not func_name.endswith('__'):
	cls_name = cls.__name__.lstrip('_')
	if cls_name: func_name = '_' + cls_name + func_name
      return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
      for cls in cls.mro():
	try:
	  func = cls.__dict__[func_name]
	except KeyError:
	  pass
	else:
	  break
      return func.__get__(obj, cls)
copy_reg.pickle(MethodType,_pickle_method, _unpickle_method)
