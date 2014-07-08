import pyfits
import numpy
import pymc
import pylab
import Paradise
from scipy import ndimage
from scipy import interpolate
from scipy import special
from scipy import optimize
from UserDict import UserDict


class SSPlibrary(UserDict):
    """A library of template spectra that can be used to fit an observed
    spectrum.

    Parameters
    ----------
    data : numpy.ndarray, optional
        A 2D numpy array containing the template spectra. This parameter
        will be ignored when `filename` is provided.
    wave : numpy.ndarray, optional
        A 1D numpy array with the wavelength values corresponding to
        `data`. This parameter will be ignored when `filename` is provided.
    spectralFWHM : numpy.ndarray, optional
        The spectral resolution in the same units as `wave`. This
        parameter will be ignored when `filename` is provided.
    infoSSP : pyfits.tableHDU, optional
        A table containing information on the template spectra. This
        parameter will be ignored when `filename` is provided.
    coefficients : nummpy.ndarray, optional
        The weights that each template spectrum has???
    normalization : bool???, optional
        Whether the spectrum is normalized???
    filename : string, optional
        A pyfits-file containing the template spectra.
    """
    def __init__(self, data=None, wave=None, spectralFWHM=None, infoSSP=None,
         coefficients=None, normalization=None, filename=None):
        UserDict.__init__(self)
        if filename is not None:
            hdu = pyfits.open(filename)
            hdr = hdu[0].header
            self.__data = hdu[0].data.T
            self.__nbasis = self.__data.shape[1]
            self.__wave = (numpy.arange(hdr['NAXIS1']) - (hdr['CRPIX1'] - 1)) * hdr['CDELT1'] + hdr['CRVAL1']
            self.__spectralFWHM = hdr['SPECFWHM']
            self.__normalization = normalization
            infoSSP = hdu[1].data
            columns = infoSSP.names
            for column in columns:
                self[column] = infoSSP.field(column)
            hdu.close()

        elif data is not None and wave is not None and spectralFWHM is not None and infoSSP is not None:
            self.__data = data
            self.__nbasis = self.__data.shape[1]
            self.__wave = wave
            self.__spectralFWHM = spectralFWHM
            self.__normalization = normalization
            for key in infoSSP.keys():
                self[key] = infoSSP[key]

        if coefficients is None:
            self.__coefficients = numpy.zeros(self.__nbasis, dtype=numpy.float32)
        else:
            self.__coefficients = coefficients
        self.__vel_sampling = None

    def getBase(self):
        """Obtain all template spectra as a 2D numpy array."""
        return self.__data

    def getBaseNumber(self):
        """Obtain the number of template spectra."""
        return self.__nbasis

    def getWave(self):
        """Obtain the wavelength grid as a 1D numpy array."""
        return self.__wave

    def getSpec(self, i):
        """Obtain the ith template spectrum as a Spectrum1D object."""
        spec = Paradise.Spectrum1D(wave=self.__wave, data=self.__data[:, i])
        try:
            spec.setVelSampling(self.getVelSampling())
        except:
            pass
        return spec

    def getCoefficients(self):
        """Get the weights/coefficients associated with the template spectra
        as a 1D numpy array."""
        return self.__coefficients

    def setCoefficients(self, coeff):
        """Change the weights/coefficients associated with the template spectra
        by providing a 1D numpy array."""
        self.__coefficients = coeff

    def getNormalization(self):
        """Obtain the normalization applied to all the template spectra as a
        2D numpy array."""
        return self.__normalization

    def getData(self):
        """Has the same functionality as SSPlibrary.getBase()."""
        return self.__data

    def setVelSampling(self, vel_sampling):
        """Adjust the velocity sampling applied to the template spectra."""
        self.__vel_sampling = vel_sampling

    def getVelSampling(self):
        """Obtain the velocity sampling applied to the template spectra."""
        return self.__vel_sampling

    def subWaveLibrary(self, min_wave=None, max_wave=None):
        """Applies a wavelength cut and returns a new library with the
        wavelength cut applied to.

        Parameters
        ----------
        min_wave : float, optional
            The minimum wavelength of the new library.
        max_wave : float, optional
            The maximum wavelength of the new library.

        Returns
        -------
        new_SSP : SSPlibrary
            A new instance where the wavelength cut has been applied to.
        """
        select_wave = numpy.ones(len(self.__wave), dtype='bool')
        if min_wave is not None:
            select_wave[self.__wave <= min_wave] = False
        if max_wave is not None:
            select_wave[self.__wave >= max_wave] = False
        infoSSP = {}
        for key in infoSSP.keys():
            infoSSP[key] = self[key]
        new_SSP = SSPlibrary(data=self.__data[select_wave, :], wave=self.__wave[select_wave], spectralFWHM=self.__spectralFWHM,
            infoSSP=infoSSP, coefficients=self.__coefficients)
        new_SSP.__vel_sampling = self.__vel_sampling
        return new_SSP

    def subLibrary(self, select):
        """Obtain a library where only a selected sample of template spectra is
        used.

        Parameters
        ----------
        select : numpy.ndarray
            An array with the indices of the template spectra that will be
            saved.

        Returns
        -------
        new_SSP : SSPlibrary
            A new instance with only the selected sample of template spectra.
        """
        data = self.__data[:, select]
        coefficients = self.__coefficients[select]
        infoSSP = {}
        for key in infoSSP.keys():
                infoSSP[key] = self[key][select]
        new_SSP = SSPlibrary(data=data, wave=self.__wave, spectralFWHM=self.__spectralFWHM,
        infoSSP=infoSSP, coefficients=coefficients)
        new_SSP.__vel_sampling = self.__vel_sampling
        return new_SSP

    def randomSubLibrary(self, modkeep):
        """Obtain a library with a random sample of template spectra.

        Parameters
        ----------
        modkeep : float
            The percentage of template spectra that will be saved in the random
            sample.

        Returns
        -------
        new_SSP : SSPlibrary
            A new instance with only the randomized sample of template spectra.
        select : numpy.ndarray
            The indices of the selected sample of template spectra.
        """
        select = (numpy.random.random(self.getBaseNumber()) <= modkeep)
        return self.subLibrary(select), select

    def normalizeBase(self, pixel_width, exclude_obj=None, redshift=None):
        """This function returns a normalized version of the template spectra by
        division through a running mean.

        Parameters
        ----------
        pixel_width : int
            The window over which the running mean will be calculated.
        exclude_obj : CustomMasks, optional
            The intervals that will be used to mask out regions.
        redshift : float, optional
            The redshift information will be used to shift the `exclude_obj`
            information to the right wavelengths.

        Returns
        -------
        new_SSP : SSPlibrary
            A new instance with only the template spectra normalized.
        """
        if exclude_obj is not None and redshift is not None:
            mask = exclude_obj.maskPixelsRest(self.__wave, redshift)
        else:
            mask = numpy.zeros(len(self.__wave), dtype="bool")
        mean = numpy.zeros((self.__data.shape), dtype=numpy.float32)
        data_temp = numpy.zeros((self.__data.shape), dtype=numpy.float32)
        data_temp[:] = self.__data
        select_bad = mask == True
        data_temp[select_bad] = 0.0
        uniform = ndimage.filters.convolve1d(data_temp, numpy.ones(pixel_width, dtype=numpy.int16), axis=0, mode='nearest')
        summed = ndimage.filters.generic_filter(numpy.logical_not(mask).astype('int16'), numpy.sum, pixel_width, mode='nearest')
        select = summed > 0
        mean[select, :] = uniform[select, :] / summed[select][:, numpy.newaxis]
        mean[numpy.logical_not(select), :] = 1
        select_zero = mean == 0
        mean[select_zero] = 1
        new_data = self.__data / mean
        new_SSP = SSPlibrary(data=new_data, wave=self.__wave, spectralFWHM=self.__spectralFWHM, infoSSP=self,
        coefficients=self.__coefficients, normalization=mean)
        return new_SSP

    def unnormalizedBased(self):
        """Returns an object in which the normalization is removed from the
        template spectra.
        """
        new_SSP = SSPlibrary(data=self.__data * self.__normalization, wave=self.__wave, spectralFWHM=self.__spectralFWHM,
             infoSSP=self, coefficients=self.__coefficients, normalization=None)
        return new_SSP

    def compositeSpectrum(self, coefficients=None):
        """Calculates a spectrum by multiplying every template spectra with
        weights and then adding them together.

        Parameters
        ----------
        coefficients : numpy.ndarray, optional
            The coefficients that will be used to combine the template spectra
            into a single spectrum.

        Returns
        -------
        compositeSpec : Spectrum1D
            The spectrum composed by adding the template spectra weighted by the
            coefficients.
        """
        if coefficients is not None:
            coeff = coefficients
        else:
            coeff = self.__coefficients
        compositeSpec = Paradise.Spectrum1D(wave=self.__wave, data=numpy.dot(self.__data, coeff))
        try:
            compositeSpec.setVelSampling(self.__vel_sampling)
        except:
            pass
        return compositeSpec

    def lumWeightedPars(self, coefficients, min_age=None, max_age=None):
        """From the weights of the template spectra in the library, this
        function computes the parameters by taking luminosity-weighted averages.

        Parameters
        ----------
        coefficients : numpy.ndarray
            The weights that each template spectrum has.
        min_age : float, optional
            Ignore the spectra with ages below a certain threshold.
        max_age : float, optional
            Ignore the spectra with ages above a certain threshold.

        Returns
        -------
        mean : numpy.ndarray
            Luminosity-weighted averages stored in a 1D numpy array. The first
            value is age, the second value the mass-to-light ratio, the third
            value [Fe/H], and the fourth value [alpha/Fe]
        """
        parameters = ['age', 'mass-to-light', '[Fe/H]', '[A/Fe]']

        select_age = self['age'] > 0
        if min_age is not None:
            select_age[self['age'] < min_age] = False
        if max_age is not None:
            select_age[self['age'] > max_age] = False

        if numpy.sum(select_age)>0:
            if self.__normalization is not None:
                mean_out = [numpy.sum(coefficients[select_age])]
            else:
                mean_out = [numpy.sum(self['luminosity'][select_age] * coefficients[select_age]) /
                numpy.sum(self['luminosity'] * coefficients)]
            for par in parameters:
                if self.__normalization is not None:
                    mean_out.append(numpy.sum(self[par][select_age] * coefficients[select_age]) / numpy.sum(coefficients[select_age]))
                else:
                    mean_out.append(numpy.sum(self[par][select_age] * self['luminosity'][select_age] * coefficients[select_age]) /
                        numpy.sum(self['luminosity'][select_age] * coefficients[select_age]))
        else:
            mean_out = [0.0, numpy.nan, numpy.nan, numpy.nan, numpy.nan]
        return numpy.array(mean_out)

    def massWeightedPars(self, coefficients, min_age=None, max_age=None):
        """From the weights of the template spectra in the library, this
        function computes the parameters by taking mass-weighted averages.

        Parameters
        ----------
        coefficients : numpy.ndarray
            The weights that each template spectrum has.
        min_age : float, optional
            Ignore the spectra with ages below a certain threshold.
        max_age : float, optional
            Ignore the spectra with ages above a certain threshold.

        Returns
        -------
        mean : numpy.ndarray
            Mass-weighted averages stored in a 1D numpy array. The first
            value is age, the second value the mass-to-light ratio, the third
            value [Fe/H], and the fourth value [alpha/Fe]
        """
        parameters = ['age', 'mass-to-light', '[Fe/H]', '[A/Fe]']

        select_age = self['age'] > 0
        if min_age is not None:
            select_age[self['age'] < min_age] = False
        if max_age is not None:
            select_age[self['age'] > max_age] = False

        if numpy.sum(select_age)>0:
            if self.__normalization is not None:
                mean_out = [numpy.sum(coefficients[select_age])]
            else:
                mean_out = [numpy.sum(self['mass'][select_age] * coefficients[select_age]) /
                numpy.sum(self['mass'] * coefficients)]
            for par in parameters:
                if self.__normalization is not None:
                    mean_out.append(numpy.sum(self[par][select_age] * self['mass'][select_age] / self['luminosity'][select_age] *
                        coefficients[select_age]) / numpy.sum(self['mass'][select_age] / self['luminosity'][select_age]
                        * coefficients[select_age]))
                else:
                    mean_out.append(numpy.sum(self[par][select_age] * self['mass'][select_age] * coefficients[select_age]) /
                        numpy.sum(self['mass'][select_age] * coefficients[select_age]))
        else:
            mean_out = [0.0, numpy.nan, numpy.nan, numpy.nan, numpy.nan]
        return numpy.array(mean_out)

    def resampleBase(self, new_wave):
        """Returns an object with the template spectra resampled on a new
        wavelength grid.

        Parameters
        ----------
        new_wave : numpy.ndarray
            The new wavelength array on which the spectra will be resampled.

        Returns
        -------
        new_SSP : SSPlibrary
            A new instance of the SSPlibrary with a new wavelength grid and
            resampled spectra.
        """
        data = numpy.zeros((len(new_wave), self.__nbasis), dtype=numpy.float32)
        if self.__normalization is not None:
            normalization = numpy.zeros((len(new_wave), self.__nbasis), dtype=numpy.float32)
        else:
            normalization = None
        for i in range(self.__nbasis):
            intp = interpolate.UnivariateSpline(self.__wave, self.__data[:, i], s=0)
            data[:, i] = intp(new_wave)
            if self.__normalization is not None:
                intp = interpolate.UnivariateSpline(self.__wave, self.__normalization[:, i], s=0)
                normalization[:, i] = intp(new_wave)
        new_SSP = SSPlibrary(data=data, wave=new_wave, spectralFWHM=self.__spectralFWHM, infoSSP=self, coefficients=
        self.__coefficients, normalization=normalization)
        return new_SSP

    def resampleWaveStepLinear(self, step, redshift):
        """Returns a new library with the template spectra resampled to a new
        linear wavelength grid.

        Parameters
        ----------
        step : float
            the new linear step size in the new wavelength grid
        redshift : float
            the redshift in which the object resides and to which the
            wavelengths will be shifted to.

        Returns
        -------
        new_SSP : SSPlibrary
            A new instance of the SSPlibrary with a new wavelength grid and
            resampled spectra.
        """
        new_wave = numpy.arange(self.__wave[0], self.__wave[-1], step / (1 + float(redshift)))
        data = numpy.zeros((len(new_wave), self.__nbasis), dtype=numpy.float32)
        if self.__normalization is not None:
            normalization = numpy.zeros((len(new_wave), self.__nbasis), dtype=numpy.float32)
        else:
            normalization = None
        for i in range(self.__nbasis):
            intp = interpolate.UnivariateSpline(self.__wave, self.__data[:, i], s=0)
            data[:, i] = intp(new_wave)
            if self.__normalization is not None:
                intp = interpolate.UnivariateSpline(self.__wave, self.__normalization[:, i], s=0)
                normalization[:, i] = intp(new_wave)
        new_SSP = SSPlibrary(data=data, wave=new_wave, spectralFWHM=self.__spectralFWHM, infoSSP=self,
        coefficients=self.__coefficients, normalization=normalization)
        return new_SSP

    def rebinLogarithmic(self):
        """Rebin the template spectra from a linear wavelength grid to a
        logarithmic wavelength grid.

        Returns
        -------
        new_SSP : SSPlibrary
            A new instance of the SSPlibrary with a logarithmic wavelength grid
            and resampled spectra.
        """
        wave_log = 10 ** numpy.arange(numpy.log10(self.__wave[0]), numpy.log10(self.__wave[-1]), (numpy.log10(self.__wave[-1])
        - numpy.log10(self.__wave[0])) / len(self.__wave))
        new_SSP = self.resampleBase(wave_log)
        new_SSP.__vel_sampling = (self.__wave[1] - self.__wave[0]) / self.__wave[0] * 300000.0
        return new_SSP

    def matchInstFWHM(self, instFWHM, obs_velocity):
        """Obtain a new SSP library where the spectra are broadened to match the
        instrumental resolution and shifted to the velocity of the object.

        Parameters
        ----------
        instFWHM : float
            the instrumental resolution in FWHM in the same units as the
            wavelength grid.
        obs_velocity : float
            the velocity of the object in km/s to which the template spectra
            will be moved.

        Returns
        -------
        new_SSP : SSPlibrary
            A new instance of the SSPlibrary with the spectra broadened and
            shifted.
        """
        redshift = (1 + obs_velocity / 300000.0)
        wave = self.__wave * redshift
        if instFWHM > self.__spectralFWHM * redshift:
            smooth_FWHM = numpy.sqrt(instFWHM ** 2 - (self.__spectralFWHM * redshift) ** 2)
            disp_pix = (smooth_FWHM / 2.354) / (wave[1] - wave[0])
            data = ndimage.filters.gaussian_filter1d(self.__data, disp_pix, axis=0, mode='constant')
            new_SSP = SSPlibrary(data=data, wave=self.__wave, spectralFWHM=self.__spectralFWHM, infoSSP=self,
                 coefficients=self.__coefficients)
        else:
            raise ValueError("The instrinic spectral resolution of the template spectra %E is higher than the targetFWHM spectral resolution %E in the given observed frame z=%E" %(self.__spectralFWHM, instFWHM, redshift))

        return new_SSP

    def applyGaussianLOSVD(self, vel, disp_vel):
        """Obtain a new SSP library where the spectra are broadened with a
        Gaussian profile and shifted to the velocity of the object.

        Parameters
        ----------
        vel : float
            The velocity of the object in km/s to which the template spectra
            will be moved.
        disp_vel : float
            The velocity dispersion of the object in km/s to which the template
            spectra will be broadened.

        Returns
        -------
        new_SSP : SSPlibrary
            A new instance of the SSPlibrary with the spectra broadened and
            shifted.
        """
        disp_pix = disp_vel / self.__vel_sampling
        data = ndimage.filters.gaussian_filter1d(self.__data, numpy.fabs(disp_pix), axis=0, mode='constant')
        if self.__normalization is not None:
            normalization = ndimage.filters.gaussian_filter1d(self.__normalization, numpy.fabs(disp_pix), axis=0, mode='constant')
        else:
            normalization = None
        wave = self.__wave * (1 + vel / 300000.0)
        new_SSP = SSPlibrary(data=data, wave=wave, spectralFWHM=self.__spectralFWHM, infoSSP=self,
        coefficients=self.__coefficients, normalization=normalization)
        new_SSP.__vel_sampling = self.__vel_sampling
        return new_SSP

    def applyExtinction(self, A_V, law='Cardelli', R_V=3.1):
        micron = self.__wave / 10000.0
        wave_number = 1.0 / micron
        y = wave_number - 1.82
        ax = 1 + (0.17699 * y) - (0.50447 * y ** 2) - (0.02427 * y ** 3) + (0.72085 * y ** 4) + (0.01979 * y ** 5)
        - (0.77530 * y ** 6) + (0.32999 * y ** 7)
        bx = (1.41338 * y) + (2.28305 * y ** 2) + (1.07233 * y ** 3) - (5.38434 * y ** 4) - (0.62251 * y ** 5)
        + (5.30260 * y ** 6) - (2.09002 * y ** 7)
        Arat = ax + (bx / R_V)
        Alambda = Arat * A_V
        data = self.__data * 10 ** (Alambda[:, numpy.newaxis] / -2.5)
        new_SSP = SSPlibrary(data=data, wave=self.__wave, spectralFWHM=self.__spectralFWHM, infoSSP=self,
        coefficients=self.__coefficients)
        try:
            new_SSP.__vel_sampling = self.__vel_sampling
        except:
            pass
        return new_SSP

    def modelSpec(self, vel, vel_disp, A_V, wave, coeff=None):
        if A_V <= 0:
            A_V = 0
        if vel_disp <= 0:
            vel_disp = 0
        tempLib = self.applyExtinction(A_V)
        tempLib = tempLib.applyGaussianLOSVD(vel, vel_disp)
        tempLib = tempLib.resampleBase(wave)
        if coeff is None:
            coeff = self.__coefficients
        return tempLib.compositeSpectrum(coeff)

    def fitMCMC(self, vel_min, vel_max, vel_disp_min, vel_disp_max, A_V_min, A_V_max, inputSpec):
        valid_pix = numpy.logical_not(inputSpec._mask)
        wave = inputSpec._wave[valid_pix]

        vel = pymc.Uniform('vel', lower=vel_min, upper=vel_max)
        disp = pymc.Uniform('disp', lower=vel_disp_min, upper=vel_disp_max)
        a = pymc.Uniform('a', lower=A_V_min, upper=A_V_max)

        @pymc.deterministic(plot=False)
        def m(vel=vel, disp=disp, a=a):
            return self.modelSpec(vel, disp, a, wave)[1]
        d = pymc.Normal('d', mu=m, tau=inputSpec._error[valid_pix], value=inputSpec._data[valid_pix], observed=True)

        M = pymc.MCMC([vel, disp, a, m, d])
        M.sample(burn=1000, iter=1000, thin=10)

        return M

    def residuumFullFit(self, parameters, inputSpec):
        vel = parameters[0]
        vel_disp = parameters[1]
        extinction = parameters[2]
        bestfit = inputSpec.fitComposite(self, vel, vel_disp, extinction)
        return bestfit[3]

    def findBestFit(self, parameters, inputSpec):
        output = optimize.fmin(self.residuumFullFit, parameters, args=(inputSpec, ))
        tempLib = self.applyExtinction(output[2])
        tempLib = tempLib.applyGaussianLOSVD(output[0], output[1])
        best_fit = tempLib.findLinearCombination(inputSpec)
        self.setCoefficients(best_fit[2])
        return output, best_fit



