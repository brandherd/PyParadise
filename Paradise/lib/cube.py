from Paradise.lib.header import *
from Paradise.lib.spectrum1d import Spectrum1D
import pyfits
import numpy
from scipy import ndimage
from copy import deepcopy
from multiprocessing import cpu_count
from multiprocessing import Pool


class Cube(Header):

    def __getitem__(self, slice):

        if not isinstance(slice, int):
            raise TypeError('Slice index need to be an integer')

        if slice >= self._res_elements or slice < self._res_elements * -1:
            raise IndexError('The Cube contains only %i resolution elments for which the index %i is invalid' %
            (self._res_elements, slice))

        if self._data is not None:
            data = self._data[slice, :, :]
        else:
            data = None

        if self._error is not None:
            error = self._error[slice, :, :]
        else:
            error = None
        if self._mask is not None:
            mask = self._mask[slice, :, :]
        else:
            mask = None
        return Image(data=data, error=error, mask=mask)

    def __init__(self, data=None, wave=None, error=None, mask=None, error_weight=None, normalization=None, header=None):
        Header.__init__(self, header=header)
        if data is None:
            self._data = None
        else:
            self._data = data
            self._res_elements = data.shape[0]
            self._dim_y = data.shape[1]
            self._dim_x = data.shape[2]
        if wave is None:
            self._wave = None
        else:
            self._wave = numpy.array(wave)
        if error is None:
            self._error = None
        else:
            self._error = numpy.array(error)

        if error_weight is None:
            self._error_weight = None
        else:
            self._error_weight = numpy.array(error_weight)

        if mask is None:
            self._mask = None
        else:
            self._mask = numpy.array(mask)

        self._normalization = normalization

    def getWave(self):
        return self._wave

    def getWaveStep(self):
        return self._wave[1] - self._wave[0]

    def getShape(self):
        return self._data.shape

    def getSpec(self, x, y):
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

    def subCubeWave(self, wave_start=None, wave_end=None):
        select = numpy.ones(len(self.getWave()), dtype='bool')
        if wave_start is not None:
            select[self.getWave() < wave_start] = False
        if wave_end is not None:
            select[self.getWave() > wave_end] = False
        if self._error is not None:
            error = self._error[select, :, :]
        else:
            error = None
        if self._mask is not None:
            mask = self._mask[select, :, :]
        else:
            mask = None
        if self._normalization is not None:
            normalization = self._normalization[select, :, :]
        else:
            normalization = None
        cube_out = Cube(wave=self.getWave()[select], data=self._data[select, :, :], error=error, mask=mask,
        normalization=normalization)
        return cube_out

    def correctError(self, replace_error=1e10):
        if self._error is not None:
            select = (self._error <= 0)
            if self._mask is not None:
                self._mask[select] = True
            self._error[select] = replace_error

    def normalizeSpec(self, pixel_width, mask_norm):
        mean = numpy.zeros(self._data.shape, dtype=numpy.float32)
        data_temp = numpy.zeros(self._data.shape, dtype=numpy.float32)
        data_temp[:] = self._data
        if self._mask is not None:
            mask = numpy.logical_or(self._mask, mask_norm[:, numpy.newaxis, numpy.newaxis])
            select_bad = mask
        else:
            mask = mask_norm[:, numpy.newaxis, numpy.newaxis]
            select_bad = mask
        data_temp[select_bad] = 0.0
        uniform = ndimage.filters.convolve(data_temp, numpy.ones((pixel_width, 1, 1), dtype=numpy.int16), mode='nearest')
        summed = ndimage.filters.generic_filter(numpy.logical_not(mask).astype('int16'), numpy.sum, (pixel_width, 1, 1),
        mode='nearest')
        select = summed > 0
        mean[select] = uniform[select] / summed[select].astype('float32')
        mean[numpy.logical_not(select)] = 1
        select_zero = mean == 0
        mean[select_zero] = 1
        new_data = self._data / mean
        new_error = self._error / mean
        cube_out = Cube(wave=self._wave, data=new_data, error=new_error, mask=self._mask, normalization=mean)
        return cube_out

    def unnormalizedSpec(self):
        data = self.__data * self.__normalization
        if self.__error is not None:
            error = self.__error * self.__normalization
        else:
            error = None
        cube_out = Cube(wave=self._wave, data=data, error=error, mask=self._mask, normalization=None)
        return cube_out

    def loadFitsData(self, file, extension_data=None, extension_mask=None, extension_error=None, extension_errorweight=None,
    extension_hdr=0):
        """
            Load data from a FITS image into a Cube object

            Parameters
            --------------
            filename : string
                Name or Path of the FITS image from which the data shall be loaded

            extension_data : int, optional with default: None
                Number of the FITS extension containing the data

            extension_mask : int, optional with default: None
                Number of the FITS extension containing the masked pixels

            extension_error : int, optional with default: None
                Number of the FITS extension containing the errors for the values
        """
        hdu = pyfits.open(file)
        if extension_data is None and extension_mask is None and extension_error is None and extension_errorweight is None:
                self._data = hdu[0].data
                self._res_elements = self._data.shape[0]
                self._dim_y = self._data.shape[1]
                self._dim_x = self._data.shape[2]
                self.setHeader(header=hdu[extension_hdr].header, origin=file)
                if len(hdu) > 1:
                    for i in range(1, len(hdu)):
                        if hdu[i].header['EXTNAME'].split()[0] == 'ERROR':
                            self._error = hdu[i].data
                        elif hdu[i].header['EXTNAME'].split()[0] == 'BADPIX':
                            self._mask = hdu[i].data.astype('bool')
                        elif hdu[i].header['EXTNAME'].split()[0] == 'ERRWEIGHT':
                            self._error_weight = hdu[i].data
                self._wave = numpy.arange(self._res_elements) * self.getHdrValue('CDELT3') + self.getHdrValue('CRVAL3')
        else:
            self.setHeader(header=hdu[extension_hdr].header, origin=file)
            if extension_data is not None:
                self._data = hdu[extension_data].data
                self._res_elements = self._data.shape[0]
                self._dim_y = self._data.shape[1]
                self._dim_x = self._data.shape[2]

            if extension_mask is not None:
                self._mask = hdu[extension_mask].data
                self._res_elements = self._mask.shape[0]
                self._dim_y = self._mask.shape[1]
                self._dim_x = self._mask.shape[2]
            if extension_error is not None:
                self._error = hdu[extension_error].data
                self._res_elements = self._error.shape[0]
                self._dim_y = self._error.shape[1]
                self._dim_x = self._error.shape[2]
            if extension_error is not None:
                self._error_weight = hdu[extension_errorweight].data
                self._res_elements = self._error_weight.shape[0]
                self._dim_y = self._error_weight.shape[1]
                self._dim_x = self._error_weight.shape[2]
            self._wave = numpy.arange(self._res_elements) * self.getHdrValue('CDELT3') + self.getHdrValue('CRVAL3')
        hdu.close()

        if extension_hdr is not None:
            self.setHeader(hdu[extension_hdr].header, origin=file)

    def writeFitsData(self, filename, extension_data=None, extension_mask=None, extension_error=None,
    extension_errorweight=None):
        """
            Save information from a Cube object into a FITS file.
            A single or multiple extension file are possible to create.

            Parameters
            --------------
            filename : string
                Name or Path of the FITS image from which the data shall be loaded

            extension_data : int (0, 1, or 2), optional with default: None
                Number of the FITS extension containing the data

            extension_mask : int (0, 1, or 2), optional with default: None
                Number of the FITS extension containing the masked pixels

            extension_error : int (0, 1, or 2), optional with default: None
                Number of the FITS extension containing the errors for the values
        """
        hdus = [None, None, None, None]  # create empty list for hdu storage

        # create primary hdus and image hdus
        # data hdu
        if extension_data is None and extension_error is None and extension_mask is None and extension_errorweight is None:
            hdus[0] = pyfits.PrimaryHDU(self._data)
            if self._error is not None:
                hdus[1] = pyfits.ImageHDU(self._error, name='ERROR')
            if self._error_weight is not None:
                hdus[2] = pyfits.ImageHDU(self._error_weight, name='ERRWEIGHT')
            if self._mask is not None:
                hdus[3] = pyfits.ImageHDU(self._mask.astype('uint8'), name='BADPIX')
        else:
            if extension_data == 0:
                hdus[0] = pyfits.PrimaryHDU(self._data)
            elif extension_data > 0 and extension_data is not None:
                hdus[extension_data] = pyfits.ImageHDU(self._data, name='DATA')

            # mask hdu
            if extension_mask == 0:
                hdu = pyfits.PrimaryHDU(self._mask.astype('uint8'))
            elif extension_mask > 0 and extension_mask is not None:
                hdus[extension_mask] = pyfits.ImageHDU(self._mask.astype('uint8'), name='BADPIX')

            # error hdu
            if extension_error == 0:
                hdu = pyfits.PrimaryHDU(self._error)
            elif extension_error > 0 and extension_error is not None:
                hdus[extension_error] = pyfits.ImageHDU(self._error, name='ERROR')

            if extension_errorweight == 0:
                hdu = pyfits.PrimaryHDU(self._error_weight)
            elif extension_errorweight > 0 and extension_errorweight is not None:
                hdus[extension_errorweight] = pyfits.ImageHDU(self._error_weight, name='ERRWEIGHT')

        # remove not used hdus
        for i in range(len(hdus)):
            try:
                hdus.remove(None)
            except:
                break

        if len(hdus) > 0:
            hdu = pyfits.HDUList(hdus)  # create an HDUList object
            if self._header is not None:
                if self._wave is not None:
                    self.setHdrValue('CRVAL3', self._wave[0])
                    self.setHdrValue('CDELT3', (self._wave[1] - self._wave[0]))
                hdu[0].header = self.getHeader()  # add the primary header to the HDU
                hdu[0].update_header()
            else:
                if self._wave is not None:
                    hdu[0].header.update('CRVAL3', self._wave[0])
                    hdu[0].header.update('CDELT3', self._wave[1] - self._wave[0])
        hdu.writeto(filename, clobber=True)  # write FITS file to disc

    def fit_Kin_Lib_simple(self, SSPLib, nlib_guess, vel_min, vel_max, disp_min, disp_max, min_x, max_x, min_y, max_y, mask_fit,
        iterations=2, burn=1500, samples=4000, thin=2, verbose=False, parallel='auto'):

        cube_model = numpy.zeros(self.getShape(), dtype=numpy.float32)
        vel_fit = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.float32)
        vel_fit_err = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.float32)
        disp_fit = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.float32)
        disp_fit_err = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.float32)
        chi2 = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.float32)
        x_pix = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.int16)
        y_pix = numpy.zeros(self._dim_y * self._dim_x, dtype=numpy.int16)
        fitted = numpy.zeros(self._dim_y * self._dim_x, dtype="bool")
        coeff = numpy.zeros((self._dim_y * self._dim_x, SSPLib.getBaseNumber()), dtype=numpy.float32)
        if parallel == 'auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus, maxtasksperchild=200)
            result_fit = []
            m = 0
            for x in range(self._dim_x):
                for y in range(self._dim_y):
                    spec = self.getSpec(x, y)
                    if spec.hasData() and x >= (min_x - 1) and x <= (max_x - 1) and y >= (min_y - 1) and y <= (max_y - 1):
                        result_fit.append(pool.apply_async(spec.fit_Kin_Lib_simple, args=(SSPLib, nlib_guess, vel_min, vel_max,
                        disp_min, disp_max, mask_fit, iterations, burn, samples, thin)))
                    else:
                        result_fit.append(None)
                    x_pix[m] = x
                    y_pix[m] = y
                    m += 1
                    #if m == 1550:
                        #break
                #if m == 1550:
                    #break

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
                        cube_model[:, y_pix[m], x_pix[m]] = result[4].unnormalizedSpec().getData()
                    except ValueError:
                        print "Fitting failed because of bad spectrum."
                #if m == 1550:
                    #break
        else:
            m = 0
            for x in range(self._dim_x):
                for y in range(self._dim_y):
                    spec = self.getSpec(x, y)

                    if spec.hasData() and x >= (min_x - 1) and x <= (max_x - 1) and y >= (min_y - 1) and y <= (max_y - 1):
                        if verbose:
                            print "Fitting Spectrum (%d, %d) of cube" % (x + 1, y + 1)
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
                            x_pix[m] = x
                            y_pix[m] = y
                            cube_model[:, y, x] = result[4].unnormalizedSpec().getData()
                            if verbose:
                                print "vel_fit: %.3f  disp_fit: %.3f chi2: %.2f" % (vel_fit[m], disp_fit[m], chi2[m])
                        except (ValueError, IndexError):
                             print "Fitting failed because of bad spectrum."

                    m += 1
                    #if m == 1550:
                        #break
                #if m == 1550:
                        #break
        return vel_fit, vel_fit_err, disp_fit, disp_fit_err, fitted, coeff, chi2, x_pix, y_pix, cube_model

    def fitELines(self, par, select_wave, min_x, max_x, min_y, max_y, method='leastsq', guess_window=0.0, spectral_res=0.0, ftol=1e-4,
    xtol=1e-4, verbose=1, parallel='auto'):

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

        if parallel == 'auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus, maxtasksperchild=200)
            result_fit = []
            m = 0
            for x in range(self._dim_x):
                for y in range(self._dim_y):
                    spec = self.getSpec(x, y)
                    if spec.hasData() and x >= (min_x - 1) and x <= (max_x - 1) and y >= (min_y - 1) and y <= (max_y - 1):
                        result_fit.append(pool.apply_async(spec.fitELines,args=(par, select_wave, method, guess_window,
                        spectral_res, ftol, xtol, 1)))
                    else:
                        result_fit.append(None)
                    x_pix[m] = x
                    y_pix[m] = y
                    m += 1
            pool.close()
            pool.join()
            for  m in range(len(result_fit)):
                if result_fit[m] is not None:
                    result = result_fit[m].get()
                    cube_model[:, y_pix[m], x_pix[m]] = result[1]
                    fitted[m] = True

                    for n in par._names:
                        if par._profile_type[n] == 'Gauss':
                            maps[n]['flux'][m] = result[0][n][0]
                            maps[n]['vel'][m] = result[0][n][1]
                            maps[n]['fwhm'][m] = result[0][n][2]
        else:
            m = 0
            for x in range(self._dim_x):
                for y in range(self._dim_y):
                    spec = self.getSpec(x, y)

                    if spec.hasData() and x >= (min_x - 1) and x <= (max_x - 1) and y >= (min_y - 1) and y <= (max_y - 1):
                        if verbose:
                            print "Fitting Spectrum (%d, %d) of cube" % (x + 1, y + 1)
                        result = spec.fitELines(par, select_wave, method=method, guess_window=guess_window,
                            spectral_res=spectral_res, ftol=ftol, xtol=xtol, parallel=1)
                        x_pix[m] = x
                        y_pix[m] = y
                        fitted[m] = True
                        cube_model[:,y,x] = result[1]
                        for n in par._names:
                            if par._profile_type[n] == 'Gauss':
                                maps[n]['flux'][m] = result[0][n][0]
                                maps[n]['vel'][m] = result[0][n][1]
                                maps[n]['fwhm'][m] = result[0][n][2]
                    m += 1
        return maps, fitted, x_pix, y_pix, cube_model

    def fit_Lib_Boots(self, lib_SSP, x_cor,y_cor, vel, disp, vel_err=None, disp_err=None, par_eline=None, select_wave_eline=None,
        method_eline='leastsq', guess_window=10.0, spectral_res=0.0, ftol=1e-4, xtol=1e-4, bootstraps=100, modkeep=80,
        parallel=1, verbose=False):

        mass_weighted_pars_err = numpy.zeros((len(x_cor),5), dtype=numpy.float32)
        lum_weighted_pars_err = numpy.zeros((len(x_cor),5), dtype=numpy.float32)

        if par_eline is not None:
            maps = {}
            for n in par_eline._names:
                model = {}
                if par_eline._profile_type[n] == 'Gauss':
                    model['flux_err'] = numpy.zeros(len(x_cor), dtype=numpy.float32)
                    model['vel_err'] = numpy.zeros(len(x_cor), dtype=numpy.float32)
                    model['fwhm_err'] = numpy.zeros(len(x_cor), dtype=numpy.float32)
                maps[n] = model

        if parallel == 'auto':
            cpus = cpu_count()
        else:
            cpus = int(parallel)
        if cpus > 1:
            pool = Pool(cpus, maxtasksperchild=200)
            result_fit = []
            for m in range(len(x_cor)):
                spec = self.getSpec(x_cor[m], y_cor[m])
                result_fit.append(pool.apply_async(spec.fit_Lib_Boots, args=(lib_SSP, vel[m], disp[m], None, None, par_eline,
                         select_wave_eline, method_eline, guess_window, spectral_res, ftol, xtol, bootstraps, modkeep, 1)))
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
            for m in range(len(x_cor)):
                spec = self.getSpec(x_cor[m], y_cor[m])
                if verbose:
                    print "Fitting Spectrum (%d, %d) of cube" % (x_cor[m] + 1, y_cor[m] + 1)
                result = spec.fit_Lib_Boots(lib_SSP, vel[m], disp[m], None, None, par_eline,
                     select_wave_eline, method_eline, guess_window, spectral_res, ftol, xtol, bootstraps, modkeep, 1)

                mass_weighted_pars_err[m,:] = result[0]
                lum_weighted_pars_err[m,:] = result[1]
                if par_eline is not None:
                    for n in par_eline._names:
                        if par_eline._profile_type[n] == 'Gauss':
                            maps[n]['flux_err'][m] = result[2][n]['flux']
                            maps[n]['vel_err'][m] = result[2][n]['vel']
                            maps[n]['fwhm_err'][m] = result[2][n]['fwhm']
        if par_eline is None:
            maps = None
        return mass_weighted_pars_err, lum_weighted_pars_err, maps

def loadCube(infile, extension_data=None, extension_mask=None, extension_error=None):

    cube = Cube()
    cube.loadFitsData(infile, extension_data=None, extension_mask=None, extension_error=None)

    return cube



