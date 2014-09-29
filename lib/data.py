from Paradise.lib.header import *
import pyfits
import numpy
import pylab
from scipy import ndimage
from copy import deepcopy
from multiprocessing import cpu_count
from multiprocessing import Pool


class Data(Header):

    def __init__(self, data=None, wave=None, error=None, mask=None, error_weight=None, normalization=None, inst_fwhm=None,
    header=None):
        Header.__init__(self, header=header)
        self._data = data
        if data is not None:
            self._dim = data.shape
            if len(data.shape) == 3:
                self._datatype = "CUBE"
                self._res_elements = data.shape[0]
                self._dim_y = data.shape[1]
                self._dim_x = data.shape[2]
            elif len(data.shape) == 2:
                self._datatype = "RSS"
                self._res_elements = data.shape[1]
                self._fibers = data.shape[0]
            elif len(data.shape) == 1:
                self._datatype = "Spectrum1D"
                self._pixels = numpy.arange(self._dim[0])

        self._wave = wave

        self._error = error

        self._error_weight = error_weight

        self._mask = mask

        self._normalization = normalization

        self._inst_fwhm = inst_fwhm

    def getWave(self):
        """Obtain the wavelength grid as a 1D numpy array."""
        return self._wave

    def getWaveStep(self):
        return self._wave[1] - self._wave[0]

    def getShape(self):
        return self._dim

    def getData(self):
        """Obtain the data as a numpy array."""
        return self._data

    def setData(self, data):
        """Set the data by providing a numpy array. The array should
        be matching the shape of the wavelength grid along the corresponding
        axis for Spectrum1D, RSS and Cube data."""
        self._data = data

    def getError(self):
        """Obtain the error associated to the data as a numpy array."""
        return self._error

    def setError(self, error):
        """Set the error by providing a numpy array. The array should be of the
        same shape as the data."""
        self._error = error_weight

    def getMask(self):
        """Obtain the mask as a numpy array."""
        return self._mask

    def setMask(self, mask):
        """Set the mask by providing a numpy array. The mask should be of
        the same shape as the data."""
        self._mask = mask

    def getFWHM(self):
        """Obtain the FWHM of the data, provided in the same units as the
        wavelength grid."""
        return self._inst_fwhm

    def setFWHM(self, FWHM):
        """Set the value of the FWHM of the data."""
        self._inst_fwhm = FWHM

    def getNormalization(self):
        """Obtain the normalization of the spectrum as a numpy array."""
        return self._normalization

    def setNormalization(self, normalization):
        """Set the normalization of the spectrum by providing a numpy array.
           The array should be of the same shape as the data."""
        self._normalization = normalization

    def correctError(self, replace_error=1e10):
        if self._error is not None:
            select = (self._error <= 0)
            if self._mask is not None:
                self._mask[select] = True
            self._error[select] = replace_error

    def subWaveMask(self, select_wave):
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

        [new_error, new_mask, new_fwhm, new_normalization] = [None, None, None, None]

        new_wave = self._wave[select_wave]
        if self._datatype == "Spectrum1D":
            new_data = self._data[select_wave]
            if self._inst_fwhm is not None:
                new_fwhm = self._inst_fwhm[select_wave]
            if self._error is not None:
                new_error = self._error[select_wave]
            if self._mask is not None:
                new_mask = self._mask[select_wave]
            if self.getNormalization() is not None:
                new_normalization = self._normalization[select_wave]
        elif self._datatype == "RSS":
            new_data = self._data[:, select_wave]
            if self._inst_fwhm is not None:
                new_fwhm = self._inst_fwhm[:, select_wave]
            if self._error is not None:
                new_error = self._error[:, select_wave]
            if self._mask is not None:
                new_mask = self._mask[:, select_wave]
            if self.getNormalization() is not None:
                new_normalization = self._normalization[:, select_wave]
        elif self._datatype == "CUBE":
            new_data = self._data[select_wave, :, :]
            if self._inst_fwhm is not None:
                new_fwhm = self._inst_fwhm[select_wave, :, :]
            if self._error is not None:
                new_error = self._error[select_wave, :, :]
            if self._mask is not None:
                new_mask = self._mask[select_wave, :, :]
            if self.getNormalization() is not None:
                new_normalization = self._normalization[select_wave, :, :]

        data_out = Data(wave=new_wave, data=new_data, error=new_error, mask=new_mask,
            normalization=new_normalization, inst_fwhm=new_fwhm)
        data_out.__class__ = self.__class__
        return data_out

    def subWaveLimits(self, wave_start=None, wave_end=None):
        select = numpy.ones(len(self.getWave()), dtype='bool')
        if wave_start is not None:
            select[self.getWave() < wave_start] = False
        if wave_end is not None:
            select[self.getWave() > wave_end] = False
        data_out = self.subWaveMask(select)
        return data_out

    def normalizeSpec(self, pixel_width, mask_norm=None):
        mean = numpy.zeros(self._data.shape, dtype=numpy.float32)
        data_temp = numpy.zeros(self._data.shape, dtype=numpy.float32)
        data_temp[:] = self._data
        if self._datatype == 'RSS':
            mask_norm_expand = mask_norm[numpy.newaxis, :]
            filter_window = (1, pixel_width)
        elif self._datatype == 'CUBE':
            mask_norm_expand = mask_norm[:, numpy.newaxis, numpy.newaxis]
            filter_window = (pixel_width, 1, 1)
        elif self._datatype == 'Spectrum1D':
            mask_norm_expand = mask_norm
            filter_window = (pixel_width)

        if self._mask is not None and mask_norm is not None:
            mask = numpy.logical_or(self._mask, mask_norm_expand)
        elif mask_norm is not None:
            mask = mask_norm_expand
        else:
            mask = mask_norm_expand

        select_bad = mask
        data_temp[select_bad] = 0.0

        uniform = ndimage.filters.convolve(data_temp, numpy.ones(filter_window, dtype=numpy.int16), mode='nearest')
        summed = ndimage.filters.generic_filter(numpy.logical_not(mask).astype('int16'), numpy.sum, filter_window,
        mode='nearest')
        select = summed > 0
        mean[select] = uniform[select] / summed[select].astype('float32')
        mean[numpy.logical_not(select)] = 1
        select_zero = mean == 0
        mean[select_zero] = 1
        new_data = self._data / mean
        new_error = self._error / numpy.fabs(mean)
  #      pylab.plot(self._wave,self._data[:,15,15],'-k')
  #      pylab.plot(self._wave,self._error[:,15,15],'-r')
  #      pylab.plot(self._wave,self._error[:,15,15]/numpy.fabs(mean[:,15,15]),'-b')
 #       pylab.plot(self._wave,new_data[:,15,15],'-b')
 #       pylab.plot(self._wave,new_error[:,15,15],'-g')
  #      pylab.show()
        data_out = Data(wave=self._wave, data=new_data, error=new_error, mask=self._mask, normalization=mean,
        inst_fwhm=self._inst_fwhm)
        data_out.__class__ = self.__class__
        return data_out

    def unnormalizedSpec(self):
        data = self._data * self._normalization
        if self._error is not None:
            error = self._error * numpy.fabs(self._normalization)
        else:
            error = None
        data_out = Data(wave=self._wave, data=data, error=error, mask=self._mask, normalization=None, inst_fwhm=self._inst_fwhm)
        data_out.__class__ = self.__class__
        return data_out

    def applyNormalization(self, normalization):
        """Apply the normalization to the data and the errors."""
        if self._normalization is None:
            self._data = self._data / normalization
            if self._error is not None:
                self._error = self._error / numpy.fabs(normalization)

    def setVelSampling(self, vel_sampling):
        """Change the velocity sampling of the spectra (float, km/s)."""
        self._vel_sampling = vel_sampling

    def getVelSampling(self):
        """Obtain the velocity sampling of the spectra in km/s."""
        return self._vel_sampling

    def loadFitsData(self, file, extension_data=None, extension_mask=None, extension_error=None, extension_errorweight=None,
    extension_hdr=0):
        """
            Load data from a FITS image into a Data object

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
        self._dim = None
        if extension_data is None and extension_mask is None and extension_error is None and extension_errorweight is None:
                self._data = hdu[0].data
                self._dim = self._data.shape
                self.setHeader(header=hdu[extension_hdr].header, origin=file)
                if len(hdu) > 1:
                    for i in range(1, len(hdu)):
                        if hdu[i].header['EXTNAME'].split()[0] == 'ERROR':
                            self._error = hdu[i].data
                        elif hdu[i].header['EXTNAME'].split()[0] == 'BADPIX':
                            self._mask = hdu[i].data.astype('bool')
                        elif hdu[i].header['EXTNAME'].split()[0] == 'ERRWEIGHT':
                            self._error_weight = hdu[i].data

        else:
            self.setHeader(header=hdu[extension_hdr].header, origin=file)
            if extension_data is not None:
                self._data = hdu[extension_data].data
                self._dim = self._data.shape
            if extension_mask is not None:
                self._mask = hdu[extension_mask].data
                self._dim = self._mask.shape
            if extension_error is not None:
                self._error = hdu[extension_error].data
                self._dim = self._error.shape
            if extension_error is not None:
                self._error_weight = hdu[extension_errorweight].data
                self._dim = self._error_weight.shape

            self._wave = numpy.arange(self._res_elements) * self.getHdrValue('CDELT3') + self.getHdrValue('CRVAL3')
        hdu.close()
        if self._dim is not None:
            if len(self._dim) == 3:
                self._datatype = "CUBE"
                self._res_elements = self._dim[0]
                self._dim_y = self._dim[1]
                self._dim_x = self._dim[2]
                self._wave = numpy.arange(self._res_elements) * self.getHdrValue('CDELT3') + self.getHdrValue('CRVAL3')
            elif len(self._dim) == 2:
                self._datatype = "RSS"
                self._res_elements = self._dim[1]
                self._fibers = self._dim[0]
                self._wave = numpy.arange(self._res_elements) * self.getHdrValue('CDELT1') + self.getHdrValue('CRVAL1')
            elif len(self._dim) == 1:
                self._datatype = "Spectrum1D"
                self._pixels = numpy.arange(self._dim[0])
                self._res_elements = self._dim[0]
                self._wave = numpy.arange(self._res_elements) * self.getHdrValue('CDELT1') + self.getHdrValue('CRVAL1')

        if extension_hdr is not None:
            self.setHeader(hdu[extension_hdr].header, origin=file)

    def writeFitsData(self, filename, extension_data=None, extension_mask=None, extension_error=None,
    extension_errorweight=None,extension_normalization=None):
        """
            Save information of the Data object into a FITS file.
            A single or multiple extension file are possible to create.

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
        hdus = [None, None, None, None,None]  # create empty list for hdu storage

        # create primary hdus and image hdus
        # data hdu
        if extension_data is None and extension_error is None and extension_mask is None and extension_errorweight is None and extension_normalization is None:
            hdus[0] = pyfits.PrimaryHDU(self._data)
            if self._error is not None:
                hdus[1] = pyfits.ImageHDU(self._error, name='ERROR')
            if self._error_weight is not None:
                hdus[2] = pyfits.ImageHDU(self._error_weight, name='ERRWEIGHT')
            if self._mask is not None:
                hdus[3] = pyfits.ImageHDU(self._mask.astype('uint8'), name='BADPIX')
            if self._normalization is not None:
                hdus[4] = pyfits.ImageHDU(self._normalization, name='NORMALIZE')
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

            if extension_normalization == 0:
                hdu = pyfits.PrimaryHDU(self._normalization)
            elif extension_normalization > 0 and extension_normalization is not None:
                hdus[extension_normalization] = pyfits.ImageHDU(self._normalization, name='NORMALIZE')

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
                    if self._datatype == 'CUBE':
                        self.setHdrValue('CRVAL3', self._wave[0])
                        self.setHdrValue('CDELT3', (self._wave[1] - self._wave[0]))
                    else:
                        self.setHdrValue('CRVAL1', self._wave[0])
                        self.setHdrValue('CDELT1', (self._wave[1] - self._wave[0]))
                hdu[0].header = self.getHeader()  # add the primary header to the HDU
                hdu[0].update_header()
            else:
                if self._wave is not None:
                    if self._datatype== 'CUBE':
                        hdu[0].header.update('CRVAL3', self._wave[0])
                        hdu[0].header.update('CDELT3', self._wave[1] - self._wave[0])
                    else:
                        hdu[0].header.update('CRVAL1', self._wave[0])
                        hdu[0].header.update('CDELT1', self._wave[1] - self._wave[0])
        hdu.writeto(filename, clobber=True)  # write FITS file to disc