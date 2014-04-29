from Paradise.lib.cube import Cube
import pyfits
import numpy
from scipy import ndimage
from copy import deepcopy

def RSSToCube(rss):
    """
    Transform an array with spectra in RSS-format to an array containing
    spectra in Cube format (last dimension is of length 1).
    """
    if rss is None or rss.ndim == 3:
        return rss
    return rss.reshape((rss.shape[0], rss.shape[1], 1))

def CubeToRSS(cube):
    """
    Transform an array with spectra in cube-format into an array with
    spectra in RSS-format. It checks if one of the dimensions is zero, else
    an assertion error is raised.
    """
    if cube is None or cube.ndim == 2:
        return cube
    assert cube.shape[2] == 1 or cube.shape[1] == 1
    return cube.reshape((cube.shape[0], cube.shape[1]))

class RSS(Cube):

    def __init__(self, data=None, wave=None, error=None, mask=None, error_weight=None, normalization=None, header=None):
        if data is not None:
            data = RSSToCube(data)
        if error is not None:
            error = RSSToCube(error)
        if error_weight is not None:
            error_weight = RSSToCube(error_weight)
        if mask is not None:
            mask = RSSToCube(mask)
        if normalization is not None:
            normalization = RSSToCube(normalization)
        Cube.__init__(self, data, wave, error, mask, error_weight, normalization, header)
    
    ## One can use the subCubeWave, normalizeSpec and unnormalizedSpec of the Cube class
    ## if one replaces the `cube_out = RSS(...)' and `cube_out = Cube(...)' by
    ## `cube_out = self.__class__(...)' and therefore replace the instances below.
    ## I only have no idea if it is bad coding and if it can have any side effects.
    def subCubeWave(self, wave_start=None, wave_end=None):
        select = numpy.ones(len(self.getWave()), dtype='bool')
        if wave_start is not None:
            select[self.getWave() < wave_start] = False
        if wave_end is not None:
            select[self.getWave() > wave_end] = False
        if self._error is not None:
            error = self._error[select]
        else:
            error = None
        if self._mask is not None:
            mask = self._mask[select]
        else:
            mask = None
        if self._normalization is not None:
            normalization = self._normalization[select]
        else:
            normalization = None
        cube_out = RSS(wave=self.getWave()[select], data=self._data[select], error=error, mask=mask,
        normalization=normalization)
        return cube_out

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
        cube_out = RSS(wave=self._wave, data=new_data, error=new_error, mask=self._mask, normalization=mean)
        return cube_out

    def unnormalizedSpec(self):
        data = self.__data * self.__normalization
        if self.__error is not None:
            error = self.__error * self.__normalization
        else:
            error = None
        cube_out = RSS(wave=self._wave, data=data, error=error, mask=self._mask, normalization=None)
        return cube_out

    def loadFitsData(self, file, extension_data=None, extension_mask=None, extension_error=None, extension_errorweight=None,
    extension_hdr=0):
        """
            Load data from a FITS image into an RSS object

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
            self._data = RSSToCube(hdu[0].data)
            self._res_elements = self._data.shape[0]
            self._dim_y = self._data.shape[1]
            self._dim_x = self._data.shape[2]
            self.setHeader(header=hdu[extension_hdr].header, origin=file)
            if len(hdu) > 1:
                for i in range(1, len(hdu)):
                    if hdu[i].header['EXTNAME'].split()[0] == 'ERROR':
                        self._error = RSSToCube(hdu[i].data)
                    elif hdu[i].header['EXTNAME'].split()[0] == 'BADPIX':
                        self._mask = RSSToCube(hdu[i].data).astype('bool')
                    elif hdu[i].header['EXTNAME'].split()[0] == 'ERRWEIGHT':
                        self._error_weight = RSSToCube(hdu[i].data)
        else:
            self.setHeader(header=hdu[extension_hdr].header, origin=file)
            if extension_data is not None:
                self._data = RSSToCube(hdu[extension_data].data)
                self._res_elements = self._data.shape[0]
                self._dim_y = self._data.shape[1]
                self._dim_x = self._data.shape[2]

            if extension_mask is not None:
                self._mask = RSSToCube(hdu[extension_mask].data)
                self._res_elements = self._mask.shape[0]
                self._dim_y = self._mask.shape[1]
                self._dim_x = self._mask.shape[2]
            if extension_error is not None:
                self._error = RSSToCube(hdu[extension_error].data)
                self._res_elements = self._error.shape[0]
                self._dim_y = self._error.shape[1]
                self._dim_x = self._error.shape[2]
            if extension_error is not None:
                self._error_weight = RSSToCube(hdu[extension_errorweight].data)
                self._res_elements = self._error_weight.shape[0]
                self._dim_y = self._error_weight.shape[1]
                self._dim_x = self._error_weight.shape[2]
        try:
            self._wave = numpy.arange(self._res_elements) * self.getHdrValue('CDELT1') + self.getHdrValue('CRVAL1')
        except KeyError:
            self._wave = numpy.arange(self._res_elements) * self.getHdrValue('CD1_1') + self.getHdrValue('CRVAL1')
        hdu.close()

        if extension_hdr is not None:
            self.setHeader(hdu[extension_hdr].header, origin=file)

    def writeFitsData(self, filename, extension_data=None, extension_mask=None, extension_error=None,
    extension_errorweight=None):
        """
            Save information from an RSS object into a FITS file.
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
            hdus[0] = pyfits.PrimaryHDU(CubeToRSS(self._data))
            if self._error is not None:
                hdus[1] = pyfits.ImageHDU(CubeToRSS(self._error), name='ERROR')
            if self._error_weight is not None:
                hdus[2] = pyfits.ImageHDU(CubeToRSS(self._error_weight), name='ERRWEIGHT')
            if self._mask is not None:
                hdus[3] = pyfits.ImageHDU(CubeToRSS(self._mask).astype('uint8'), name='BADPIX')
        else:
            if extension_data == 0:
                hdus[0] = pyfits.PrimaryHDU(CubeToRSS(self._data))
            elif extension_data > 0 and extension_data is not None:
                hdus[extension_data] = pyfits.ImageHDU(CubeToRSS(self._data), name='DATA')

            # mask hdu
            if extension_mask == 0:
                hdu = pyfits.PrimaryHDU(CubeToRSS(self._mask).astype('uint8'))
            elif extension_mask > 0 and extension_mask is not None:
                hdus[extension_mask] = pyfits.ImageHDU(CubeToRSS(self._mask).astype('uint8'), name='BADPIX')

            # error hdu
            if extension_error == 0:
                hdu = pyfits.PrimaryHDU(CubeToRSS(self._error))
            elif extension_error > 0 and extension_error is not None:
                hdus[extension_error] = pyfits.ImageHDU(CubeToRSS(self._error), name='ERROR')

            if extension_errorweight == 0:
                hdu = pyfits.PrimaryHDU(CubeToRSS(self._error_weight))
            elif extension_errorweight > 0 and extension_errorweight is not None:
                hdus[extension_errorweight] = pyfits.ImageHDU(CubeToRSS(self._error_weight), name='ERRWEIGHT')

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
                    self.setHdrValue('CRVAL1', self._wave[0])
                    self.setHdrValue('CDELT1', (self._wave[1] - self._wave[0]))
                hdu[0].header = self.getHeader()  # add the primary header to the HDU
                hdu[0].update_header()
            else:
                if self._wave is not None:
                    hdu[0].header.update('CRVAL1', self._wave[0])
                    hdu[0].header.update('CDELT1', self._wave[1] - self._wave[0])
        hdu.writeto(filename, clobber=True)  # write FITS file to disc

def loadRSS(infile, extension_data=None, extension_mask=None, extension_error=None):

    rss = RSS()
    rss.loadFitsData(infile, extension_data=None, extension_mask=None, extension_error=None)

    return rss



