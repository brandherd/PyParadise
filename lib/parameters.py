from UserDict import UserDict
import numpy
import Paradise
import pylab

class Parameter(object):
    """A container for storing a parameter value and the corresponding
    description.

    Parameters
    ----------
    value : float
        The value of the parameter that we want to store.
    description : string, optional
        The type of object that is stored.
    """
    def __init__(self, value, description=''):
        self.setValue(value)
        self.setDescription(description)

    def getValue(self):
        """Obtain the value of the parameter."""
        return self.__value

    def getDescription(self):
        """Get the description of the parameter as a string."""
        return self.__description

    def setValue(self, value):
        """Change the value of the parameter."""
        self.__value = value

    def setDescription(self, description):
        """Change the description of the parameter by providing a string."""
        self.__description = description


class ParameterList(UserDict):
    def __init__(self, filename=None):
        UserDict.__init__(self)
        if filename is not None:
            infile = open(filename, 'r')
            lines = infile.readlines()
            infile.close()
            for i in range(len(lines)):
                (prepart, description) = lines[i].split('!')
                (parname, value) = prepart.split()
                self[parname] = Parameter(value, description.replace('\n', ''))

    def addParameter(self, Name, parameter):
        self[Name] = parameter


class CustomMasks(UserDict):
    def __init__(self, filename):
        UserDict.__init__(self)
        infile = open(filename, 'r')
        lines = infile.readlines()
        infile.close()
        fixed_frame = []
        rest_frame = []
        fixed_found = 0
        rest_found = 0
        for i in range(len(lines)):
            line = lines[i].split()
            if len(line) == 1 and line[0] == '[observed-frame]':
                fixed_found = 1
                rest_found = 0
            elif len(line) == 1 and line[0] == '[rest-frame]':
                fixed_found = 0
                rest_found = 1
            elif len(line) == 2 and fixed_found == 1:
                fixed_frame.append([float(line[0]), float(line[1])])
            elif len(line) == 2 and rest_found == 1:
                rest_frame.append([float(line[0]), float(line[1])])
        self['observed_frame'] = numpy.array(fixed_frame)
        self['rest_frame'] = numpy.array(rest_frame)

    def maskPixelsObserved(self, wave, redshift, init_mask=None):
        if init_mask is None:
            mask = numpy.zeros(len(wave), dtype="bool")
        else:
            mask = init_mask
        for i in range(len(self['observed_frame'])):
            mask = numpy.logical_or(mask, (wave >= self['observed_frame'][i][0]) & (wave <= self['observed_frame'][i][1]))
        for i in range(len(self['rest_frame'])):
            mask = numpy.logical_or(mask, (wave >= self['rest_frame'][i][0] * (1 + redshift)) & (wave <= self['rest_frame'][i][1]
            * (1 + redshift)))
        return mask

    def maskPixelsRest(self, wave, redshift, init_mask=None):
        if init_mask is None:
            mask = numpy.zeros(len(wave), dtype="bool")
        else:
            mask = init_mask
        for i in range(len(self['observed_frame'])):
            mask = numpy.logical_or(mask, (wave >= self['observed_frame'][i][0] / (1 + redshift)) & (wave <=
                self['observed_frame'][i][1] / (1 + redshift)))
        for i in range(len(self['rest_frame'])):
            mask = numpy.logical_or(mask, (wave > self['rest_frame'][i][0]) & (wave <= self['rest_frame'][i][1]))
        return mask

    def maskSpectrum(self, inputSpec, redshift):
        wave = inputSpec.getWave()
        values = inputSpec.getData()
        error = inputSpec.getError()
        mask = inputSpec.getMask()
        new_mask = self.maskPixelsObserved(wave=wave, redshift=redshift, init_mask=mask)
        new_spectrum = Paradise.Spectrum1D(wave=wave, data=values, error=error, mask=new_mask)
        return new_spectrum
