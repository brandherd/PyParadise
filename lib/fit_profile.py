# -*- coding: utf-8 -*-

import numpy
from scipy import special
from scipy import linalg
from scipy import optimize
from copy import deepcopy
from multiprocessing import cpu_count
from multiprocessing import Pool
from time import sleep
try:
    import pylab
except:
    pass


fact = numpy.sqrt(2.*numpy.pi)

class fit_linearComb(object):
    """This class performs and stores results from the least squares fitting.
    """
    def __init__(self, basis, coeff=None):
        """Create an instance of fit_linearComb by supplying the template
        spectra or restore an instance by supplying the template spectra and
        the coefficients of a fit.

        Parameters
        ----------
        basis : numpy.ndarray
            A set of template spectra supplied as a 2D numpy array.
        coeff : numpy.ndarray, optional
            A 1D numpy array representing the combination of spectra in `basis`
            which provides the best fit to a certain spectrum.
        """
        self._basis = basis
        if coeff is None:
            self._coeff = numpy.zeros(self._basis.shape[1], dtype=numpy.float32)
        else:
            self._coeff = coeff

    def __call__(self):
        """Returns the spectrum which best fitted the spectrum in the function
        `fit_linearComb.fit`.
        """
        return numpy.dot(self._basis, self._coeff)

    def getCoeff(self):
        return self._coeff

    def chisq(self, y, sigma=1.0, mask=None):
        """Returns the chi^2 value between the best fit and the spectrum.

        Parameters
        ----------
        y : numpy.ndarray
            The observed spectrum as a 1D numpy array.
        sigma : float, optional
            The error spectrum corresponding to `y`.
        mask : numpy.ndarray, optional
            A mask, as a 1D boolean numpy array, representing any regions which
            where masked out during the fitting.
        """
        if mask is None:
            valid_pix = numpy.ones(len(y), dtype="bool")
        else:
            valid_pix = numpy.logical_not(mask)
        return numpy.sum((((y-self())/sigma)**2)[valid_pix])

    def fit(self, y, sigma=1.0, mask=None, negative=False):
        """Performs a least squares fit.

        Parameters
        ----------
        y : numpy.ndarray
            The observed spectrum as a 1D numpy array.
        sigma : float, optional
            The error spectrum corresponding to `y`.
        mask : numpy.ndarray, optional
            A mask, as a 1D boolean numpy array, representing any regions which
            will be masked out during the fitting.
        negative : boolean, optional
            When `negative` is False, it will apply non-negative least
            squares to determine the best fit. If `negative` is True,
            ordinary least squares fitting will be applied.

        Notes
        -----
        It is recommended to set `negative` to False in the case that it is
        expected that the flux in the spectrum cannot be less than zero.
        """
        if len(sigma)==0:
            error = numpy.ones(len(y), dtype=numpy.float32)
        else:
            error = sigma
        if mask is None:
            valid_pix = numpy.ones(len(y), dtype="bool")
        else:
            valid_pix = numpy.logical_not(mask)
        if negative==False:
            (coeff_vector, residual) = optimize.nnls((self._basis/error[:, numpy.newaxis])[valid_pix, :], y[valid_pix]/error[valid_pix])
        else:
            (coeff_vector, residual, rank, s) = linalg.lstsq((self._basis/error[:, numpy.newaxis])[valid_pix, :], y[valid_pix]/error[valid_pix])
        self._coeff = coeff_vector
        self._residual = residual


class fit_profile1D(object):
    """This class handles the fitting of line profiles.
    """
    def __init__(self, par, func, guess_par=None, args=None):
        """
        Parameters
        ----------
        par : list of dicts
            The parameters of the emission lines and additional properties.
            Each element in the list represents one emission line and the dict
            consists of the properties and corresponding values of the
            emission line.
        func : function
            A python function that will be used to fit the line profile.
        guess_par : list of dicts
            ?
        args : ?
            ?
        """
        self._par = par
        self._func = func
        self._guess_par = guess_par
        self._args=args

    def __call__(self, x):
        return self._func(x)

    def getPar(self):
        """Obtain a list of dictionaries describing the properties of the
        emission lines."""
        return self._par

    def res(self, par, x, y, sigma=1.0):
        """Calculates the residuals between the best fit and the data points
        supplied to the function.

        Parameters
        ----------
        par : list of dictionaries
            Stores the parameters of the emission lines and additional
            properties. Each element in the list represents one emission line
            and the dict consists of the properties and corresponding values of
            the emission line.
        x : numpy.ndarray or float
            The `x` points corresponding to the `y` measurements.
        y : numpy.ndarray or float
            The measurements at `x`. Should be of the same shape as `x`.
        sigma : numpy.ndarray or float, optional
            The errors for the measurements.

        Returns
        -------
        residuals : numpy.ndarray or float
            The residuals at each data point `x`.
        """
        self._par = par
        return (y-self(x))/sigma


    def residuum(self, par, x, y, sigma=1.0):
        """Calculates the sum of the residuals between the best fit and the
        data points.

        Parameters
        ----------
        par : list
            Stores the parameters of the emission lines and additional
            properties. Each element in the list represents one emission line
            and the dict consists of the properties and corresponding values of
            the emission line.
        x : numpy.ndarray or float
            The `x` points corresponding to the `y` measurements.
        y : numpy.ndarray or float
            The measurements at `x`. Should be of the same shape as `x`.
        sigma : numpy.ndarray or float, optional
            The errors for the measurements.

        Returns
        -------
        residuals : float
            The sum of the residuals from each data point.
        """
        self._par = par
        return numpy.sum(((y-self(x))/sigma)**2)

    def chisq(self, x, y, sigma=1.0):
        """Calculates the sum of the residuals between the current best fit and
        the data points.

        Parameters
        ----------
        x : numpy.ndarray or float
            The `x` points corresponding to the `y` measurements.
        y : numpy.ndarray or float
            The measurements at `x`. Should be of the same shape as `x`.
        sigma : numpy.ndarray or float, optional
            The errors for the measurements.

        Returns
        -------
        residuals : float
            The sum of the residuals from each data point.
        """
        return numpy.sum(((y-self(x))/sigma)**2)

    def fit(self, x, y, sigma=1.0, p0=None, ftol=1e-8, xtol=1e-8, maxfev=9999, err_sim=0, warning=True, method='leastsq',parallel='auto'):
        """Fits the provided set of data. The parameters of the fit and the
        corresponding errors are stored in the class. The parameters are
        available by calling the function `getPar`.

        Parameters
        ----------
        x : numpy.ndarray
            The independent variable.
        y : numpy.ndarray
            The dependent variable.
        sigma : numpy.ndarray or float, optional
            The error associated with `y`.
        p0 : list of dictionaries, optional
            The initial estimates used in fitting. Each element in the list
            represents one emission line and the dict consists of the properties
            and corresponding values of the emission line.
        ftol : float, optional
            The maximum acceptable error for fit convergence.
        xtol : float, optional
            The relative acceptable error for fit convergence.
        maxfev : int, optional
            The maximum number of calls to the function for convergence.
        err_sim : int, optional
            The number of times the fit is repeated to determine the standard
            deviation of the fit.
        warning : bool, optional
            ?
        method : {'leastsq', 'simplex'}, optional
            This argument specifies if ordinary least squares fitting
            (`leastsq`) should be applied, or if a downhill simplex algorithm
            (`simplex`) should be used.
        parallel : {'auto', int}, optional
            If parallel is not equal to one, the python multiprocessing routine
            shall be used to run parts of the code in parallel. With the option
            `auto`, it adjusts the number of parallel processes to the number
            of cpu-cores available.
        """
        if  p0 is None and self._guess_par is not None:
            self._guess_par(x, y)
        p0 = self._par
        if method=='leastsq':
            try:
                model = optimize.fmin(self.res, p0, (x, y, sigma), maxfev=maxfev, ftol=ftol, xtol=xtol,warning=warning)
                #model = optimize.leastsq(self.res, p0, (x, y, sigma), None, 0, 0, ftol, xtol, 0.0, maxfev, 0.0, 100.0, None, warning)
            except TypeError:
                model = optimize.leastsq(self.res, p0, (x, y, sigma), maxfev=maxfev,ftol=ftol, xtol=xtol)
            self._par = model[0]

        if method=='simplex':
            try:
                model = optimize.fmin(self.residuum, p0, (x, y, sigma), ftol=ftol, xtol=xtol,disp=0, full_output=0,warning=warning)
                #model = optimize.leastsq(self.res, p0, (x, y, sigma), None, 0, 0, ftol, xtol, 0.0, maxfev, 0.0, 100.0, None, warning)
            except TypeError:
                model = optimize.fmin(self.residuum, p0, (x, y, sigma), ftol=ftol, xtol=xtol,disp=0,full_output=0)
            self._par = model
            #model = optimize.leastsq(self.res, p0, (x, y, sigma),None, 0, 0, ftol, xtol, 0.0, maxfev, 0.0, 100.0, None)

        if err_sim!=0:
            numpy.random.seed()
            if parallel=='auto':
                cpus = cpu_count()
            else:
                cpus = int(parallel)
            self._par_err_models = numpy.zeros((err_sim, len(self._par)), dtype=numpy.float32)
            if cpus>1:
                pool = Pool(processes=cpus)
                results=[]
                for i in xrange(err_sim):
                    perr = deepcopy(self)
                    if method=='leastsq':
                       results.append(pool.apply_async(optimize.leastsq, args=(perr.res, perr._par, (x, numpy.random.normal(y, sigma), sigma), None, 0, 0, ftol, xtol, 0.0, maxfev, 0.0, 100, None)))
                    if method=='simplex':
                        results.append(pool.apply_async(optimize.fmin, args=(perr.residuum, perr._par, (x, numpy.random.normal(y, sigma), sigma), xtol, ftol, maxfev, None, 0, 0, 0)))
                    sleep(0.01)
                pool.close()
                pool.join()
                for i in xrange(err_sim):
                    if method=='leastsq':
                        self._par_err_models[i, :]= results[i].get()[0]
                    elif method=='simplex':
                        self._par_err_models[i, :]= results[i].get()
            else:
                for i in xrange(err_sim):
                    perr = deepcopy(self)
                    if method=='leastsq':
                        try:
                            model_err = optimize.leastsq(perr.res, perr._par, (x, numpy.random.normal(y, sigma), sigma), maxfev=maxfev, ftol=ftol, xtol=xtol, warning=warning)
                        except TypeError:
                            model_err = optimize.leastsq(perr.res, perr._par, (x, numpy.random.normal(y, sigma), sigma), maxfev=maxfev, ftol=ftol, xtol=xtol)
                        self._par_err_models[i, :] = model_err[0]
                    if method=='simplex':
                        try:
                            model_err = optimize.fmin(perr.residuum, perr._par, (x, numpy.random.normal(y, sigma), sigma), disp=0,ftol=ftol, xtol=xtol, warning=warning)
                        except TypeError:
                            model_err = optimize.fmin(perr.residuum, perr._par, (x, numpy.random.normal(y, sigma), sigma), disp=0, ftol=ftol, xtol=xtol)
                        self._par_err_models[i, :] = model_err

            self._par_err = numpy.std(self._par_err_models, 0)
        else:
            self._par_err = None


    def plot(self, x, y=None):
        """Plots the data and best fit. This requires the matplotlib package to
        be available. The data will be plotted as black circles and the best
        fit as a red line.

        Parameters
        ----------
        x : numpy.ndarray
            The `x` positions for which the best fit will be plotted.
        y : numpy.ndarray, optional
            The data points at the `x positions.`
        """
        if y is not None:
            pylab.plot(x, y, 'ok')
        pylab.plot(x, self(x), '-r')
        pylab.show()

class parFile(fit_profile1D):
    """Handles the storage guessing and loading of the parameters to the class
    fit_profile1D.

    self._par is a list
    self._parameters is a dictionary with the names as a key
    """
    def freePar(self):
        """Saves all the fixed parameters?"""
        parameters=[]
        for n in self._names:
            if self._profile_type[n]=='Gauss':
                if self._fixed[n]['flux']==1:
                    parameters.append(float(self._parameters[n]['flux']))
                if self._fixed[n]['vel']==1:
                    parameters.append(float(self._parameters[n]['vel']))
                if self._fixed[n]['disp']==1:
                    parameters.append(float(self._parameters[n]['disp']))
            if self._profile_type[n]=='TemplateScale':
                if self._fixed[n]['scale']==1:
                    parameters.append(float(self._parameters[n]['scale']))
        self._par = parameters

    def restoreResult(self):
        """No clue..."""
        m=0
        for n in self._names:
            if self._profile_type[n]=='TemplateScale':
                if self._fixed[n]['scale']==1:
                    self._parameters[n]['scale']=self._par[m]
                    m+=1
                self._parameters[n]['start_wave']=float(self._parameters[n]['start_wave'])
                self._parameters[n]['end_wave']=float(self._parameters[n]['end_wave'])
                self._parameters[n]['TemplateSpec']=self._parameters[n]['TemplateSpec']
            if self._profile_type[n]=='Gauss':
                if self._fixed[n]['flux']==1:
                    self._parameters[n]['flux'] = self._par[m]
                    m+=1
                if self._fixed[n]['vel']==1:
                    self._parameters[n]['vel'] = self._par[m]
                    m+=1
                if self._fixed[n]['disp']==1:
                    self._parameters[n]['disp']=self._par[m]
                    m+=1
                self._parameters[n]['restwave']=float(self._parameters[n]['restwave'])
        for n in self._names:
            if self._profile_type[n]=='TemplateScale':
                if self._fixed[n]['scale']!=1:
                    try:
                        float(self._parameters[n]['scale'])
                    except ValueError:
                        line = self._parameters[n]['scale'].split(':')
                        if len(line)==1:
                            self._parameters[n]['scale'] = self._parameters[line[0]]['scale']
                        else:
                            self._parameters[n]['scale'] = self._parameters[line[0]]['scale']*float(line[1])
            if self._profile_type[n]=='Gauss':
                if self._fixed[n]['flux']!=1:
                    try:
                        self._parameters[n]['flux']=float(self._parameters[n]['flux'])
                    except ValueError:
                        line = self._parameters[n]['flux'].split(':')
                        if len(line)==1:
                            self._parameters[n]['flux'] = self._parameters[line[0]]['flux']
                        else:
                            self._parameters[n]['flux'] = self._parameters[line[0]]['flux']*float(line[1])
                if self._fixed[n]['vel']!=1:
                    try:
                        self._parameters[n]['vel']=float(self._parameters[n]['vel'])
                    except ValueError:
                        self._parameters[n]['vel'] = self._parameters[self._parameters[n]['vel']]['vel']
                if  self._fixed[n]['disp']!=1:
                    try:
                        self._parameters[n]['disp']=float(self._parameters[n]['disp'])
                    except ValueError:
                        self._parameters[n]['disp'] = self._parameters[self._parameters[n]['disp']]['disp']

    def guessPar(self, x, y):
        """Make an educated guess to what the parameters should be based on the
        data"""
        w = self._guess_window
        dx = numpy.median(x[1:]-x[:-1])
        temp_y = deepcopy(y)
        for n in self._names:
            if self._profile_type[n]=='Gauss':
                restwave=float(self._parameters[n]['restwave'])
                if self._fixed[n]['vel']==1:
                    vel=float(self._parameters[n]['vel'])
                    select = numpy.logical_and(x>restwave*(vel/300000.0 +1)-w/2.0, x<restwave*(vel/300000.0 +1)+w/2.0)
                    idx=numpy.argsort(temp_y[select])
                    vel = (x[select][idx[-1]]/restwave-1)*300000.0
                    self._parameters[n]['vel'] = vel
                if self._fixed[n]['flux']==1:
                    try:
                        vel=float(self._parameters[n]['vel'])
                    except ValueError:
                        vel = float(self._parameters[self._parameters[n]['vel']]['vel'])
                    select = numpy.logical_and(x>restwave*(vel/300000.0 +1)-w/2.0, x<restwave*(vel/300000.0 +1)+w/2.0)
                    flux = numpy.sum(temp_y[select])*dx
                    self._parameters[n]['flux']=flux
                if self._fixed[n]['disp']==1:
                    try:
                        vel=float(self._parameters[n]['vel'])
                    except ValueError:
                        vel = float(self._parameters[self._parameters[n]['vel']]['vel'])
                    select = numpy.logical_and(x>restwave*(vel/300000.0 +1)-w/2.0, x<restwave*(vel/300000.0 +1)+w/2.0)
                    try:
                        width = numpy.sqrt(numpy.sum((temp_y[select]*(x[select]-restwave*(vel/300000.0 +1))**2))/(numpy.sum(temp_y[select])))
                        if width>self._spec_res and numpy.isnan(width)==False:
                            disp = numpy.sqrt(width**2-self._spec_res**2)/(restwave*(vel/300000.0+1))*300000.0
                            self._parameters[n]['disp']=disp
            # else:
            #self._parameters[n]['disp']=0.0
                    except:
                        pass
        #print(self._parameters)
        self.freePar()


    def _profile(self, x):
        y = numpy.zeros(len(x))
        m=0
        for n in self._names:
            if self._profile_type[n]=='Gauss':
                if self._fixed[n]['flux']==1:
                    flux=self._par[m]
                    self._parameters[n]['flux']=self._par[m]
                    m+=1
                else:
                    try:
                        flux = float(self._parameters[n]['flux'])
                    except ValueError:
                        line = self._parameters[n]['flux'].split(':')
                        if len(line)==1:
                            flux = self._parameters[line[0]]['flux']
                        else:
                            flux = float(self._parameters[line[0]]['flux'])*float(line[1])
                if self._fixed[n]['vel']==1:
                    vel=self._par[m]
                    self._parameters[n]['vel']=vel
                    wave = float(self._parameters[n]['restwave'])*(vel/300000.0 +1)
                    m+=1
                else:
                    try:
                        vel= float(self._parameters[n]['vel'])
                    except ValueError:
                        vel = float(self._parameters[self._parameters[n]['vel']]['vel'])
                    wave = float(self._parameters[n]['restwave'])*(vel/300000.0 +1)
                if self._fixed[n]['disp']==1:
                    disp=self._par[m]
                    self._parameters[n]['disp']=disp
                    m+=1
                else:
                    try:
                        disp= float(self._parameters[n]['disp'])
                    except ValueError:
                        disp = float(self._parameters[self._parameters[n]['disp']]['disp'])

                width=numpy.sqrt((disp/300000.0*wave)**2+self._spec_res**2)
                y += flux*numpy.exp(-0.5*((x-wave)/width)**2)/(fact*numpy.fabs(width))
        return y

    def __init__(self, file,spec_res=0):
        """Reads in a parameter file and stores the profile type, the name of
        the line and the wavelength, the flux velocity and dispersion guess.

        Parameters
        ----------
        file : str
            The filename of the parameter file.
        spec_res : float, optional
            The spectral resolution?

        Example
        -------
        The different emission lines should be seperated by at least one empty
        line. The first block of lines describing an emission line, should have
        the format "a: b"; there is no space between "a" and the colon. "a"
        represents the type of emission line (i.e. "Gauss") and "b" the name of
        the emission line. Not that there should not be any spaces in the name
        of "b": "OIII5007" is valid but "OIII 5007" not. Following the first
        line of each block, should be the properties/initial guesses of the
        line. For example, "restwave 5006.84" represents that the position line
        is guessed to be at 5006.84 and "vel 3700.0 1" represents that the
        velocity of the line is fixed (described by the 1) at 3700 km/s.

        An example of how the contents of a parameter file can be given:

        Gauss: Halpha
        restwave 6562.80
        flux 10.0 1
        vel 3700.0 1
        disp 200.0 1

        Gauss: Hbeta
        restwave 4861.33
        flux 10.0 1
        vel Halpha
        disp Halpha

        Gauss: OIII5007
        restwave 5006.84
        flux 10.0 1
        vel Halpha
        disp Halpha
        """
        fpar = open(file, 'r')
        lines = fpar.readlines()
        self._names=[]
        self._spec_res=spec_res
        self._profile_type={}
        self._parameters={}
        self._fixed={}

        par_comp={}
        par_fix ={}
        for i in range(len(lines)):
            line = lines[i].split()
            if len(line)>0:
                if line[0]=='Gauss:' or line[0]=='Cont:':
                    if len(par_comp)!=0:
                        self._parameters[self._names[-1]]=par_comp
                        self._fixed[self._names[-1]] = par_fix
                        par_comp={}
                        par_fix ={}
                    self._names.append(line[1])
                    self._profile_type[line[1]] = line[0][:-1]
                else:
                    par_comp[line[0]] = line[1]
                    if len(line)>2:
                        par_fix[line[0]] = int(line[2])
                    else:
                        par_fix[line[0]] = 0
        self._parameters[self._names[-1]]=par_comp
        self._fixed[self._names[-1]] = par_fix
        self.freePar()
        fit_profile1D.__init__(self, self._par, self._profile)




class Gaussian(fit_profile1D):
    """Represents a 1D Gaussian profile following fit_profile1D.

    `par` will be of the form [A, mu, sigma] and the profile is of the form
    A * e^(-(x - mu)^2 / (2 * sigma^2))
    """
    def _profile(self, x):
        return self._par[0]*numpy.exp(-0.5*((x-self._par[1])/abs(self._par[2]))**2)/(fact*abs(self._par[2]))

    def _guess_par(self, x, y):
        sel = numpy.isfinite(y)
        dx = numpy.median(x[1:]-x[:-1])
        self._par[0] = numpy.sum(y[sel])
        self._par[1] = numpy.sum(x[sel]*y[sel])/self._par[0]
        self._par[2] = numpy.sqrt(numpy.sum((y[sel]*(x[sel]-self._par[1])**2))/self._par[0])
        self._par[0]*=dx

    def __init__(self, par):
        fit_profile1D.__init__(self, par, self._profile, self._guess_par)

class Gaussian_const(fit_profile1D):
    """Represents a 1D Gaussian profile following fit_profile1D. This function
    differs from Gaussian because it allows a y-offset to be applied to the
    gaussian function.

    `par` will be of the form [A, mu, sigma, y_off] and the profile is of the form
    y_off + A * e^(-(x - mu)^2 / (2 * sigma^2))
    """
    def _profile(self, x):
        return self._par[0]*numpy.exp(-0.5*((x-self._par[1])/self._par[2])**2)/(fact*self._par[2]) + self._par[3]

    def _guess_par(self, x, y):
        sel = numpy.isfinite(y)
        dx = numpy.median(x[1:]-x[:-1])
        ymin = numpy.min(y[sel])
        self._par[0] = numpy.sum(y[sel]-ymin)
        self._par[1] = numpy.sum(x[sel]*(y[sel]-ymin))/self._par[0]
        self._par[2] = numpy.sqrt(numpy.sum(((y[sel]-ymin)*(x[sel]-self._par[1])**2))/(self._par[0]))
        self._par[3] = ymin
        self._par[0]*= dx

    def __init__(self, par):
        fit_profile1D.__init__(self, par, self._profile, self._guess_par)

class Gaussian_poly(fit_profile1D):
    """Represents a 1D Gaussian profile where the y-offset is represented by a
    polynomial.

    `par` will be of the form [A, mu, sigma, y_off, a1, a2, ...] and the
    profile is of the form A * e^(-(x - mu)^2 / (2 * sigma^2)) + polynomial.
    The polynomial is of the form a1 * x^(n-1) + a2 * x^(n-2) with the
    polynomial coefficients a1, a2, ...
    """
    def _profile(self, x):
        return self._par[0]*numpy.exp(-0.5*((x-self._par[1])/self._par[2])**2)/(fact*self._par[2]) + numpy.polyval(self._par[3:],x)

    def _guess_par(self, x, y):
        sel = numpy.isfinite(y)
        dx = abs(x[1]-x[0])
        self._par[0] = numpy.sum(y[sel])
        self._par[1] = numpy.sum(x[sel]*y[sel])/self._par[0]
        self._par[2] = numpy.sqrt(numpy.sum((y[sel]*(x[sel]-self._par[1])**2))/self._par[0])
        self._par[0]*=dx

    def __init__(self, par):
        fit_profile1D.__init__(self, par, self._profile, self._guess_par)

class Gaussians(fit_profile1D):
    """Represents a sum of 1D Gaussian profiles. The parameters are like for
    Gaussian1D, except that their are repeated n times for n Gaussians."""
    def _profile(self, x):
        y = numpy.zeros(len(x), dtype=numpy.float32)
        ncomp = len(self._par)/3
        for i in xrange(ncomp):
            y += self._par[i]*numpy.exp(-0.5*((x-self._par[i+ncomp])/abs(self._par[i+2*ncomp]))**2)/(fact*abs(self._par[i+2*ncomp]))
        return y

    def __init__(self, par):
        fit_profile1D.__init__(self, par, self._profile)

class Gaussians_width(fit_profile1D):
    """Represents a sum of 1D Gaussian profiles, like Gaussians, but with the
    difference that they can all have the same width.
    """
    def _profile(self, x):
        y = numpy.zeros(len(x))
        ncomp = len(self._args)
        for i in xrange(ncomp):
            y += self._par[i+1]*numpy.exp(-0.5*((x-self._args[i])/self._par[0])**2)/(fact*self._par[0])
        return y

    def __init__(self, par, args):
        fit_profile1D.__init__(self, par, self._profile, args=args)

class Gaussians_offset(fit_profile1D):
    """Represents a sum of 1D Gaussian profiles, like Gaussians_width, but were
    the mean position is offsetted.
    """
    def _profile(self, x):
        y = numpy.zeros(len(x))
        ncomp = len(self._args)
        for i in xrange(ncomp):
            y += self._par[i+1]*numpy.exp(-0.5*((x-self._args[i]+self._par[-1])/self._par[0])**2)/(fact*self._par[0])
        return y

    def __init__(self, par, args):
        fit_profile1D.__init__(self, par, self._profile, args=args)



class Gauss_Hermite(fit_profile1D):
    """Represents a Gauss-Hermite quadrature profile.

    `par` will be of the form [a, mu, sigma, h3, h4] where h3 and h4 are
    the roots of the third and the fourth hermite polynomial.
    """
    def _profile(self, x):
        a, mean, sigma, h3, h4 = self._par
        w  = (x-mean)/sigma
        H3 = (2.828427*w**3 - 4.242641*w)*0.408248
        H4 = (4.*w**4 - 12.*w**2 + 3.)*0.204124
        y = a*numpy.exp(-0.5*w**2)*(1. + h3*H3 + h4*H4)/(fact*sigma)
        return y

    def _guess_par(self, x, y):
        sel = numpy.isfinite(y)
        self._par[0] = numpy.sum(y[sel])
        self._par[1] = numpy.sum(x[sel]*y[sel])/self._par[0]
        self._par[2] = numpy.sqrt(numpy.sum((y[sel]*(x[sel]-self._par[1])**2))/self._par[0])
        self._par[3] = 0.
        self._par[4] = 0.

    def __init__(self, par):
        fit_profile1D.__init__(self, par, self._profile, self._guess_par)

class Exponential_constant(fit_profile1D):
    """Represents an exponential profile with a y-offset.

    `par` will be of the form [A, time, y_off] representing the
    function A * e^(t/time) + y_off
    """
    def _profile(self, x):
        scale, time, const = self._par
        y = scale*numpy.exp(x/time)+const
        return y

    def __init__(self, par):
        fit_profile1D.__init__(self, par, self._profile)

class LegandrePoly(object):
    """Represents a legendre polynomial profile.
    """
    def __init__(self, coeff, min_x=None, max_x=None):
        self._min_x = min_x
        self._max_x = max_x
        self._coeff = coeff
        self._poly =[]
        for i in range(len(coeff)):
            self._poly.append(special.legendre(i))

    def __call__(self, x):
        y = numpy.zeros(len(x), dtype=numpy.float32)
        if self._min_x is None:
            self._min_x = numpy.min(x)
        if self._max_x is None:
            self._max_x = numpy.max(x)
        x_poly = (x-self._min_x)*1.98/numpy.abs((numpy.abs(self._max_x)-numpy.abs(self._min_x)))-0.99
        for i in range(len(self._coeff)):
            y+=self._poly[i](x_poly)*self._coeff[i]
        return y

    def fit(self, x, y):
        eigen_poly = numpy.zeros(( len(x), len(self._coeff)), dtype=numpy.float32)
        for i in range(len(self._coeff)):
                self._coeff = numpy.zeros(len(self._coeff))
                self._coeff[i]=1
                eigen_poly[:, i] = self(x)
#        print(eigen_poly, y)
        self._coeff=numpy.linalg.lstsq(eigen_poly, y)[0]


