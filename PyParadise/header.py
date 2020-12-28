import astropy.io.fits as pyfits


class Header(object):
    def __init__(self, header=None, origin=None):
        """
            Creates an Header object
            
            Parameters
            --------------
            header : pyfits.header object, optional
                    Fits header as header
            origin : string, optional
                    Name of the Fits file as the origin for the header,
                    can be the full path of the file
                       
        """
        if header is not None:
            # Assign private variable and convert header to card list
            self._header = header
        else:
            # Create empty Header and CardList objects
            self._header = None
        
        # Set the Fits file origin of the header if given
        if origin is not None:
            self._origin = origin
        else:
            self._origin = None
            
    def setHeader(self, header, origin=None):
        self._header = header
        self._origin = origin
    
    def loadFitsHeader(self, filename,  extension=0):
        """
            Loads the header information from a Fits file
            
            Parameters
            ---------------
            filename : string
                        Filename of the Fits file from which the header should be loaded.
                        The full path to the file can be given.
            extension : integer, optional
                        Extension of the Fits file from the header shall be read
        """
        self._header = pyfits.getheader(filename, ext=extension)
        self._origin = filename
        
    def writeFitsHeader(self, filename=None, extension=0):
        """
            Writes the header to an existing Fits file
            
            Parameters:
            ---------------
            filename : string, optional
                        Filename of the Fits file to which the header is written.
                        The full path to the file can be given.
                        If filename is none, the value of _origin ise used.
            extenstion : integer, optional
                        Extension of the Fits file to which the header is written.
        """
        
        if filename is None:
            f_out = self._origin
        else:
            f_out = filename
        hdu = pyfits.open(f_out, mode='update')
        hdu[extension].header = self._header
        hdu[extension].update_header()
        hdu.flush()

    def getHdrValue(self, keyword):
        """
            Returns the value of a certain keyword in the header
        
            Parameters:
            ---------------
            keyword : string
                        valid keyword in the header
            
            Returns:
            ---------------
            out : string, integer or float
                        stored value in the header for the given keyword
        """
        return self._header[keyword]

    def getHdrKeys(self):
        """
            Returns all valid keywords of the Header
        
            Returns:
            ---------------
            out : list
                        list of strings representing the keywords in the header
        """
        return self._header.keys()
        
    def getHeader(self):
        return self._header
        
    def setHdrValue(self,  keyword,  value,  comment=None):
        if self._header is None:
            self._header = pyfits.Header()
        if comment is None:
            try:
                self._header.update(keyword, value)
            except ValueError:
                self._header[keyword] = (value)
        else:
            try:
                self._header.update(keyword, value, comment)
            except ValueError:
                self._header[keyword] = (value, comment)
