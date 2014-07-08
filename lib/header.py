import pyfits

class Header(object):
    def __init__(self, header=None, cardlist=None, origin=None):
        """
            Creates an Header object
            
            Parameters
            --------------
            header : pyfits.header object, optional
                    Fits header as header
            cardlist : pyfits.CardList object, optional
                    Fits header as a card list,
                    if header is given cardlist parameter will be ignored
            origin : string, optional
                    Name of the Fits file as the origin for the header,
                    can be the full path of the file
                       
        """
        if header != None:
            # Assign private variable and convert header to card list
            self._header = header
        else:
            # Create empty Header and CardList objects
            self._header = None
        
        # Set the Fits file origin of the header if given
        if origin != None:
            self._origin = origin
        else:
            self._origin = None
            
    def setHeader(self, header, origin=None):
        self._header = header
        self._origin=origin
    
    def loadFitsHeader(self, filename,  extension = 0, removeEmpty=0):    
        """
            Loads the header information from a Fits file
            
            Parameters
            ---------------
            filename : string
                        Filename of the Fits file from which the header should be loaded.
                        The full path to the file can be given.
            extension : integer, optional
                        Extenstion of the Fits file from the header shall be read
            removeEmpty : integer (0 or 1), optional
                        Removes empty entries from the header if set to 1.
        """
        self._header = pyfits.getheader(filename, ext = extension)
        self._origin = filename
        if removeEmpty==1:
            self.removeHdrEntries()
        
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
        
        if filename==None:
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
    
        
    #def copyHdrKey(self, Header, key):
        #new_cardlist = self._cardlist+[Header.getHdrCard(key)]
        #self._cardlist = pyfits.CardList(new_cardlist)
        #self._header = pyfits.Header(self._cardlist)
        
    #def appendHeader(self, Header):
        #new_cardlist = self._cardlist+Header.getHdrCardlist()
        #self._cardlist = pyfits.CardList(new_cardlist)
        #self._header = pyfits.Header(self._cardlist)
        
        
    def setHdrValue(self,  keyword,  value,  comment=None):
        if self._header==None:
            self._header=pyfits.Header()
        if comment==None:
            self._header.update(keyword, value)
        else:
            self._header.update(keyword, value, comment)
        
    def extendHierarch(self, keyword, add_prefix, verbose=1):
        if self._header!=None:
            if self._header.has_key(add_prefix.upper()+' '+keyword.upper())==0:
                self._header.rename_key(keyword, 'hierarch '+add_prefix+' '+keyword)
            else:
                if verbose==1:
                    print "The keyword %s does already exists!"%(add_prefix.upper()+' '+keyword.upper())
        else:
            pass
       
#def combineHdr(headers):
    #combHdr = []
    #for i in range(len(headers)):
        #if i==0:
            #final_card = headers[i]._header.ascardlist()
        #final_keys = final_card.keys()
        #if i>0:
            #card = headers[i]._header.ascardlist()
            #keys = card.keys()
            #for k in keys:
                #if not k in final_keys:
                    #final_card=final_card+[card[k]]
            #final_card = pyfits.CardList(final_card)
    #outHdr = Header(pyfits.Header(cards=final_card))
    #return outHdr
    
    
