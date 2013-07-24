from UserDict import UserDict

class Parameter(object):
    def __init__(self, value, description=''):
        self.setValue(value)
        self.setDescription(description)
        
    def getValue(self):
        return self.__value
        
    def getDescription(self):
        return self.__description
        
    def setValue(self, value):
        self.__value = value
        
    def setDescription(self, description):
        self.__description = description

class ParameterList(UserDict):
    def __init__(self, filename=None):
        UserDict.__init__(self)
        if filename!=None:
            infile = open(filename, 'r')
            lines = infile.readlines()
            infile.close()
            for i in range(len(lines)):
                (prepart, description) = lines[i].split('!')
                (parname, value) = prepart.split()
                self[parname] = Parameter(value, description.replace('\n', ''))
            
