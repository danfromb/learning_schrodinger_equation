import numpy as np
from PIL import Image
import constants

class BaseModel:
    
    @classmethod
    def fromDefaultParameters(cls):
        model = cls()
        model._setDefaultParameters()
        model.initHamiltonian()
        return model
    
    @classmethod
    def fromPotentialImage(cls, filePath): 
        model = cls()
        image = Image.open(filePath)
        model._setParametersFromImage(image)
        model.initHamiltonian()
        return model
    
    def _setDefaultParameters(self):
        self._setDefaultLatticeSize()
        self._initSystemParametersForLatticeConstant(1)
        modelShape = (self.N, self.M)
        self.V = np.zeros(modelShape)

    def _setDefaultLatticeSize(self):        
        self.N = 2
        self.M = 2
        
    def _initSystemParametersForLatticeConstant(self, latticeConstant):
        self.a = latticeConstant
        self.t = 1 / (2 * self.a**2)
        
    def initHamiltonian(self):              
        self.initEmptyModel()
        self.addOnsiteEnergies()
        self.addHoppings()
        
    def initEmptyModel(self):
        raise NotImplementedError()
        
    def addOnsiteEnergies(self):
        raise NotImplementedError()
        
    def addHoppings(self):
        raise NotImplementedError()
        
    def _setParametersFromImage(self, image):
        self.M, self.N = image.size
        self._initSystemParametersForLatticeConstant(1)
        self.V = self._convertImageToPotential(image)
        
    def _convertImageToPotential(self, image):
        potentialData = np.array(image)
        # white (255) gets interpreted as maximal potential while black (0) gets interpreted as potential zero potential
        potentialData = constants.maxPotential * potentialData / 255
        return potentialData
    
    def calcStateCoordinates(self):
        raise NotImplementedError()
        