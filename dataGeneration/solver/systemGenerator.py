import numpy as np
import scipy.sparse.linalg
import solver.system
import solver.matrixTightBindingModel
from PIL import Image

class SystemGenerator:
    
    def __init__(self):
        self.tightBindingModelBackend = solver.matrixTightBindingModel

    def fromDefaultParameters(self):
        tightBindingModel = self.tightBindingModelBackend.Model.fromDefaultParameters()
        system = self.fromTightBindingModel(tightBindingModel)
        return system
    
    def fromPotentialImage(self, imagePath):
        tightBindingModel = self.tightBindingModelBackend.Model.fromPotentialImage(imagePath)
        system = self.fromTightBindingModel(tightBindingModel)
        return system
    
    def fromTightBindingModel(self, tightBindingModel):
        system = solver.system.System()
        #stateCoordinates = tightBindingModel.calcStateCoordinates()
        eigenenergy = self._calcEigensystemFromTightBindingModel(tightBindingModel)
        system.eigenenergies = eigenenergy
        #system.eigenstates = eigenstates
        #system.stateCoordinates = stateCoordinates
        #system.N = tightBindingModel.N
        #system.M = tightBindingModel.M
        return system
    
    def _calcEigensystemFromTightBindingModel(self, tightBindingModel):
        hamiltonian = tightBindingModel.H
        eigenvalue = scipy.sparse.linalg.eigsh(hamiltonian, k = 1, which = 'SM', return_eigenvectors = False)
        return eigenvalue
    
    def fromBinaryData(self, potentialFolder):
        system = solver.system.System()
        system.eigenenergies = np.load(potentialFolder + 'eigenenergies.npy')
        system.eigenstates = np.load(potentialFolder + 'eigenstates.npy')
        system.stateCoordinates = np.load(potentialFolder + 'stateCoordinates.npy')
        image = Image.open(potentialFolder + 'potential.png')
        system.M, system.N = image.size
        return system
