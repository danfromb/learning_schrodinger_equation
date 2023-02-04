import numpy as np
import solver.projector
    
class System:
    
    def __init__(self):
        self.eigenenergies = NotImplemented
        self.eigenstates = NotImplemented
        self._stateCoordinates = _StateCoordinates()
        self.stateExpansionCoefficients = NotImplemented
        self.N = NotImplemented
        self.M = NotImplemented
        
    @property
    def stateCoordinates(self):
        return self._stateCoordinates.stateCoordinates
    
    @stateCoordinates.setter
    def stateCoordinates(self, stateCoordinates):
        self._stateCoordinates.stateCoordinates = stateCoordinates
    
    def setInitialState(self, state):
        projector = solver.projector.Projector(self.eigenstates)
        self.stateExpansionCoefficients = projector.eigenbasisExpansionCoefficients(state)
        
    def calcWaveFunctionAtTime(self, t):        
        prefactors = np.multiply(np.exp(-1j * t * self.eigenenergies), self.stateExpansionCoefficients)
        waveFunction = self.eigenstates.transpose().dot(prefactors)
        return waveFunction
    
    def calcAspectRatioOfStateCoordinates(self):
        return self._stateCoordinates.calcAspectRatioOfStateCoordinates()
    

class _StateCoordinates():
    
    def __init__(self):
        self.stateCoordinates = None
        
    def calcAspectRatioOfStateCoordinates(self):
        xCoordinates, yCoordinates = self.stateCoordinates.transpose()
        xAxisLength = self._calcAxisLength(xCoordinates)
        yAxisLength = self._calcAxisLength(yCoordinates)
        return yAxisLength / xAxisLength
        
    def _calcAxisLength(self, stateCoordinatesAlongAxis):
        minimum = np.amin(stateCoordinatesAlongAxis)
        maximum = np.amax(stateCoordinatesAlongAxis)
        return maximum - minimum
