import numpy as np

class GaussianWavePacket:
    
    def __init__(self):
        self.centerPoint = np.array([0, 0])
        self.variance = 1
        self.k_0 = np.array([0, 0])
        
    def toVectorForStateCoordinates(self, stateCoordinates):
        nonNormalizedVector = self._calcNonNormalizedVectorForStateCoordinates(stateCoordinates)
        normalizedVector = self._calcNormalizedVector(nonNormalizedVector)
        return normalizedVector
    
    def _calcNonNormalizedVectorForStateCoordinates(self, stateCoordinates):
        vector = []
        for point in stateCoordinates:
            nonNormalizedValue = self._calcNonNormalizedValueAtPoint(point)
            vector.append(nonNormalizedValue)
        return np.array(vector)
        
    def _calcNonNormalizedValueAtPoint(self, point):
        absoluteValue = self._calcAbsoluteValueAtPoint(point)
        phase = np.dot(self.k_0, point)
        nonNormalizedValue = absoluteValue * np.exp(1j * phase)
        return nonNormalizedValue
    
    def _calcAbsoluteValueAtPoint(self, point):
        distanceFromCenterPoint = np.subtract(point, self.centerPoint)
        squaredDeviation = np.linalg.norm(distanceFromCenterPoint)**2
        exponent = -1 * squaredDeviation / ( 4 * self.variance )
        absoluteValue = np.exp(exponent)
        return absoluteValue
    
    def _calcNormalizedVector(self, vector):
        norm = np.linalg.norm(vector)
        normalizedVector = 1 / norm * vector
        return normalizedVector
    