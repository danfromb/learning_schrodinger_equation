import numpy as np

class Projector:
    
    def __init__(self, eigenvectors):
        self._eigenvectors = eigenvectors
        
    def eigenbasisExpansionCoefficients(self, vec):
        coefficients = []
        for eigenvector in self._eigenvectors:
            coefficient = np.vdot(eigenvector, vec)
            coefficients.append(coefficient)
        return coefficients