'''
This represents the tight binding model obtained from discretizing the hamiltonian on a square lattice.
The sites are numbered from left to right and then from top to bottom, i.e.:
1  2  3  4  5
6  7  8  9  10
11 12 13 14 15
'''

import numpy as np
import solver.baseTightBindingModel
import scipy.sparse

class Model(solver.baseTightBindingModel.BaseModel):
        
    def initEmptyModel(self):
        hamiltonianShape = (self.N * self.M, self.N * self.M)
        self.H = scipy.sparse.dia_matrix(hamiltonianShape)
        
    def addOnsiteEnergies(self):
        diagonal = self._calcDiagonalForOnsiteEnergies()
        diagonalMatrix = scipy.sparse.diags(diagonal)
        self.H = self.H + diagonalMatrix
        
    def _calcDiagonalForOnsiteEnergies(self):
        diagonalLength = self.N * self.M
        hoppingOnsiteEnergy = np.full((diagonalLength), 4 * self.t)
        potentialOnsiteEnergy = self.V.reshape((diagonalLength))
        return hoppingOnsiteEnergy + potentialOnsiteEnergy
        
    def addHoppings(self):
        self._addHoppingInXDirection()
        self._addHoppingInYDirection()
    
    def _addHoppingInXDirection(self):
        offDiagonal = self._calcOffDiagonalForHoppingInXDirection()
        offDiagonalMatrix = scipy.sparse.diags(offDiagonal, offsets = 1)
        offDiagonalMatrix = offDiagonalMatrix + scipy.sparse.diags(offDiagonal, offsets = -1)
        self.H = self.H + offDiagonalMatrix
        
    def _calcOffDiagonalForHoppingInXDirection(self):
        hoppingsForOneRow = np.full((self.M - 1), -self.t)
        offDiagonal = hoppingsForOneRow
        for i in range(self.N - 1):
            offDiagonal = np.append(offDiagonal, 0)
            offDiagonal = np.append(offDiagonal, hoppingsForOneRow)
        return offDiagonal
        
    def _addHoppingInYDirection(self):
        diagonalLength = (self.N - 1) * self.M
        offDiagonal = np.full((diagonalLength), -self.t)
        offDiagonalMatrix = scipy.sparse.diags(offDiagonal, offsets = self.M)
        offDiagonalMatrix = offDiagonalMatrix + scipy.sparse.diags(offDiagonal, offsets = -self.M)
        self.H = self.H + offDiagonalMatrix
        
    def calcStateCoordinates(self):
        coordinates = []
        for i in range(self.N):
            for j in range(self.M):
                coordinates.append([j * self.a, -i * self.a])
        return np.array(coordinates)