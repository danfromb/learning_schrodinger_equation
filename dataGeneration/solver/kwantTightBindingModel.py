import kwant
import solver.baseTightBindingModel
    
class Model(solver.baseTightBindingModel.BaseModel):
    
    @property
    def H(self):
        finalizedSystem = self.system.finalized()
        return finalizedSystem.hamiltonian_submatrix()
        
    def initEmptyModel(self):
        self.lattice = kwant.lattice.square(self.a)
        self.system = kwant.Builder()
            
    def addOnsiteEnergies(self):
        for i in range(self.M):
            for j in range(self.N):
                self.system[self.lattice(i, j)] = self._onsiteEnergy(i, j)

    def _onsiteEnergy(self, i, j):
        # image data is stored from top left to bottom right (first index denoting the rows)
        # kwant orders the lattice from bottom left to top right (first index denoting the columns)
        # hence the indeces to V (image data) are 'flipped' (i is second index) and first index counting the rows
        # count from top down
        return 4 * self.t + self.V[self.N - 1 - j][i]
        
    def addHoppings(self):
        self.system[self.lattice.neighbors()] = -self.t
    
    def calcStateCoordinates(self):
        systemSites = kwant.plotter.sys_leads_sites(self.system)[0]
        stateCoordinates = kwant.plotter.sys_leads_pos(self.system, systemSites)
        return stateCoordinates
    