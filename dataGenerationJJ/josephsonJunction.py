# -*- coding: utf-8 -*-

import kwant
import tinyarray
import numpy as np
import scipy.sparse.linalg as sla
from time import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import multiprocessing as mp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
name = MPI.Get_processor_name().split('.')[0]

L = 30
W = 20
SCGap = 20
t = -1
spinOrbit = 0.75
superconductingGap = 0.2
spinScattering = 0.1
#numberOfScatteres = 10

tau_0z = tinyarray.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
#tau_xz = tinyarray.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]])
#tau_yz = tinyarray.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 1j, 0, 0], [0, 0, 0, -1j]])
tau_zz = tinyarray.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
tau_0x = tinyarray.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
tau_0y = tinyarray.array([[0, 0, -1j, 0], [0, 0, 0, -1j], [1j, 0, 0, 0], [0, 1j, 0, 0]])
#tau_x0 = tinyarray.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
tau_z0 = tinyarray.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

lat = kwant.lattice.honeycomb()
a, b = lat.sublattices
hoppings = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
hoppingsSpinOrbit = (((-1, 1), a, a), ((0, 1), a, a), ((1, 0), a, a), 
                     ((1, -1), a, a), ((0, -1), a, a), ((-1, 0), a, a), 
                     ((-1, 1), b, b), ((0, 1), b, b), ((1, 0), b, b), 
                     ((1, -1), b, b), ((0, -1), b, b), ((-1, 0), b, b))
    
def buildGaussians(scattererIndices):
    numberOfScatteres = scattererIndices.shape[0]
    standardDeviation = 1
    covarrianceMatrix = [[standardDeviation, 0], [0, standardDeviation]]
    gaussians = np.zeros((numberOfScatteres), dtype = object)
    for i in range(numberOfScatteres):
        means = a(*scattererIndices[i]).pos
        gaussians[i] = multivariate_normal(means, covarrianceMatrix).pdf
    return gaussians
  
def makeSystem():
    sys = kwant.Builder()    
    sys[(a(i, j) for i in range(L) for j in range(W))] = onsiteEnergy
    sys[(b(i, j) for i in range(L) for j in range(W))] = onsiteEnergy          
    sys[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = t * tau_0z
    sys[[kwant.builder.HoppingKind(*hopping) for hopping in hoppingsSpinOrbit]] = spinOrbitHopping
    return sys
 
def onsiteEnergy(site, phase, gaussians):
    (x, y) = site.pos
    return (
            np.cos(superconductingPhase(site, phase)) * superconductingRegion(site) * superconductingGap * tau_0x
            + np.sin(superconductingPhase(site, phase)) * superconductingRegion(site) * superconductingGap * tau_0y
            + scatteringRegion(site) * spinScattering * tau_z0
            + getScatteringAtPosition(site.pos, gaussians) * spinScattering * tau_z0
            )

# gap the upper channel
def scatteringRegion(site):
    (x, y) = site.pos
    if y > a(0, 10).pos[1]:
        return 1
    else:
        return 0

def superconductingPhase(site, phase):
    tag = site.tag
    (x, y) = site.pos
    if x <= a(L/2, tag[1]).pos[0]:
        return 0
    else:
        return phase

def superconductingRegion(site):
    tag = site.tag
    (x, y) = site.pos
    if x > a(L/2, tag[1]).pos[0] - SCGap/2 and x < a(L/2, tag[1]).pos[0] + SCGap/2:
        return 0
    else:
        return 1
     
def spinOrbitHopping(site1, site2, phase, gaussians):
    return 1j * spinOrbit / (3 * np.sqrt(3)) * haldanePhases(site1, site2) * tau_zz

def haldanePhases(site1, site2):
    delta = site1.tag - site2.tag
    if site1.family == a:
        if delta == (-1, 1) or delta == (1, 0) or delta == (0, -1):
            return -1
        else:
            return 1
    else:
        if delta == (-1, 1) or delta == (1, 0) or delta == (0, -1):
            return 1
        else:
            return -1
        
def getScatteringAtPosition(position, gaussians):
    scattering = 0
    for gaussian in gaussians:
        scattering += gaussian(position)
    return scattering
        
def calcAndPlotTotalWaveFunctionAroundEnergy(sys, params):
    ev, evecs = calcWaveFunctions(sys, params, n = 1)#0.089)
    print(ev)
    print(ev/0.2)
    for i in range(1):
        #vec = np.sqrt(np.abs(evecs[0::8, i])**2 + np.abs(evecs[1::8, i])**2 + np.abs(evecs[2::8, i])**2 + np.abs(evecs[3::8, i])**2)
        vec = np.sqrt(np.abs(evecs[0::4, i])**2 + np.abs(evecs[1::4, i])**2)
        #vec = np.sqrt(np.abs(evecs[0::2, i])**2 + np.abs(evecs[1::2, i])**2)
        plotWaveFunction(sys, vec)

def calcWaveFunctions(sys, params, n = 10):
    ham_mat = sys.hamiltonian_submatrix(params = params, sparse=True)
    return sla.eigsh(ham_mat, k = n, which = "LM", sigma = 0, return_eigenvectors = True)
        
def plotWaveFunction(sys, vec):
    plt.figure()
    axes = plt.axes()
    kwant.plotter.map(sys, np.abs(vec)**2, colorbar=True, cmap='jet', ax=axes, oversampling=1,)# file = 'a.pdf')
    #plt.savefig('a.pdf')
   
def calcLowestEigenEnergy(sys, params):
    ev, evecs = calcWaveFunctions(sys, params, n = 1)
    return np.abs(ev[0])
    
def calcLowestEigenEnergies(sys, params):
    ev, evecs = calcWaveFunctions(sys, params, n = 4)
    return ev

def calcEigenEnergies(scattererIndices):
    scattererIndices = splitIntoChunks(scattererIndices)
    exportData = []
    t0 = time()
    for i, scattererIndicesSingleConfiguration in enumerate(scattererIndices):
        eigenvalue = calcLowestEigenEnergyForScattering(scattererIndicesSingleConfiguration)
        exportData.append(eigenvalue)
        if i % 10 == 0:
            t1 = time()
            print('Time for 10 configurations: {:6.2f} sec.; rank: {:>2}; name: {:>7}; Configuration: {:>6} of {:>6}'\
                  .format(t1 - t0, rank, name, i, len(scattererIndices)))
            t0 = t1
    data = comm.gather(exportData, root = 0)
    if rank == 0:
        return flattenExportData(data)
    
def splitIntoChunks(scattererIndices):
    if rank == 0:
        chunks = np.array_split(scattererIndices, comm.size)
    else:
        chunks = None
    return comm.scatter(chunks, root = 0)

def flattenExportData(data):
    exportData = []
    if rank == 0:
        for process in range(comm.size):
            for dat in data[process]:
                exportData.append(dat)    
        return exportData

def plotGaussians(gaussians):
    x = np.linspace(0, 40, 100)
    y = np.linspace(0, 20, 100)
    [X, Y] = np.meshgrid(x, y)
    z = np.zeros((100, 100))
    
    for i, x_val in enumerate(x):
        for j, y_val in enumerate(y):
            z[j, i] = getScatteringAtPosition((x_val, y_val), gaussians)
    
    plt.contour(X, Y, z)
    
def loadScattererIndices():    
    scattererRelativePositions = loadScattererRelativePositions()
    scattererIndices = np.zeros(scattererRelativePositions.shape, dtype = np.int8)
    for i, configuration in enumerate(scattererRelativePositions):
        for j, relativePosition in enumerate(scattererRelativePositions[i]):
            scattererIndices[i, j] = convertRelativePositionToIndex(relativePosition)
    return scattererIndices

def loadScattererRelativePositions():
    scattererRelativePositionsFile = np.load('scattererRelativePositions.npz')
    scattererRelativePositions = scattererRelativePositionsFile['relativePositions']
    scattererRelativePositionsFile.close()
    return scattererRelativePositions

def convertRelativePositionToIndex(position):
    position = np.multiply(position, np.array([L, W]))
    position = np.rint(position)
    return position.astype(int)

def calcLowestEigenEnergyForScattering(scattererIndicesSingleConfiguration):
    gaussians = buildGaussians(scattererIndicesSingleConfiguration)
    params = dict(phase = np.pi, gaussians = gaussians)
    return calcLowestEigenEnergy(sysF, params)

def main():
    global sys, sysF
    sys = makeSystem()
    sysF = sys.finalized()

    scattererIndices = loadScattererIndices()
   
    if rank == 0:
        t0 = time()
    data = calcEigenEnergies(scattererIndices)
    if rank == 0:
        np.savetxt('energies.txt', data)
        duration = time() - t0
        print('toal time ellapsed: {:.2f}'.format(duration))

if __name__ == '__main__':
    main()
