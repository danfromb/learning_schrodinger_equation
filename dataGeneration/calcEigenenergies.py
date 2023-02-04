#!/usr/bin/env python3

import sys
sys.path.append("../")
import solver.systemGenerator
import numpy as np
from time import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
name = MPI.Get_processor_name().split('.')[0]

numberOfImages = 200

def main():
    if rank == 0:
        t0 = time()
    data = calcEigenvalues()
    if rank == 0:
        head = 'Image Number; Lowest Eigenenergy'
        np.savetxt('energies.txt', data, header = head)
        duration = time() - t0
        print('toal time ellapsed: {:.2f}'.format(duration))
    
def calcEigenvalues():
    if rank == 0:
        imageNumbers = range(numberOfImages)
        chunks = np.array_split(imageNumbers, comm.size)
    else:
        imageNumbers = None
        chunks = None
    imageNumbers = comm.scatter(chunks, root = 0)
    numberOfImagesCurrentProcess = len(imageNumbers)
    exportData = []
    t0 = time()
    i = 0
    for imageNumber in imageNumbers:
        imagePath = 'potentials/pot' + str(imageNumber) + '.jpg'
        generator = solver.systemGenerator.SystemGenerator()
        system = generator.fromPotentialImage(imagePath)
        eigenvalue = system.eigenenergies[0]
        exportData.append([imageNumber, eigenvalue])
        if imageNumber % 10 == 0:
            t1 = time()
            print('Time for 10 images: {:6.2f} sec.; rank: {:>2}; name: {:>7}; Image: {:>6} of {:>6}'.format(
                    t1 - t0, rank, name, i, numberOfImagesCurrentProcess))
            t0 = t1
        i = i + 1
    data = comm.gather(exportData, root = 0)
    exportData = []
    if rank == 0:
        for Process in range(comm.size):
            for dat in data[Process]:
                exportData.append(dat)    
        return exportData

if __name__ == '__main__':
    main()