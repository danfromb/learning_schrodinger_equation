# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import random
from time import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
name = MPI.Get_processor_name().split('.')[0]

size = 256
numberOfPotentials = 200000

pot = np.zeros((size, size), dtype=np.uint8)

def generateAndSaveRandomImages():
    if rank == 0:
        potentialNumbers = range(numberOfPotentials)
        chunks = np.array_split(potentialNumbers, comm.size)
    else:
        potentialNumbers = None
        chunks = None
    potentialNumbers = comm.scatter(chunks, root = 0)
    numberOfPotentialsCurrentProcess = len(potentialNumbers)
    t0 = time()
    i = 0
    for potentialNumber in potentialNumbers:
        generateAndSaveRandomImage('potentials/pot' + str(potentialNumber) + '.jpg')
        if i % 20 == 0:
            t1 = time()
            print('Time for 200 images: {:6.2f} sec.; rank: {:>2}; name: {:>7}; Image: {:>6} of {:>6}'.format(
                    t1 - t0, rank, name, i, numberOfPotentialsCurrentProcess))
            t0 = t1
        i = i + 1

def generateAndSaveRandomImage(path):
    image = generateRandomImage()
    image.save(path)

def generateRandomImage():
    params = generateRandomPotentialParameters()
    return generateImage(params)

def generateRandomPotentialParameters():
    cx = random.randrange(size / 2 - 30, size / 2 + 30)
    cy = random.randrange(size / 2 - 30, size / 2 + 30)
    kx = random.uniform(0, 1)
    ky = random.uniform(0, 1)
    return [cx, cy, kx, ky]

def generateImage(params):
    generatePotentialArray(params)
    return Image.fromarray(pot, 'L')

def generatePotentialArray(params):
    for i in range(size):
        for j in range(size):
            V = potential([i, j] + params)
            if V <= 255:
                pot[i, j] = V
            else:
                pot[i, j] = 255

def potential(params):
    x, y, x0, y0, kx, ky = params
    return 0.5 * (kx * (x - x0)**2 + ky * (y - y0)**2)

if __name__ == '__main__':
    if rank == 0:
        t0 = time()
    generateAndSaveRandomImages()
    if rank == 0:
        duration = time() - t0
        print('toal time ellapsed: {:.2f}'.format(duration))
