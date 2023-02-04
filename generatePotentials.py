c# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np

size = 256

potential = np.zeros((size, size), dtype=np.uint8)

# in a.u. units
def potential(x, y, x0, y0, kx, ky):
    pot = 0.5 * (kx * (x - x0)**2 + ky * (y - y0)**2) 

def pixelToPos(i, j):
    
