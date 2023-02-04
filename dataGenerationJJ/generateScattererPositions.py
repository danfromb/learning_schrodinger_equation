#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:09:15 2020

@author: daniel
"""

import numpy as np

numberOfScatterers = 10
numberOfConfigurations = 100000

relativePositions = np.random.rand(numberOfConfigurations, numberOfScatterers, 2)
np.savez('scattererRelativePositions', relativePositions = relativePositions)