#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

plt.rc('text', usetex = True)
plt.rc('font', family='serif')

def loadData():
    file = np.load('trainedModelsJJ/RF/yVals.npz')
    yTest = file['yTest']
    yPred = file['yPred']
    file.close()
    return yTest, yPred
    
yTest, yPred = loadData()

errors = np.subtract(yPred, yTest)
mse = mean_squared_error(yTest, yPred)
print(mse)

plt.figure(figsize = (5, 5))
#axes = plt.gca()
#axes.set_ylim([0, 1])
plt.plot(np.log(yTest), np.log(yPred), 'o', markersize = 1)

plt.figure(figsize = (5, 5))
axes = plt.gca()
#axes.set_ylim([0, 1])
#axes.set_xticks([-0.005, 0.005])
#axes.set_xticklabels([-0.005, 0.005])
plt.hist(errors / np.sqrt(mse), 100, range = (-1, 1))

plt.figure(figsize = (6, 6))
x = np.log(yTest)
y = np.log(yPred)
hist = plt.hist2d(x, y, bins = 100, range = [[-15, -3], [-15, -3]], cmap = plt.get_cmap('hot'))
axes = plt.gca()
axes.set_xlabel('Log True Energy')
axes.set_ylabel('Log Predicted Energy')
axes.text(-13, -5, 'MSE $ = 9.69 \cdot 10^{-8} $', color = 'gray')

insetAxes = axes.inset_axes([0.55, 0.05, 0.4, 0.4])
insetAxes.tick_params('both', direction = 'in', pad = -20, top = True, bottom = False, left = False, color = 'gray')
insetAxes.set_xticks([-1, 1])
insetAxes.xaxis.set_tick_params(labeltop = True, labelbottom = False, colors = 'gray')
insetAxes.xaxis.label.set_color('gray')
insetAxes.set_xlabel(r'Error / $ \sqrt{\textrm{MSE}} $', labelpad = -19)
insetAxes.set_ylim([0, 7e3])
insetAxes.xaxis.set_label_position('top')
insetAxes.set_yticks([])
insetAxes.set_ylabel('Count', labelpad = -19)
insetAxes.yaxis.label.set_color('gray')
insetAxes.set_facecolor((0.05, 0.05, 0.05))
insetAxes.hist(errors / np.sqrt(mse), 100, range = (-1.1, 1.1), color = 'gray')

#plt.savefig('test.png', bbox_inches = 'tight', dpi = 300)