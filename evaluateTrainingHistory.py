#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

batchSizes = [16, 32, 64, 128, 256, 512, 1024]

def main():
    losses, validationLosses = loadBatchHistory()
    
#    print(losses)
#    print(validationLosses)
    
    plotLossesVSEpoch(losses)
    plotLossesVSEpoch(validationLosses)
    
    finalValidationLosses = validationLosses[:, -1]
    minFinalValidationLosses = np.min(finalValidationLosses)
    
    
    print(finalValidationLosses)
    print(finalValidationLosses / minFinalValidationLosses)

def loadBatchHistory():
    losses = []
    validationLosses = []
    for batchSize in batchSizes:
        historyPath = 'trainedModels/batchSizes/' + str(batchSize) + '/'
        loss, validationLoss = loadHistory(historyPath)
        losses.append(loss)
        validationLosses.append(validationLoss)
    losses = np.array(losses)
    validationLosses = np.array(validationLosses)
    return losses, validationLosses

def loadHistory(path):
    history = np.load(path + 'trainingHistory.npz')
    loss = history['loss']
    validationLoss = history['validationLoss']
    return loss, validationLoss

def plotLossesVSEpoch(losses):
    fig = plt.figure(figsize = (8, 8))
    plt.rcParams.update({'font.size': 15})
    for i in range(len(batchSizes)):
        plt.plot(losses[i, -5:], 'o', markersize = 15)
        plt.legend(batchSizes, fontsize = 15)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

if __name__ == '__main__':
    main()
