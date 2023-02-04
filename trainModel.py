#!/usr/bin/env python3

import numpy as np
from time import time
import util.datasetBuildingJJ
import util.modelGenerationJJ

import tensorflow as tf


def main():
    t0 = time()
    
#    model, _ = trainModel(1024)
#    
#    tf.keras.utils.plot_model(model, to_file = 'model.png', show_shapes = True)
    
#    for batchSize in [128, 256, 512, 1024, 2048]:
#        print(batchSize)
#        trainModel(batchSize)
    
#    trainDataset, validationDataset = util.datasetBuildingJJ.buildDataset(2048)
#    pred = model.predict(trainDataset)
#    print(pred.shape)
    
    trainAndSaveModel(2048)
    
    print('time: {:.2f}'.format(time() - t0))
    
def trainAndSaveModel(batchSize):
    model, losses = trainModel(batchSize)
    savePath = 'trainedModelsJJ/NN/'
    model.save(savePath + 'model')
    saveHistory(losses, savePath)
    
def trainModel(batchSize):
    trainDataset, validationDataset = util.datasetBuildingJJ.buildDataset(batchSize)
    model = util.modelGenerationJJ.buildModel()
    history = model.fit(trainDataset, validation_data = validationDataset, epochs = 1000)
    loss, validationLoss = getLossHistory(history)
#    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
#    model.compile(optimizer = optimizer, loss = 'mse')
#    history = model.fit(trainDataset, validation_data = validationDataset, epochs = 100)
#    loss2, validationLoss2 = getLossHistory(history)
#    loss = loss + loss2
#    validationLoss = validationLoss + validationLoss2
    return model, [loss, validationLoss]

def getLossHistory(history):
    loss = history.history['loss']
    validationLoss = history.history['val_loss']
    return loss, validationLoss

def saveHistory(losses, savePath):
    loss = losses[0]
    validationLoss = losses[1]
    np.savez(savePath + 'trainingHistory', loss = loss, validationLoss = validationLoss)
    
if __name__ == '__main__':
    main()
    