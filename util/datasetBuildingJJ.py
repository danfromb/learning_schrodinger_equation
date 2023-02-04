#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

numberOfConfigurations = 40000
batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

def buildDataset(batchSize):
    global batch_size
    batch_size = batchSize
    x, y = loadData()
    dataset = convertNumpyToTFDataset((x, y))
    splitDatasets = splitDataset(dataset)
    trainDataset, validationDataset = optimizeDatasets(splitDatasets)
    return trainDataset, validationDataset

def loadData():
    x = loadX()
    y = loadY()
    print(x.shape)
    print(y.shape)
    return x, y

def loadX():
    xFile = np.load('dataJJ/scattererRelativePositions.npz')
    x = xFile['relativePositions']
    xFile.close()
    x = x[:numberOfConfigurations]
    x = np.reshape(x, (x.shape[0], 20))
    return x

def loadY():
    y = np.loadtxt('dataJJ/energies.txt', unpack = True)
    y = y.astype('float32') / np.max(y)
    y = y[:numberOfConfigurations]
    return y

def convertNumpyToTFDataset(numpyData):
    dataset = tf.data.Dataset.from_tensor_slices(numpyData)
    dataset = dataset.shuffle(numberOfConfigurations, reshuffle_each_iteration = False)
    return dataset

def splitDataset(dataset):
    validationDatasetSize = int(0.2 * numberOfConfigurations)
    trainDataset = dataset.skip(validationDatasetSize)
    validationDataset = dataset.take(validationDatasetSize)
    return trainDataset, validationDataset

def optimizeDatasets(datasets):
    trainDataset, validationDataset = datasets
    trainDataset = optimizeDataset(trainDataset)
    validationDataset = optimizeDataset(validationDataset)
    return trainDataset, validationDataset

def optimizeDataset(dataset):
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size = 1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size = AUTOTUNE)
    return dataset
