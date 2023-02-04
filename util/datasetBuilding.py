#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

numberOfImages = 10000
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
    return x, y

def loadX():
    x = np.zeros((numberOfImages), dtype = object)
    for i in range(numberOfImages):
        x[i] = 'data/potentials/pot' + str(i) + '.jpg'
    return x

def loadY():
    _, y = np.loadtxt('data/energies.txt', unpack = True)
    y = y.astype('float32') / np.max(y)
    y = y[:numberOfImages]
    return y

def convertNumpyToTFDataset(numpyData):
    dataset = tf.data.Dataset.from_tensor_slices(numpyData)
    dataset = dataset.map(processPath)
    dataset = dataset.shuffle(numberOfImages, reshuffle_each_iteration = False)
    return dataset

def processPath(path, yVal):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    return image, yVal

def splitDataset(dataset):
    validationDatasetSize = int(0.2 * numberOfImages)
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
