#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

inputShape = (20,)


def buildModel():
#    mirrored_strategy = tf.distribute.MirroredStrategy()
#    with mirrored_strategy.scope():
#        model = tf.keras.Sequential()
#        addLayers(model)
    model = tf.keras.Sequential()
    addLayers(model)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
    model.compile(optimizer = optimizer, loss = 'mse')
    return model

def addLayers(model):
    #addConvolutionalLayers(model)
    #model.add(tf.keras.layers.Flatten())
    addFullyConnectedNet(model)
    
def addConvolutionalLayers(model):
    reducingLayers, nonReducingLayers = buildLayers()
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape = inputShape))
    for i in range(7):
        model.add(reducingLayers[i])
        model.add(nonReducingLayers[i][0])
        model.add(nonReducingLayers[i][1])
    model.add(tf.keras.layers.Flatten())
    
def addFullyConnectedNet(model):
    layerDepth = 20
    model.add(tf.keras.layers.Dense(layerDepth, activation = 'relu', kernel_initializer = 'he_uniform', input_shape = inputShape))
    for i in range(10):
        model.add(tf.keras.layers.Dense(layerDepth, activation = 'relu', kernel_initializer = 'he_uniform'))
        #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, name = 'test'))
    
def buildLayers():
    reducingLayers = np.empty((7), dtype = object)
    nonReducingLayers = np.empty((7, 2), dtype = object)
    
    for i in range(7):
        reducingLayers[i] = tf.keras.layers.Conv2D(64, (3, 3), strides = (2, 2), padding = 'same', activation = 'relu', 
                      kernel_initializer = 'he_uniform')
        for j in range(2):
            nonReducingLayers[i][j] = tf.keras.layers.Conv2D(16, (4, 4), padding = 'same', activation = 'relu', 
                             kernel_initializer = 'he_uniform')
    return reducingLayers, nonReducingLayers