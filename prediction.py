#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import util.datasetBuilding
import util.modelGeneration


model = tf.keras.models.load_model('trainedModel')

image = tf.io.read_file('data/potentials/pot0.jpg')
image = tf.image.decode_jpeg(image, channels=1)

x, y = util.datasetBuilding.loadData()
test = tf.data.Dataset.from_tensor_slices((x[:1], y[:1]))
test = test.map(util.datasetBuilding.processPath)
test = util.datasetBuilding.optimizeDataset(test)
    
yPrediction = model.predict(test)
print('Predicted: {}'.format(yPrediction))
print('Actual: {}'.format(y[0]))
print('Difference: {}'.format(y[0] - yPrediction))