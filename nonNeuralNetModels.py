#!/usr/bin/env python3

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
import sklearn
import numpy as np
import scipy
from time import time
from joblib import dump

def main():
    t0 = time()
    #tuneHyperparameters()
    #trainKRR()
    #trainKRRPolynomial()
    #trainRF()
    print('total time: {:.2f}'.format(time() - t0))

def loadData():
    x = loadX()
    y = loadY()
    return x, y

def loadX():
    xFile = np.load('dataJJ/scattererRelativePositions.npz')
    x = xFile['relativePositions']
    xFile.close()
    x = x[:numberOfConfigurations]
    x = x.astype('float32')
    x = np.reshape(x, (x.shape[0], 20))
    return x

def loadY():
    y = np.loadtxt('dataJJ/energies.txt', unpack = True)
    y = y[:numberOfConfigurations]
    print(np.max(y))
    y = y.astype('float32') / np.max(y)
    return y

def tuneHyperparameters():
    reg = RandomForestRegressor(max_features = 'auto', n_jobs = -1)
    parameters = {'n_estimators': scipy.stats.randint(low = 500, high = 1500), 
                  'min_samples_leaf': loguniform(1e-4, 1e-2), 
                  'max_features': scipy.stats.randint(low = 15, high = 21)}
    parameters = {'n_estimators': scipy.stats.randint(low = 1000, high = 1300), 
                  'min_samples_leaf': scipy.stats.uniform(loc = 1e-4, scale = 1e-2)}
    cv = RandomizedSearchCV(reg, parameters, scoring = 'neg_mean_squared_error', n_iter = 100, n_jobs = -1)
    cv.fit(xTrain, yTrain)
    print(-cv.score(xTrain, yTrain))
    print(cv.best_params_)

def trainKRR():
    reg = KernelRidge(kernel = 'rbf', alpha = 0.011910616759208445, gamma = 0.22088851834074053)
    reg.fit(xTrain, yTrain)
    yPred = reg.predict(xTest)
    mse = mean_squared_error(yTest, yPred)
    print(mse)
    
def trainKRRPolynomial():
    reg = KernelRidge(kernel = 'polynomial', alpha = 0.08227812354903298, gamma = 0.19381421487331665,
                      degree = 3.590426600986404, coef0 = 3.8570212657395495)
    reg.fit(xTrain, yTrain)
    yPred = reg.predict(xTest)
    mse = mean_squared_error(yTest, yPred)
    print(mse)
    
def trainRF():
    reg = RandomForestRegressor(max_features = 'auto', n_estimators = 1247, min_samples_leaf = 0.000218500812165272,
                                n_jobs = -1)
    reg.fit(xTrain, yTrain)
    #yPred = reg.predict(xTest)
    #mse = mean_squared_error(yTest, yPred)
    #np.savez('trainedModelsJJ/yVals', yTest = yTest, yPred = yPred, xTest = xTest)
    #dump(reg, 'trainedModelsJJ/reg')
    #print(mse)
     
numberOfConfigurations = 40000
x, y = loadData()
#xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
xTrain = x
yTrain = y

if __name__ == '__main__':
    main()