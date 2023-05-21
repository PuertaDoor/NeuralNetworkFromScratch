import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mltools import *
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from copy import deepcopy

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass
    
def my_log(yhat):
    '''
    Fonction qui permet d'éviter les trop grandes valeurs pour np.log
    '''
    yhat = np.where(yhat > 1e2, 1e2, yhat) #avoid overflow
    yhat = np.where(yhat < -1e2, -1e2, yhat)
    return np.maximum((-100 * np.ones(yhat.shape)), np.log(yhat + 1e-10))

class MSE(Loss):
    def forward(self, y, yhat):
        return np.linalg.norm((y - yhat).astype(np.float64), axis=1)**2 # axis=1 to have batch sized vector
    
    def backward(self, y, yhat):
        return -2 * (y - yhat) # taille batch*d, gradient de la MSE en fonction des yhat.
    
class CE(Loss):
    def forward(self, y, yhat): # y and yhat are matrix
        return 1 - np.sum(y*yhat, axis=1)
    
    def backward(self, y, yhat):
        return yhat - y
    
class BCE(Loss):
    def forward(self, y, yhat):
        yhat = np.where(yhat > 1e6, 1e6, yhat) #avoid overflow
        yhat = np.where(yhat < -1e6, -1e6, yhat)
        return -np.mean(y*my_log(yhat) + (1-y)*my_log(1-yhat), axis=1)
    
    def backward(self, y, yhat):
        return ((1 - y) / (1 - yhat + 1e-10)) - (y / yhat + 1e-10)
    
class KLDivergence(Loss):
    def forward(self, y, yhat):
        return np.sum(yhat*my_log(yhat/(y+1e-10)), axis=1)
    
    def backward(self, y, yhat):
        return np.log((yhat/(y + 1e-10)) + 1e-10) + 1
    
class CElogSoftmax(Loss):
    def forward(self, y, yhat):
        yhat = np.where(yhat < -1e2, -1e2, yhat) # avoid overflowing
        yhat = np.where(yhat > 1e2, 1e2, yhat)
        return np.log(np.sum(np.exp(yhat), axis=1) + 1e-10) - np.sum(y * yhat, axis=1)
    
    def backward(self, y, yhat):
        yhat = np.where(yhat < -1e2, -1e2, yhat) # avoid overflowing
        yhat = np.where(yhat > 1e2, 1e2, yhat)
        exp = np.exp(yhat)
        return exp / (np.sum(exp, axis=1).reshape((-1, 1)) + 1e-10) - y