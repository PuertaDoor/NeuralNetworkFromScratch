from Module import *

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mltools import *
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from copy import deepcopy

class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.tanh(X.astype(np.float64))
    
    def backward_delta(self, X, delta):
        return (1 - self.forward(X)**2) * delta # d'après la dérivée en fonction de X de tanh(X), delta joue le rôle de constante.

    def update_parameters(self, gradient_step=1e-3):
        pass # nous n'avons pas de paramètre dans ce modèle
    

def sigmoide(X):
    X = np.where(X > 1e2, 1e2, X) #avoid overflow
    X = np.where(X < -1e2, -1e2, X)
    return 1 / (1 + np.exp(-X)) # sigmoide n'existe pas dans NumPy, nous l'implémentons ici.

class Sigmoide(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        return sigmoide(X)
    
    def backward_delta(self, X, delta):
        tmp = sigmoide(X)
        return delta * tmp * (1-tmp) # d'après la dérivée en fonction de X de sigmoide(X) (avec une constante lambda de 1), delta joue le rôle de constante.
    
    def update_parameters(self, gradient_step=1e-3):
        pass # nous n'avons pas de paramètre dans ce modèle

class Softmax(Module):
    def __init__(self):
        super().__init__()
    

    def forward(self, Z):
        Z = np.where(Z > 1e2, 1e2, Z) #avoid overflow
        Z = np.where(Z < -1e2, -1e2, Z)
        e = np.exp(Z)
        return e / np.sum(e, axis=1).reshape(-1,1)
    
    def backward_delta(self, Z, delta):
        forw = self.forward(Z)
        return delta * (forw * (1-forw))
    
    def update_parameters(self, gradient_step=1e-3):
        pass # nous n'avons pas de paramètre dans ce modèle

class LogSoftmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X_shifted = X - np.max(X, axis=-1, keepdims=True)
        return X_shifted - np.log(np.sum(np.exp(X_shifted), axis=-1, keepdims=True))
    
    def backward_delta(self, Z, delta):
        e = np.exp(Z)
        return delta * (1 - e / e.sum(axis=1, keepdims=True))
    
    def update_parameters(self, gradient_step=1e-3):
        pass # nous n'avons pas de paramètre dans ce modèle

class MaxPool1D(Module):
    def __init__(self, k_size=3, stride=1):
        self._parameters = None
        self._gradient = None
        self._k_size = k_size
        self._stride = stride
    
    def forward(self, X):
        batch, length, chan_in = X.shape
        
        d_out = ((length - self._k_size) // self._stride) + 1
        out = np.zeros((batch, d_out, chan_in))
        
        for i in range(0, d_out, self._stride):
            out[:,i,:] = np.max(X[:,i:i+self._k_size,:], axis=1)

        return out
        

    def backward_delta(self, X, delta):
        batch, length, chan_in = X.shape
        
        d_out = ((length - self._k_size) // self._stride) + 1
        out = np.zeros(X.shape)
        
        for i in range(0, d_out, self._stride):
            indexes_argmax = np.argmax(X[:,i:i+self._k_size,:], axis=1) + i
            out[np.repeat(range(batch),chan_in),indexes_argmax.flatten(),list(range(chan_in))*batch]=delta[:,i,:].reshape(-1)

        return out
    
    def update_parameters(self, gradient_step=1e-3):
        pass # nous n'avons pas de paramètre dans ce modèle
    
class ReLU(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        return np.maximum(np.zeros(X.shape), X)
    
    def backward_delta(self, X, delta):
        return delta * (X > 0).astype(np.float64)
    
    def update_parameters(self, gradient_step=1e-3):
        pass # nous n'avons pas de paramètre dans ce modèle
    
class Flatten(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        batch = X.shape[0]
        return X.reshape(batch, -1)

    def backward_delta(self, X, delta):
        return delta.reshape(X.shape)
    
    def update_parameters(self, gradient_step=1e-3):
        pass # nous n'avons pas de paramètre dans ce modèle