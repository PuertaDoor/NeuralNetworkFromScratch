import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mltools import *
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from copy import deepcopy

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None
        self._bias = None
        self._gradient_bias = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters = self._parameters - gradient_step*self._gradient
        self._bias = self._bias - gradient_step*self._gradient_bias

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass
    
'''
MODULE LINEAIRE
'''

class Linear(Module):
    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output
        self._parameters = 2 * (np.random.rand(self.input, self.output) - 0.5) # (batch*input @ input*output) -> (batch*output) pour forward pass
        self._gradient = np.zeros(self._parameters.shape)
        self._bias = 2 * (np.random.rand(1, self.output) - 0.5)
        self._gradient_bias = np.zeros((1, self.output))
    
    def zero_grad(self): # reinitialise à 0 le gradient
        self._gradient = np.zeros(self._gradient.shape)
        self._gradient_bias = np.zeros(self._gradient_bias.shape)
    
    def forward(self, X):
        res = X @ self._parameters + self._bias
        res = np.where(res > 1e6, 1e6, res)
        res = np.where(res < -1e6, -1e6, res)
        return res
    
    def backward_update_gradient(self, X, delta):
        self._gradient = self._gradient + X.T @ delta
        self._gradient_bias = self._gradient_bias + np.sum(delta, axis=0)
    
    def backward_delta(self, X, delta):
        delta = np.where(delta > 1e6, 1e6, delta)
        delta = np.where(delta < -1e6, -1e6, delta)
        self._parameters = np.where(self._parameters > 1e6, 1e6, self._parameters)
        self._parameters = np.where(self._parameters < -1e6, -1e6, self._parameters)
        return delta @ self._parameters.T

'''
MODULE DE CONVOLUTION 1D
'''

class Conv1D(Module):

    def __init__(self, k_size, chan_in, chan_out, stride=1):
        super().__init__()
        self._k_size = k_size
        self._chan_in = chan_in
        self._chan_out = chan_out
        self._stride = stride
        self._parameters = 2 * (np.random.rand(k_size, chan_in, chan_out) - 0.5)
        self._gradient = np.zeros(self._parameters.shape)
        self._bias = 2 * (np.random.rand(chan_out) - 0.5)
        self._gradient_bias = np.zeros(chan_out)

    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)
        self._gradient_bias = np.zeros(self._gradient_bias.shape)
        

    def forward(self, X):
        batch_size, length, chan_in = X.shape

        d_out = (length - self._k_size) // self._stride + 1

        X_view = np.lib.stride_tricks.sliding_window_view(X, (1, self._k_size, self._chan_in))[::1, ::self._stride, ::1]
        X_view = X_view.reshape(batch_size, d_out, self._chan_in, self._k_size)

        return np.einsum("bock, kcd -> bod", X_view, self._parameters) + self._bias


    def backward_update_gradient(self, X, delta):
        batch_size, length, chan_in = X.shape

        d_out = (length - self._k_size) // self._stride + 1

        X_view = np.lib.stride_tricks.sliding_window_view(X, (1, self._k_size, self._chan_in))[::1, :: self._stride, ::1]
        X_view = X_view.reshape(batch_size, d_out, self._chan_in, self._k_size)

        self._gradient = self._gradient + (np.einsum("bock, bod -> kcd", X_view, delta) / batch_size)
        self._gradient_bias = self._gradient_bias + np.sum(delta, axis=(0, 1)) / batch_size

    def backward_delta(self, X, delta):
        _, length, chan_in = X.shape

        d_out = (length - self._k_size) // self._stride + 1
        
        d_in = np.einsum("bod, kcd -> kboc", delta, self._parameters)
        
        out = np.zeros(X.shape)

        for i in range(self._k_size):
            out[:, i:i + d_out * self._stride : self._stride, :] = out[:, i:i + d_out * self._stride : self._stride, :] + d_in[i]

        return out
    
# MODULE DE RÉGRESSION LINÉAIRE

class LinearRegression:
    
    def __init__(self, nn, loss, epochs, gradient_step):
        self.nn = nn
        self.loss = loss
        self._epochs = epochs
        self._gradient_step = gradient_step
        
    def score(self, y, yhat):
        return r2_score(y, yhat)
    
    def fit(self, X_train, y_train, X_valid, y_valid):
        
        X_train = X_train.astype(float)
        y_train = y_train.astype(float)
        X_valid = X_valid.astype(float)
        y_valid = y_valid.astype(float)
        losses_train = []
        scores_train = []
        losses_valid = []
        scores_valid = []

        for i in tqdm(range(self._epochs)):
            # Forward pass
            yhat_train = self.nn.forward(X_train) # yhat = z^k = M^k(z^k-1, W^k)
            yhat_valid = self.nn.forward(X_valid)
            losses_train.append(self.loss.forward(y_train, yhat_train).mean())
            scores_train.append(self.score(y_train, yhat_train))
            losses_valid.append(self.loss.forward(y_valid, yhat_valid).mean())
            scores_valid.append(self.score(y_valid, yhat_valid))

            # Backward pass
            last_delta = self.loss.backward(y_train, yhat_train) # delta de la dernière couche
            self.nn.backward_update_gradient(X_train, last_delta)
            self.nn.update_parameters(gradient_step=self._gradient_step)
            self.nn.zero_grad()
            
        return losses_train, scores_train, losses_valid, scores_valid
    
    def predict(self, X):
        return self.nn.forward(X)