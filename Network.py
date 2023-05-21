import numpy as np
from Module import *
from Loss import *
from Activation import *

class Sequentiel:
    def __init__(self, modules, loss):
        self._modules = modules
        self._loss = loss
        self._outputs = []
        
    def forward(self, X):
        z = deepcopy(X)
        self._outputs.append(z)
        for mod in range(len(self._modules)):
            z = self._modules[mod].forward(z)
            self._outputs.append(z)
        
        return z # yhat
    
    def backward(self, X, delta):
        for mod in range(len(self._modules)-1, -1, -1):
            self._modules[mod].backward_update_gradient(self._outputs[mod].astype(float), delta)
            delta = self._modules[mod].backward_delta(self._outputs[mod].astype(float), delta)
            
            
    def update_parameters(self, gradient_step=1e-4):
        for mod in range(len(self._modules)):
            self._modules[mod].update_parameters(gradient_step=gradient_step)
    
    def zero_grad(self):
        for mod in range(len(self._modules)):
            self._modules[mod].zero_grad()

    def predict(self, X):
        return self.forward(X)
    

class Optim:
    def __init__(self, net, loss, eps):
        self._net = net
        self._loss = loss
        self._eps = eps

    def score_onehot(self, y, yhat):
        yhat = np.argmax(yhat, 1)
        y = np.argmax(y, 1)
        
        return np.where(yhat == y, 1, 0).sum() / y.size

    def score_bin(self, y, yhat):
        yhat = np.where(yhat>=0.5, 1, 0)
        return np.where(yhat - y == 0, 1, 0).sum() / y.size

        
    def step(self, batch_xtrain, batch_ytrain, batch_xvalid, batch_yvalid):
        # Forward pass
        output = self._net.forward(batch_xtrain)
        loss_train = self._loss.forward(batch_ytrain, output)
        valid_out = self._net.forward(batch_xvalid)
        loss_valid = self._loss.forward(batch_yvalid, valid_out)
        if output.shape[1] > 1:
            score_train = self.score_onehot(batch_ytrain, output)
            score_valid = self.score_onehot(batch_yvalid, valid_out)
        else:
            score_train = self.score_bin(batch_ytrain, output)
            score_valid = self.score_bin(batch_yvalid, valid_out)
        
        # Backward pass
        self._net.zero_grad()
        last_delta = self._loss.backward(batch_ytrain, output)
        self._net.backward(batch_xtrain, last_delta)
        self._net.update_parameters(gradient_step=self._eps)
        
        return loss_train.mean(), loss_valid.mean(), score_train, score_valid # for graphics