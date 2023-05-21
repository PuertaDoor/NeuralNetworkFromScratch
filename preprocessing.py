import numpy as np

def normalize(vector): # takes a vector of numbers and return a normalized vector between 0 and 1 (both included)
    return ((vector - np.min(vector))/(np.max(vector)-np.min(vector))).astype(float)

def onehot(vector): # encodes non numeric values
    cpt = 0
    values = np.unique(vector)
    onehot = np.zeros(vector.shape)
    for v in values[1:]:
        cpt += 1
        onehot[vector==v] = float(cpt)
    return onehot/cpt