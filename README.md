# NeuralNetworkFromScratch
Implementation of Neural Networks in Python, from scratch, with NumPy.

The goal of this project is to reproduct the Lua PyTorch version.

Module descriptions :
- Activation.py -> implements main activation layers, such as ReLU, TanH, Sigmoid, max pooling.
- Loss.py -> contains Cross-entropy loss, MSE loss, BCE loss, KL divergence.
- Module.py -> Linear Module, convolutional module in 1D.
- Network.py -> encapsulates all these modules to connect all layers (Sequentiel class), and does implement the step() function as well.
- preprocessing.py -> just two functions for data encoding.

A notebook, experimentations.ipynb, can be used to see how does implementation works.
