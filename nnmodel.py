#author: Rewa Singh Chauhan
#date: 6th April, 2020
#modules implemented: 

#import libraries
import numpy as np 
import matplotlib.pyplot as plt

#helper functions 
#implementation of relu
def relu(z):
    return np.maximum(0,z) #discrads the negative values from the fn
#implementation of loss function
def sigmoid(z):
    return 1/(1+np.exp(-1*z))

#derivatives of helper fns
def relugradient(X): # Xis the list of neurons
    X[X>=0]=1
    X[X<0]=0
    return X
def siggradient(X):
    return sigmoid(X)*(1-sigmoid(X))

#implementattion of neural networks
class Model:
    def __init__(self):     #python constructor; this is equivalent to self
        self.layers = []    #the neurons in 1 layer
        self.W = {}
        self.b = {}
        self.A = {}         #A: activation; Z: loss
        self.Z = {}           
        self.dW = {}
        self.db = {}
        self.dA = {}         
        self.dZ = {}
        self.cost = 0      # cost = (y predicted) - error(ie the rms)
        self.m = 0         # no of parameteres for linear regressions
        self.lam = 0       # slope of lin reg
        self.alpha = 0     # rate of learning
        self.L = 0         # no of neurons in 1 layer
        self.iterations = 0         #epochs
        self.cost_history = []      
        self.accuracy_history = []            
        self.alpha_history = []
        return
    def Add_Layer(self,neuronlist): #initializing layers
        self.layers = neuronlist
        self.L = len(self.layers)
        return 
    def init_params(self):        #initializing the connecting lines ie wt and bias
        for i in range(1,self.L+1):
            self.W[str(i)]=np.random.randn(self.layers[i],self.layers[i-1])*np.sqrt(2/self.layers[i-1])
            self.b[str(i)]=np.zeros((self.layers[i],1))
        return

