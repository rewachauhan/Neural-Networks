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
            #we hav to use str because W is a dictionary,therefore the data type of key value must be a string
            self.W[str(i)]=np.random.randn(self.layers[i],self.layers[i-1])*np.sqrt(2/self.layers[i-1])#uniform distribution initialization; division by sqrt because the wt value must stay in between 0 &1 
                                                                                                     #no of units in current layer and no of units in previous layer 
            self.b[str(i)]=np.zeros((self.layers[i],1)) # 1 is the order matrix(1d)
        return
    def feed_forward(self,X):
        self.A["0"]=X 
        for i in range(1,self.L+1):
            self.Z[str(i)]=np.dot(self.W[str(i)],self.A[str(i-1)])+self.b[str(i)] #y=wx+b
            #y is reprsented by Z; np.dot is used to find the dot product of w and x ;
            #initial layer of i/p layer is x.
            if(i==self.L):  #ie for final layer
                self.A[str(i)]=sigmoid(self.Z[str(i)])
            else:
                self.A[str(i)]=relu(self.Z[str(i)])
        return
    def compute_cost(self,Y):
        self.cost=-1*np.sum(np.multiply(Y,np.log(self.A[str(self.L)]))+np.multiply(1-Y,np.log(1-self.A[str(self.L)])))/self.m  #cross entropy formula               
        #−(ylog(p)+(1−y)log(1−p))
        if(self.lam!=0):
            reg=self.lam/(2*self.m)         #1/2m (XW-y)(XW-y)^Transope
            for i in range(1, self.L+1):
                reg += np.sum(np.dot(self.W[str(i)], self.W[str(i)].T))
            self.cost += reg
        self.cost_history.append(self.cost)
        return
    def backward_prop(self, Y):
        #we start from output layer
        self.dA[str(self.L)] = -1*((np.divide(Y,self.A[str(self.L)])) - np.divide(1-Y,1-self.A[str(self.L)]))
        self.dZ[str(self.L)] = np.multiply(self.dA[str(self.L)], siggradient(self.Z[str(self.L)]))
        self.dW[str(self.L)] = np.dot(self.dZ[str(self.L)], self.A[str(self.L-1)].T)/ self.m +(self.lam/self.m)*self.W[str(self.L)]
        self.db[str(self.L)] = np.sum(self.dZ[str(self.L)], axis=1, keepdims=True)/self.m
        self.dA[str(self.L-1)] = np.dot(self.W[str(self.L)].T, self.dZ[str(self.L)])

        for i in reversed(range(1, self.L)):
            self.dZ[str(i)] = np.multiply(self.dA[str(i)], relugradient(self.Z[str(i)]))
            self.dW[str(i)] = np.dot(self.dZ[str(i)], self.A[str(i-1)].T)/self.m +(self.lam/self.m)*self.W[str(i)]
            self.db[str(i)] = np.sum(self.dZ[str(i)], axis = 1, keepdims =True)/self.m
            self.dA[str(i-1)] = np.dot(self.W[str(i)].T, self.dZ[str(i)])
        return
    def update_params(self):
        for i in range(1, self.L+1):
            self.W[str(i)] = self.W[str(i)] - self.alpha*self.dW[str(i)]
            self.b[str(i)] = self.b[str(i)] - self.alpha*self.db[str(i)]
        return
    def train(self, X,Y, iterations=10, alpha=0.001, decay=True, decay_iter=5, decay_rate=0.9, stop_decay=100, lam=0):
        self.m = Y.shape[1]
        self.alpha = alpha
        self.lam = lam
        self.iterations = iterations

        self.init_params()
        for i in range(iterations):
            self.feed_forward(X)
            self.compute_cost(Y)
            self.backward_prop(Y)
            self.update_params()
            self.accuracy_history.append(self.evaluate(X,Y, training=True))
            self.alpha_history.append(alpha)
            if decay and stop_decay>0 and i%decay_iter==0:
                self.alpha = decay_rate*self.alpha
                stop_decay -=1
        return
    def predict(self,X, training=False):
        if training==False:
            self.feed_forward(X)
        pred = self.A[str(self.L)] >=0.5
        pred = np.squeeze(pred)
        return pred
    def evaluate(self,X,Y, training=False):
        egs = X.shape[1]
        pred = self.predict(X, training=training)
        pred = pred.reshape(1,egs)
        diff = np.sum(abs(pred-Y))
        acc = (egs - np.sum(diff))/egs
        return acc
    def draw_cost(self):
        plt.plot(range(self.iterations), self.cost_history)
        plt.show()
        return

