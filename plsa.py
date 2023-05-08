import numpy as np
import math


class PLSA:
    def __init__(self, counts: np.matrix, T: int):
        self.counts = counts
        self.T = T
        self.W = counts.shape[0] 
        self.D = counts.shape[1]
        self.nd = np.sum(self.counts, axis=0)
        self.initialize_params()
    
    def initialize_params(self):
        self.Phi = np.array([[1./self.W]*self.W]*self.T).T
        self.Theta = np.array([[1./self.T]*self.T]*self.D).T
        
    def step(self):
        #Z = np.dot(self.Phi, self.Theta) #(W, D)
        Z = np.zeros((self.W, self.D))
        for d in range(self.D):
            for t in range(self.T):
                for w in range(self.W):
                    if self.counts[w][d] > 0:
                        Z[w][d] += self.Phi[w][t] * self.Theta[t][d]
        nTheta = np.zeros((self.T, self.D))
        nPhi = np.zeros((self.W, self.T))
        nwt = np.zeros((self.W, self.T))
        nt = np.zeros((self.T))
        for d in range(self.D):
            #s = 0
            for t in range(self.T):           
                for w in range(self.W):
                    if abs(Z[w][d]) > 1e-9:
                        #s += (self.counts[w][d] * self.Phi[w][t] * self.Theta[t][d]) / (Z[w][d] * self.nd[d])
                        nTheta[t][d] += (self.counts[w][d] * self.Phi[w][t] * self.Theta[t][d]) / (Z[w][d] * self.nd[d])
                        nwt[w][t] += (self.counts[w][d] * self.Phi[w][t] * self.Theta[t][d]) / Z[w][d]
                        nt[t] +=  (self.counts[w][d] * self.Phi[w][t] * self.Theta[t][d]) / Z[w][d]
            #print(s)
        for t in range(self.T):
            #s = 0
            for w in range(self.W):
                if abs(nt[t]) > 1e-9:
                    #s += nwt[w][t] / nt[t]
                    nPhi[w][t] = nwt[w][t] / nt[t]
            #print(s)
        self.Theta = nTheta
        self.Phi = nPhi

    def perplexity(self) -> float:
        n = np.sum(self.nd)
        p = 0
        for d in range(self.D):
            for w in range(self.W):
                p += self.counts[w][d] * math.log(np.sum([self.Phi[w][t] * self.Theta[t][d] for t in range(self.T)]))
        return math.exp(-p / n)
