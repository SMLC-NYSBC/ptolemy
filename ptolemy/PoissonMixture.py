from __future__ import print_function, division

import numpy as np
from scipy.stats import poisson

class PoissonMixture:
    def __init__(self, pi=0.5, max_iters=100, tol=1e-3):
        self.pi = pi
        self.max_iters = max_iters
        self.tol = tol

    @property
    def mask(self):
        if hasattr(self, '_mask'):
            return self._mask
        m = (self.p > 0.5)
        self._mask = m
        return m


    def fit(self, x, verbose=False):
        self.x = x

        max_iters = self.max_iters
        tol = self.tol

        pi = self.pi

        # randomly initialize p
        p = np.random.binomial(1, pi, size=x.shape)
        
        # calculate initial mus
        mu1 = np.sum(x*p/np.sum(p))
        mu2 = np.sum(x*(1-p)/np.sum(1-p))
        
        logp1 = poisson.logpmf(x, mu1) + np.log(pi)
        logp2 = poisson.logpmf(x, mu2) + np.log1p(-pi)
        
        ma = np.maximum(logp1, logp2)
        logp = np.log(np.exp(logp1 - ma) + np.exp(logp2 - ma)) + ma
        logp_next = np.sum(logp)
        logp_cur = -np.inf
        
        p = np.exp(logp1 - ma)/(np.exp(logp1 - ma) + np.exp(logp2 - ma))
        pi = np.mean(p)
        
        if verbose:
            print('iter {} | logp {} | pi {} | mu1 {} | mu2 {}'.format(0, logp_next, pi, mu1, mu2))
        
        # iterate
        for i in range(max_iters):
            if (logp_next - logp_cur) < tol:
                break
                
            logp_cur = logp_next
                
            # estimate mu
            mu1 = np.sum(x*p)/np.sum(p)
            mu2 = np.sum(x*(1-p))/np.sum(1-p)
            
            # estimate p
            logp1 = poisson.logpmf(x, mu1) + np.log(pi)
            logp2 = poisson.logpmf(x, mu2) + np.log1p(-pi)

            ma = np.maximum(logp1, logp2)
            logp = np.log(np.exp(logp1 - ma) + np.exp(logp2 - ma)) + ma
            logp_next = np.sum(logp)

            p = np.exp(logp1 - ma)/(np.exp(logp1 - ma) + np.exp(logp2 - ma))
            pi = np.mean(p)
            
            if verbose:
                print('iter {} | logp {} | pi {} | mu1 {} | mu2 {}'.format(i+1, logp_next, pi, mu1, mu2))

        # if mu2 > mu1, flip pi
        if mu2 > mu1:
            p = 1-p
            pi = 1-pi

            temp = mu1
            mu1 = mu2
            mu2 = temp

        self.pi = pi
        self.p = p
        self.mu1 = mu1
        self.mu2 = mu2

        if verbose:
            print('terminated after {} iterations, final logp change = {}'.format(i+1, (logp_next-logp_cur)))
            
        return self

