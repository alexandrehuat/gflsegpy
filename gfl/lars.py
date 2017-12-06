#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Joseph Lefevre
Work in Progress
"""

import numpy as np
from .base import GroupFusedLasso


class GFLLARS(GroupFusedLasso):
    def __init__(self):
        raise NotImplementedError

    def detect(self, X, Y, k, epsilon=1e-9, verbose=False, weights=None):
        """
        Detect the breakpoints in X.
        :param X: a matrix n*p
        :param Y: a matrix n*p
        :param k: number of breaking points
        :return: {"lambda", "jump", "value{i}", "meansignal"} with lambda the estimated lambda values for each change-point, jump the successive change-point positions (1*k), value{i} a i*p matrix of change-point values for the first i change-points, meansignal the mean signal per column (1*p vector)
        """

        # res is the dict that we will return     
        res = {"lambda":None, "jump": None, "value": [], "meansignal" : None}
   
        # init of the variables
        if len(Y.size()) > 1:
            [n, p] = Y.size()
        else:
            n = Y.size()
            p = 0
        if weights == None:
            weights = defaultWeights(n)

        res["meansignal"] = Y.mean()
        res["lambda"] = np.zeros(k, 1)
        res["jump"] = np.zeros(k, 1)
         
        # init of cHat = X'*Y
        cHat = X.transpose()*Y

        for i in range(0, k, 1):
            cHatSquareNorm = np.square(cHat).sum(axis=1)
            bigcHat = np.max(cHatSquareNorm)
            besti = np.argmax(cHatSquareNorm)

            # In the first iteration, we add the most correlated feature to the active set. For the other iterations, this is already done at the end of the previous iteration
            if i == 0:
                res.jump[0] = besti

            # Compute the descent direction 
            # w = inv(X(:,A)'*X(:,A))*cHat(A,:)
            A = np.sort(res.jump)
            I = np.argsort(res.jump)

            w = np.linalg.inv(X[:][A].transpose(), X[:][A]).dot(cHat[A][:])
            a = X.transpose().dot(X).dot(w)

            # Compute the descent step
            # For each i we find the largest possible step alpha by solving:
            # norm(cHat(i,:)-alpha*a(i,:)) = norm(cHat(j,:)-alpha*a(j,:)) where j is in the active set.
            # We write it as a second order polynomial 
            # a1(i)*alpha^2 - defaultWeights2* a2(i)*alpha + a3(i)
            a1 = bigcHat - np.square(a).sum(axis=1)
            a2 = bigcHat - np.multiply(a, cHat).sum(axis=1)
            a3 = bigcHat - cHatSquareNorm

            # we solve it
            gammaTemp = np.zeros(2*(n-1), 1)
           
            # First those where we really have a second-order polynomial
            subset = np.where(a1 > epsilon)
            gammeTemp[subset] = np.divide(a2[subset] + np.sqrt(np.square(a2[subset]) - np.multiply(a1[subset], a3[subset])), a1[subset])
            gammeTemp[subset + n - 1] = np.divide(a2[subset] - np.sqrt(np.square(a2[subset]) - np.multiply(a1[subset], a3[subset])), a1[subset])

            # then those where the quadratic term vanishes and we have a first-order polynomial
            subset = np.where((a1 <= epsilon) and (a2 > epsilon))
            gammeTemp[subset] = np.divide(a3[subset], 2*a2[subset])
            gammeTemp[subset + n - 1] = np.divide(a3[subset], 2*a2[subset])

            # Finally the active set should not be taken into account, as well as
            # those for which the computation gives dummy solutions
            maxg = np.max(gammaTemp) + 1
            subset = np.where(a1 <= epsilon and a2 <= epsilon)
            gammaTemp[subset] = maxg
            gammaTemp[subset + n] = maxg
            gammaTemp[A] = maxg
            gammaTemp[n + A - 1] = maxg
            gammaTemp[np.where(gammaTemp <= 0)] = maxg
            gammaTemp[np.imag(gammaTemp) < 1e-5] = maxg

            # now we can take the minimum
            gamma = np.min(gammaTemp)
            nexttoadd = np.argmin(gammaTemp)
            
            # Update
            resTemp = np.zeros(i,p)
            resTemp[I][:] = gamma*w
            if i > 1:
                resTemp[:(i-1)][:] = resTemp[:i-1][:] + res["value"][-1]
            res["value"].append({i:resTemp})
            res["lambda"][i] = np.sqrt(bigcHat)

            if i < k:
                res["jump"][i+1] = 1 + np.mod(nexttoadd-1, n-1)
                cHat = cHat - gamma*a  
    return res

def defaultWeights(n):
    a = np.array(list(range(1, n))).transpose()
    w = np.sqrt(np.divide(n,(np.multiply(a,n-a))))
    return w
