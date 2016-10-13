# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 09:37:15 2016

@author: Gerwyn Jones
"""
from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt

def f(t, y, l):
    """ A function to call yprime and y """
    return y[1], l*y[0]
    
def Euler(y, f, t, dt, l):
    """ The Y array has two values so we need to split them up 
    when using euler"""
    f1,f2 = f(y, l)   # f1 gives the value y of the function and f2 gives the value of yprime of the function
    yb,yc = y        # yb gives the initial y values and yc gives initail yp values
  
    # using euler method
    ya = yb + dt*f1  # y value
    yp = yc + dt*f2  # yprime value
    
    return ya,yp
    
def Rk4(ya, f, t, dt, l):
    """ The Y array has two values so we need to split them up 
    when using euler"""
#    f1,f2 = f(t, y, l)   # f1 gives the value y of the function and f2 gives the value of yprime of the function
    yi,ypi = ya        # yb gives the initial y values and yc gives initail yp values

    # using Runge-Kutta method
    k1 = f(t, ya, l)
    ki1 = np.array(k1)
    k2 = f(t,ya + ki1*dt/2, l)
    ki2 = np.array(k2)
    k3 = f(t,ya + ki2*dt/2, l)
    ki3 = np.array(k3)
    k4 = f(t,ya + ki3*dt, l)  
  
    y = yi + (dt/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])  # y value
    yp = ypi + (dt/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])  # yprime value
#    
    return y, yp
               
def ODEsolve(Tmax, N, f, method, ic): 
    
    t = np.zeros(N)   # defining the time array
    dt = Tmax/N; t[0] = ic[0]

    y = np.zeros((2,N))  # defining a y array containging the y values and yp values
    y[0,0] = ic[1]; y[1,0] = ic[2]
    
    for i in xrange(0,int(N)-1):
        y[:,i+1]  = method(y[:,i], f, t, dt , ic[3])
        t[i+1] = t[i] + dt
        
    return y, t 
    

def f1(t,l): 
    return np.cos(l*t)    # eq of d^2y/dt^2

def ConvergenceTest(ODEsolve, Tmax, n, f, ic, method, order):
    
    R = [ODEsolve(Tmax, N, f, method, ic) for i,N in enumerate(n)]  
    Y1 = R[0][0][0]; Y2 = R[1][0][0]; Y4 = R[2][0][0]
    
    diff1 = (Y1 - Y2[::2])
    diff2 = (2**order)*(Y2[::2] - Y4[::4])
    
    return diff1,diff2

