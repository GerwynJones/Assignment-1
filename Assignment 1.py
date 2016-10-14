# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 09:37:15 2016

@author: C1331824
"""
from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt

def f(y, l):
    """ A function to call yprime and y """
    return y[1], l*y[0]
    
def Euler(ya, f, dt, l):
    """ The Y array has two values so we need to split them up 
    when using euler"""
    fy,fyp = f(ya, l)   # fy gives the value y of the function and fyp gives the value of yprime of the function
    yi,ypi = ya        # yi gives the initial y values and ypi gives initail yp values
  
    # using euler method
    y = yi + dt*fy  # y value
    yp = ypi + dt*fyp  # yprime value
    
    return y, yp
               
def ODEsolve(Tmax, N, f, method, ic): 
    
    t = np.zeros(N)   # defining the time array
    dt = Tmax/N; t[0] = ic[0]

    y = np.zeros((2,N))  # defining a y array containging the y values and yp values
    y[0,0] = ic[1]; y[1,0] = ic[2]
    
    for i in xrange(0,int(N)-1):
        y[:,i+1]  = method(y[:,i], f, dt , ic[3])
        t[i+1] = t[i] + dt
        
    return y, t 
    
#lambda
w = 2*np.pi; l = -w*w

#defining initial conditions 
ti = 0; Tmax = 1
yi = 1; ypi = 0

#time steps 
N = 1000; n = np.array([N,2*N,4*N])

#collecting initial conditions
ic = np.array([ti, yi, ypi, l])  # initial time, final time, initial y and lambda
 
# solving ODE
R = [ODEsolve(Tmax, N, f, Euler, ic) for i,N in enumerate(n)]

def f1(t,l): 
    return np.cos(l*t)    # exact solution of d^2y/dt^2

F = [r'$\Delta t$',r'$\Delta t/2$',r'$\Delta t/4$']

for i in range(len(R)):
    T = R[i][1]; Y = R[i][0][0]
    plt.subplot(2,1,1)
    plt.plot(T, Y, label=r'$\Delta t = %.5f$' %(Tmax/n[i])) 
    plt.ylabel(r'$Y$ $(m)$') 
    plt.title('Graph of ODE using Euler method')
    plt.legend(loc='best') 
    plt.grid() 
    plt.subplot(2,1,2)
    plt.plot(T, f1(T,w)-Y, label=F[i] )
    plt.xlabel(r'$Time$ $(s)$')
    plt.ylabel(r'$Error$ $(m)$') 
    plt.legend(loc='best')
    plt.grid() 
plt.savefig('Graph of ODE.png', bbox_inches='tight')

def ConvergenceTest(ODEsolve, Tmax, n, f, ic, method, order):
    
    R = [ODEsolve(Tmax, N, f, method, ic) for i,N in enumerate(n)]  
    Y1 = R[0][0][0]; Y2 = R[1][0][0]; Y4 = R[2][0][0]
    
    diff1 = (Y1 - Y2[::2])
    diff2 = (2**order)*(Y2[::2] - Y4[::4])
    
    return diff1,diff2

d1,d2 = ConvergenceTest(ODEsolve, Tmax, n, f, ic, Euler, 1)

plt.figure()
plt.subplot(2,1,1)
plt.plot(T[::4],d1, label=r'$Y - Y/2$')
plt.plot(T[::4],d2, label=r'$2\left(Y/2 - Y/4\right)$')
plt.ylabel(r'$Error$ $(m)$') 
plt.title('Graph of Convergence and Errors')
plt.legend(loc='best') 
plt.grid() 
plt.subplot(2,1,2)
plt.plot(T[::4],d1/d2, label=r'$\frac{Y - Y/2}{2\left(Y/2 - Y/4\right)}$')
plt.xlabel(r'$Time$ $(s)$')
plt.ylabel(r'$Convergence$') 
plt.legend(loc='best')
plt.grid() 
plt.savefig('Graph of Convergence.png', bbox_inches='tight')  

dt = (1/2)**np.linspace(1,18,18); Na = Tmax/dt

A = [ODEsolve(Tmax, n, f, Euler, ic) for i, n in enumerate(Na)] 

Ydt = [A[i][0][0][-1] for i in range(len(A))]
Tdt = [A[i][1][-1] for i in range(len(A))]

Te = np.array(Tdt); Ye = f1(Te,w)
 
plt.figure()   
plt.loglog(Na, Ydt-Ye, label=r'$Error$')
plt.xlabel(r'$Time$')
plt.ylabel(r'$Y$') 
plt.title('Graph of ODE')
plt.legend(loc='best') 
plt.grid() 
plt.show()    
plt.savefig('Graph of Error.png', bbox_inches='tight')  