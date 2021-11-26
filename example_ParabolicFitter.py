import os
import numpy as np
import matplotlib as mpl
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt

from ScintArcs import compute_nutSS, ParabolicFitter

"""
create three numpy arrays:
t : 1d over time in seconds
nu : 1d over frequency in Hz
DS : 2d over time and frequency in this order
"""

nu0 = 1.4e+9 #or np.mean(nu)
fD,tau,SS = compute_nutSS(t,nu,DS,nu0=nu0) #or just do FFT power spectrum

#range parameters
xmin = -10.0
xmax = 10.0
ymin = 0.0
ymax = 6.0
vmin = 5.2
vmax = 7.7
eta,eta_err = ParabolicFitter(fD,tau,SS,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,vmin=vmin,vmax=vmax)

print("eta={0} +- {1}".format(eta,eta_err))
