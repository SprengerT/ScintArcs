import os
import numpy as np
import matplotlib as mpl
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt

from ScintArcs import compute_nutSS, compute_staufD, LineFinder

"""
create three numpy arrays:
t : 1d over time in seconds
nu : 1d over frequency in Hz
DS : 2d over time and frequency in this order
"""

nu0 = 1.4e+9 #or np.mean(nu)
fD,tau,SS = compute_nutSS(t,nu,DS,nu0=nu0) #or just do FFT power spectrum

N_stau = 100 #alter to change resolution (lower resolution increases S/N)
tau_max = np.max(tau) #decrease this to cut off high delays with low S/N
stau, fD, staufD = compute_staufD(fD,tau,SS,N_stau=N_stau,tau_max=tau_max)

#range parameters
zeta_max = 2.0e-9 #covers curvatures down to 0.03 s^3
zeta,zeta_err = LineFinder(stau, fD, staufD, nu0=nu0, zeta_max=zeta_max)

eta = 1./(2.*nu0*zeta)**2
eta_err = 2.*eta*zeta_err/zeta
print("eta={0} +- {1}".format(eta,eta_err))
