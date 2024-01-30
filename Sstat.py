import os
import sys
import PathCreator
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import pickle
import math
from scipy.ndimage import histogram
from scipy import stats as st
from scipy.special import factorial
from scipy.special import gamma
import scipy.stats as st
import scipy.optimize as opt
from ListFiles import GetFiles
from ShpMaskWriter import mask_write, mask_write_treads

#FileName = "B21-166b"
#FileName = "B21-122a"
#FileName = "B21-120a"
FileName = "B21-51a"

"""

K = np.linspace(0,10)
lam = 10
S = 2**(20 - K)
LogN = 2**(K*np.log2(lam) - np.log2(gamma(K+1)) + lam/2*(2.71) + 5)

fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].plot(S, LogN,'*')


"""
pl = 2.5
pl2 = (pl)**2

rv = st.lognorm(4)
log_bins = np.logspace(0,5,100)
f = rv.pdf(log_bins)
"""
fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].plot(log_bins, f,'*')
plt.show()
"""
#exit()
print("load mask start")
with open("temp/" + FileName + "_S.pickle", "rb") as fp:
    S = pickle.load(fp)
print("load mask finish")

print(len(S))
exit()
mask = S > 10
#S = S[mask]


#theta = st.powerlognorm.fit(S,floc=0, fscale=1)
#theta = st.pareto.fit(S)
#theta = st.exponweib.fit(S,floc=0,fscale=1)

#
#rv2 = st.powerlognorm(theta[0],theta[1],theta[2],theta[3])
#rv2 = st.pareto(theta[0],theta[1],theta[2])
#rv2 = st.exponweib(theta[0],theta[1],theta[2],theta[3])
#f2 = rv2.pdf(log_bins)


log_bins = np.logspace(1,5,100)
counts, bins = np.histogram(S,bins=log_bins)
Dl = bins[1:] - bins[0:-1]
f_0 = counts/(Dl*sum(counts))


theta = st.lognorm.fit(S)
rv = st.lognorm(theta[0], theta[1], theta[2])
rng = np.random.default_rng()
x = rv.rvs(size=10**6, random_state=rng)

f2 = rv.pdf(log_bins)
KS_TEST = st.ks_1samp(x, rv.cdf)
KS_TEST2 = st.ks_1samp(S, rv.cdf,method='asymp', alternative='less')
print(KS_TEST,KS_TEST2)

counts, bins = np.histogram(x,bins=log_bins)
f_2 = counts/(Dl*sum(counts))

fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].stairs(f_0, log_bins*pl2, fill=True)
ax[0].plot(log_bins*pl2,f2)
ax[0].stairs(f_2, log_bins*pl2, fill=False)
ax[0].set_title(FileName)
fig.savefig("temp/" + FileName + "_S.png")
plt.show()