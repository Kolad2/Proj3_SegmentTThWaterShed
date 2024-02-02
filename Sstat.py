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

FileName = "B21-166b"
#FileName = "B21-122a"
#FileName = "B21-120a"
#FileName = "B21-51a"
#FileName = "B21-200b"
#FileName = "B21-192b"
#FileName = "19-5b"
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

dstr_1 = st.lognorm(4)
log_bins = np.logspace(0,5,100)
f = dstr_1.pdf(log_bins)
"""
fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].plot(log_bins, f,'*')
plt.show()
"""

print("load mask start")
with open("temp/" + FileName + "_S.pickle", "rb") as fp:
    S = pickle.load(fp)
print("load mask finish")

mask = S > 10
#S = S[mask]


#
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
N = len(S)

theta_1 = st.lognorm.fit(S); print(len(theta_1))
theta_2 = st.expon.fit(S); print(len(theta_2))
theta_3 = st.powerlaw.fit(S); print(len(theta_3))
theta_4 = st.pareto.fit(S); print(len(theta_4))
theta_5 = st.powerlognorm.fit(S); print(len(theta_5))


dstr_0 = st.rv_histogram((counts, bins), density=False)
dstr_1 = st.lognorm(theta_1[0], theta_1[1], theta_1[2])
dstr_2 = st.expon(theta_2[0], theta_2[1])
dstr_3 = st.powerlaw(theta_3[0], theta_3[1], theta_3[2])
dstr_4 = st.pareto(theta_4[0], theta_4[1], theta_4[2])
dstr_5 = st.powerlognorm(theta_5[0],theta_5[1],theta_5[2],theta_5[3])
rng = np.random.default_rng()


K = [[] for i in range(0,5)]
S_G = [[] for i in range(0,5)]
f_G = [[] for i in range(0,5)]
for i in range(0,10):
    print(i)
    S_G0 = dstr_0.rvs(size=N, random_state=rng)
    S_G[0] = dstr_1.rvs(size=N, random_state=rng)
    S_G[1] = dstr_2.rvs(size=N, random_state=rng)
    S_G[2] = dstr_3.rvs(size=N, random_state=rng)
    S_G[3] = dstr_4.rvs(size=N, random_state=rng)
    S_G[4] = dstr_5.rvs(size=N, random_state=rng)
    S_GU = st.uniform.rvs(size=N) * 2 - 1
    #S_G0 = S_G0 + 0.1*S_G0*S_GU
    #S_G0 = S_G0[(S_G0 > min(S)) & (S_G0 < max(S))]
    f_G0, bins = np.histogram(S_G0, bins=bins, density=True)
    for i in range(0,5):
        S_G[i] = S_G[i][(S_G[i] > min(S)) & (S_G[i] < max(S))]
        f_G[i], bins = np.histogram(S_G[i], bins=bins, density=True)
        K[i].append(max(abs(np.cumsum(f_G0 * Dl) - np.cumsum(f_G[i] * Dl))))

linbins = [[] for i in range(0,5)]
f_K = [[] for i in range(0,5)]
for i in range(0,5):
    K[i] = np.array(K[i])
    linbins[i] = np.linspace(K[i].mean()-3*K[i].std(),K[i].mean()+3*K[i].std(),50)
    f_K[i], _ = np.histogram(K[i],bins=linbins[i],density=True)


KS1 = st.ks_1samp(S, dstr_1.cdf)
KS2 = st.ks_1samp(S, dstr_2.cdf)
KS3 = st.ks_1samp(S, dstr_3.cdf)
KS4 = st.ks_1samp(S, dstr_4.cdf)
KS5 = st.ks_1samp(S, dstr_5.cdf)
print(KS1)
print(KS2)
print(KS3)
print(KS4)
print(KS5)


fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
#ax[0].set_xscale('log')
#ax[0].set_yscale('log')
ax[0].stairs(np.cumsum(f_K[0]*6*K[0].std()/50), linbins[0], fill=False, label='lognorm',linewidth=2.0,color='red')
ax[0].stairs(np.cumsum(f_K[1]*6*K[1].std()/50), linbins[1], fill=False, label='expon',linewidth=2.0,color='blue')
ax[0].stairs(np.cumsum(f_K[2]*6*K[2].std()/50), linbins[2], fill=False, label='power',linewidth=2.0,color='y')
ax[0].stairs(np.cumsum(f_K[3]*6*K[3].std()/50), linbins[3], fill=False, label='pareto',linewidth=2.0,color='g')
ax[0].stairs(np.cumsum(f_K[4]*6*K[4].std()/50), linbins[4], fill=False, label='powerlognorm',linewidth=2.0,color='violet')
#ax[0].stairs(f_G1, log_bins, fill=False)
ax[0].set_title(FileName)
ax[0].legend()
#ax[0].set_xlim((10**2, 10**2))
#plt.show()
fig.savefig("temp/" + FileName + "_pf_S.png")


f_1 = dstr_1.pdf(log_bins)
f_2 = dstr_2.pdf(log_bins)
f_3 = dstr_3.pdf(log_bins)
f_4 = dstr_4.pdf(log_bins)
f_5 = dstr_5.pdf(log_bins)


fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].stairs(f_0, log_bins, fill=True)
ax[0].plot(log_bins, f_1, label='lognorm',linewidth=2.0,color='red')
ax[0].plot(log_bins, f_2, label='expon',linewidth=2.0,color='blue')
ax[0].plot(log_bins, f_3, label='power',linewidth=2.0,color='y')
ax[0].plot(log_bins, f_4, label='pareto',linewidth=2.0,color='g')
ax[0].plot(log_bins, f_5, label='powerlognorm',linewidth=2.0,color='brown')
ax[0].set_title(FileName)
ax[0].legend()
#ax[0].set_xlim((10**2, 10**2))
ax[0].set_ylim((10**(-10)), 10**(-2))
fig.savefig("temp/" + FileName + "_log_pdf_S.png")
plt.show()