import os
import sys
import scipy.io
import PathCreator
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import math
import scipy as sp
from scipy.ndimage import histogram
from scipy import stats as st
from scipy.special import factorial
from scipy.special import gamma
import scipy.optimize as opt
from scipy.optimize import minimize



def target_lognorm(s, scale, X, xmin, xmax):
    dist = st.lognorm(s, 0, scale)
    Fmin = dist.cdf(xmin)
    SlnX = np.mean(np.log(X))
    SlnX2 = np.mean((np.log(X))**2)
    N = len(X)
    part1 = -SlnX - np.log(s) - (0.5*(s**2))*SlnX2
    part2 = (scale/(s**2))*SlnX - (scale**2)/(2*(s**2))
    part3 = - np.log(1-Fmin)
    print(part1 + part2 + part3, s, scale)
    return part1 + part2 + part3


def target_lognorm2(s, scale, X, xmin, xmax):
    dist = st.lognorm(s, 0, scale)
    Fmin = dist.cdf(xmin)
    Fmax = dist.cdf(xmax)
    N = len(X)
    return np.sum(np.log(dist.pdf(X))) - N*np.log(Fmax - Fmin)


def target_expon(scale, X, xmin):
    Fmin = 1 - np.exp(xmin/scale)
    S = -(-np.sum(X)/scale - len(X)*np.log(scale) - len(X)*np.log(1 - Fmin))
    return S

def GetThetaLognorm(X, xmin, xmax):
    theta = st.lognorm.fit(X, floc=0)
    res = minimize(lambda x: -target_lognorm2(x[0], x[1], X, xmin, xmax),
                   [theta[0], theta[2]],
                   bounds=((0, None), (0, None)),
                   method='Nelder-Mead', tol=1e-3)
    return res.x[0], 0, res.x[1]

def GetThetaExpon(X, xmin):
    theta = st.expon.fit(X, floc=0)
    res = minimize(lambda x: target_expon(x, X, xmin), theta[1], method='Nelder-Mead', tol=1e-3)
    return 0, res.x[0]

def GetF(S, xmin, xmax=10 ** 10):
    F_bins, F = np.unique(S, return_counts=True)
    F_bins = np.insert(F_bins,0,0)
    F_bins = np.append(F_bins, xmax)
    F = np.cumsum(F)
    F = np.insert(F, 0, 0)
    F = F / F[-1]
    return F_bins, F

def Getf(S, f_bins):
    f, _ = np.histogram(S, bins=f_bins, density=True)
    return f

FileNames = ["B21-234a",    #0
             "B21-215b",    #1
             "B21-215a",    #2
             "B21-213b",    #3
             "B21-213a",    #4
             "B21-189b",    #5
             "B21-188b_2",  #6
             "B21-151b",    #7
             "B21-107b",    #8
             "64-3",        #9
             "15-2",        #10
             "15-1",        #11
             "B21-166b",    #12
             "B21-122a",    #13
             "B21-120a",    #14
             "B21-51a",     #15
             "B21-200b",    #16
             "B21-192b",    #17
             "19-5b"]       #18
FileName = FileNames[0]

xmin = 20
xmax = 10*10

matdict = scipy.io.loadmat("temp/StatisticCorData/" + FileName + "/" + FileName + "_1" + "_S.mat", squeeze_me=True)

hS = matdict['S']
P = matdict['P']
F_0 = []
f_0 = []
F_0_bins = []
f_bins = xmin*np.logspace(0,10,30,base=2)
f_0_m = np.empty(len(f_bins))
f_0_sgm = np.empty(len(f_bins))
numinter = 500
for j in range(0,numinter):
    hS[j] = hS[j][hS[j] > xmin]
    f_0.append(Getf(hS[j], f_bins))
    F_bins, F = GetF(hS[j], xmin, np.max(hS[j]))
    F = np.interp(f_bins, F_bins[0:-1], F)
    F_0.append(F)


f_0_med = np.median(f_0, axis=0)
f_0_m = np.mean(f_0, axis=0)
f_0_sgm = np.sqrt(np.mean((f_0-f_0_m)**2, axis=0))
f_0_low = np.quantile(f_0, 0.1, axis=0)
f_0_height = np.quantile(f_0, 0.9, axis=0)
F_0_max = np.max(F_0, axis=0)
F_0_min = np.min(F_0, axis=0)


theta = [[] for i in range(0,2)]
dist = [[] for i in range(0,2)]
F = [[] for i in range(0,2)]
f = [[] for i in range(0,2)]


#theta[0] = GetThetaLognorm(np.concatenate(hS), xmin)
theta[0] = GetThetaLognorm(hS[0], xmin, max(hS[0]))
print(theta[0])
#theta[1] = GetThetaExpon(hS[0], xmin)

dist[0] = st.lognorm(theta[0][0], 0, theta[0][2])

print(sum(f_0_height*(f_bins[1:] - f_bins[0:-1])))
print(sum(f_0_low*(f_bins[1:] - f_bins[0:-1])))
print(sp.integrate.trapezoid(dist[0].pdf(f_bins)/(1 - dist[0].cdf(xmin)),f_bins))


#dist[1] = st.expon(0, theta[1][1])





F = (dist[0].cdf(f_bins) - dist[0].cdf(xmin))/(1 - dist[0].cdf(xmin))

fig = plt.figure(figsize=(20, 10))
ax = [fig.add_subplot(1, 2, 1),
      fig.add_subplot(1, 2, 2)]
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].fill_between(f_bins, F_0_min, F_0_max, alpha=.5, linewidth=0, color='red')
#ax[0].plot(f_bins, F_0_min, color='red')
#ax[0].plot(f_bins, F_0_max, color='red')
ax[0].plot(f_bins, F, color='black')
ax[0].set_xlim((f_bins[2], f_bins[-1]))
ax[0].set_ylim((F[2], F[-1]))
#%%
ax[1].set_xscale('log')
ax[1].stairs(f_0_height, f_bins, fill=True, color='red')
ax[1].stairs(f_0_low, f_bins, fill=True, color='blue')
ax[1].plot(f_bins, dist[0].pdf(f_bins)/(1 - dist[0].cdf(xmin)), color='black')
fig.savefig("temp/" + FileName + "_pf_S.png")
plt.show()
exit()


#ax[1].plot(log_bins, f[0],color='b')
#ax[1].plot(log_bins, f[1],color='r')


fig = plt.figure(figsize=(20, 10))
ax = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
ax[0].set_xscale('log')
ax[0].stairs(F_0, bins, fill=True)
ax[0].plot(log_bins, F[0],color='b')
ax[0].plot(log_bins, F[1],color='b')

ax[1].set_xscale('log')
#ax[1].set_yscale('log')
ax[1].stairs(f_0, log_bins, fill=True)
ax[1].plot(log_bins, f[0],color='b')
ax[1].plot(log_bins, f[1],color='r')
plt.show()


pl = 2.5
pl2 = (pl)**2

"""
log_bins = np.logspace(0,30,30,base=2)
f_0, bins = np.histogram(S,bins= log_bins, density=False)
dstr0 = st.rv_histogram((f_0, bins), density=False)
y = dstr0.cdf(log_bins)
print(y)
fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].set_xscale('log')
ax[0].set_yscale('log')
#ax[0].plot(log_bins,y)
ax[0].stairs(f_0, log_bins, fill=True)
plt.show()
exit()


def get_cdfcor(dist, X, xmin):
    return (dist.cdf(X) - dist.cdf(xmin)) / (1 - dist.cdf(xmin))

for i in range(0,2):
    F[i] = get_cdfcor(dist[i], log_bins, xmin)
    f[i] = (dist[i].pdf(log_bins))/(1 - dist[i].cdf(xmin))
"""



S = S + np.sqrt(S)
mask = S > 10
S = S[mask]

log_bins = np.logspace(1,20,20,base=2)
counts, bins = np.histogram(S,bins=log_bins)
Dl = bins[1:] - bins[0:-1]
f_0 = counts/(Dl*sum(counts))
N = len(S)

theta_1 = st.lognorm.fit(S,floc=0); print(len(theta_1))
theta_2 = st.expon.fit(S,floc=0); print(len(theta_2))
theta_3 = st.powerlaw.fit(S,floc=0); print(len(theta_3))
theta_4 = st.pareto.fit(S,floc=0); print(len(theta_4))
theta_5 = st.powerlognorm.fit(S,floc=0); print(len(theta_5))

dstr = [[] for i in range(0,5)]
dstr_0 = st.rv_histogram((counts, bins), density=False)
dstr[0] = st.lognorm(theta_1[0], theta_1[1], theta_1[2])
dstr[1] = st.expon(theta_2[0], theta_2[1])
dstr[2] = st.powerlaw(theta_3[0], theta_3[1], theta_3[2])
dstr[3] = st.pareto(theta_4[0], theta_4[1], theta_4[2])
dstr[4] = st.powerlognorm(theta_5[0],theta_5[1],theta_5[2],theta_5[3])
rng = np.random.default_rng()


K = [[] for i in range(0,5)]
S_G = [[] for i in range(0,5)]
f_G = [[] for i in range(0,5)]
for i in range(0,10):
    print(i)
    S_G0 = dstr_0.rvs(size=N, random_state=rng)
    for i in range(0, 5):
        S_G[i] = dstr[i].rvs(size=N, random_state=rng)
    S_GU = st.uniform.rvs(size=N) * 2 - 1
    S_G0 = S_G0 + 0.1*S_G0*S_GU
    S_G0 = S_G0[(S_G0 > min(S)) & (S_G0 < max(S))]
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


KS1 = st.ks_1samp(S, dstr[0].cdf)
KS2 = st.ks_1samp(S, dstr[1].cdf)
KS3 = st.ks_1samp(S, dstr[2].cdf)
KS4 = st.ks_1samp(S, dstr[3].cdf)
KS5 = st.ks_1samp(S, dstr[4].cdf)
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


f_1 = dstr[0].pdf(log_bins)
f_2 = dstr[1].pdf(log_bins)
f_3 = dstr[2].pdf(log_bins)
f_4 = dstr[3].pdf(log_bins)
f_5 = dstr[4].pdf(log_bins)


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