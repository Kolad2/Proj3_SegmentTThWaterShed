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
from ListFiles import GetFiles
from ShpMaskWriter import mask_write, mask_write_treads

#FileName = "B21-166b"
#FileName = "B21-122a"
#FileName = "B21-120a"
#FileName = "B21-51a"


kmax = np.uint8(5*np.log2(10) - 1)
print(kmax)
K = np.arange(1,kmax)
S = (5*np.log2(10) - K)
lam = 100
N = (K*np.log2(lam) - np.log2(gamma(K + 1)) - lam/2 + 100)



fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
#ax[0].set_xscale('log')
#ax[0].set_yscale('log')
ax[0].plot(S, N,'*')
plt.show()

exit()
print("load mask start")
with open("temp/" + FileName + "_S.pickle", "rb") as fp:
    S = pickle.load(fp)
print("load mask finish")

#mask = S > 5
#S = S[mask]

log_bins = np.logspace(0,5,100)
counts, bins = np.histogram(S,bins=log_bins)


fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].stairs(counts, log_bins, fill=True)
ax[0].set_title(FileName)
fig.savefig("temp/" + FileName + "_S.png")
plt.show()