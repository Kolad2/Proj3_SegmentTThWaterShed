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
from ListFiles import GetFiles
from ShpMaskWriter import mask_write, mask_write_treads

FileName = "B21-166b"

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

plt.show()