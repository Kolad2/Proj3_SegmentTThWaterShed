import os
import sys
import PathCreator
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import time
import pickle
import math
from ListFiles import GetFiles
from ShpMaskWriter import mask_write, mask_write_treads
from rsf_edges import modelini, get_model_edges
from CannyTest import cannythresh, cannythresh_grad
from ThinSegmentation import ThinSegmentation
from scipy.spatial import cKDTree, KDTree
import csv
import pandas as pd


with open('temp/ResultTable.csv', newline='') as csvfile:
    rows = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
    tb = pd.DataFrame(rows[1:], columns=rows[0])

#tb = tb[tb["Lognorm boolean"] == "True"]
tb = tb[tb["ТипыТектонитов"] == "Милонит"]


L_mu = pd.to_numeric(tb["Lognorm mu"]).values
Matr = pd.to_numeric(tb["%матрикса"]).values

print(L_mu)
print(Matr)

fig = plt.figure(figsize=(20, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].plot(L_mu, Matr,'*', color='black')
ax[0].set_xlabel("Lognorm mu")
ax[0].set_ylabel("%матрикса")
plt.show()
exit()