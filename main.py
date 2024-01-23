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
from ListFiles import GetFiles
from ShpMaskWriter import mask_write, mask_write_treads

FileName = "B21-166b_cut.tif"
Path0 = "includes/Pictures"
Path = Path0 + "/" + FileName
img = cv2.imread(Path)  # Try houses.jpg or neurons.jpg
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_line = cv2.imread(Path0 + "/" + "B21-166b_lin_cut.tif")

print(img.shape, img_line.shape)
img = img[0:2 ** 11 - 1, 0:2 ** 11 - 1]
img_line = img_line[0:2 ** 11 - 1, 0:2 ** 11 - 1] #9
print(img.shape, img_line.shape)
r,g,b = cv2.split(img_line)
result_line = r

from rsf_edges import modelini, get_model_edges
from CannyTest import cannythresh, cannythresh_grad
from ThinSegmentation import ThinSegmentation

from scipy.spatial import cKDTree, KDTree


"""
model = modelini()
result_rsf = get_model_edges(model, img)

with open("result.pkl", "wb") as fp:
    pickle.dump(result_rsf, fp)

"""

"""
print("load mask start")
with open("result.pkl", "rb") as fp:
    result_rsf = pickle.load(fp)
print("load mask finish")

TS0 = ThinSegmentation(img, result_rsf, result_line)
TS1 = ThinSegmentation(img, result_rsf, result_line)
TS2 = ThinSegmentation(img, result_rsf, result_line)
TS3 = ThinSegmentation(img, result_rsf, result_line)

"""
#TS0.method0()
#TS1.method1()
#TS2.method2()
#TS3.method3()

"""
print("fig = plt.figure(figsize=(10, 10))")

fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(2, 2, 1),fig.add_subplot(2, 2, 2),fig.add_subplot(2, 2, 3),fig.add_subplot(2, 2, 4)]
ax[0].imshow(TS0.area_marks)
ax[1].imshow(TS1.area_marks)
ax[2].imshow(TS2.area_marks)
ax[3].imshow(TS3.area_marks)
"""

"""
S = TS1.get_marks_areas()

with open("result_S.pkl", "wb") as fp:
    pickle.dump(S, fp)

"""

print("load mask start")
with open("result_S.pkl", "rb") as fp:
    S = pickle.load(fp)
print("load mask finish")



fig = plt.figure(figsize=(10, 10))
"""
ax = [fig.add_subplot(2, 2, 1),
      fig.add_subplot(2, 2, 2),
      fig.add_subplot(2, 2, 3),
      fig.add_subplot(2, 2, 4)]
"""
ax = [fig.add_subplot(1, 1, 1)]

S = np.log(S)
ax[0].hist(S)


plt.show()



exit()
mask_write_treads('Shapes/Shape0/Shape', TS0.get_masks())
mask_write_treads('Shapes/Shape1/Shape', TS1.get_masks())
mask_write_treads('Shapes/Shape2/Shape', TS2.get_masks())
mask_write_treads('Shapes/Shape3/Shape', TS3.get_masks())

cv2.imwrite('Shapes/Size_511.tif', img)
exit()

"""
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#l_c_bg = cv2.morphologyEx(l_c_bg, cv2.MORPH_OPEN, kernel, iterations=3)
"""
