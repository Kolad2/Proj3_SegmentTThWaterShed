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
img = img[0:2 ** 9 - 1, 0:2 ** 9 - 1]

from rsf_edges import modelini, get_model_edges
from CannyTest import cannythresh, cannythresh_grad
from ThinSegmentation import ThinSegmentation

from scipy.spatial import cKDTree, KDTree



"""model = modelini()
result = get_model_edges(model, img)
rsf_edges = result
with open("result.pkl", "wb") as fp:
    pickle.dump(result, fp)"""

print("load mask start")
with open("result.pkl", "rb") as fp:
    result = pickle.load(fp)
print("load mask finish")

print("first seg start")
TS1 = ThinSegmentation(img)
TS1.set_edge_prob(result)
print("get_bg_canny")
TS1.get_bg_canny()


fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
#ax[0].imshow(TS1.img)
#ax[0].imshow(TS1.area_marks, alpha=0.5, cmap=plt.get_cmap('tab20'))
ax[0].imshow(TS1.area_dist, alpha=0.5, cmap=plt.get_cmap('gray'))
plt.show()

exit()
print("get_marker_from_background_iter 1")
TS1.get_marker_from_background_iter()
print("get_marker_from_background_iter 2")
TS1.get_marker_from_background_iter()
print("TS1.area_threshold(20)")
#TS1.area_threshold(20)
print("TS1.marker_unbound_spread()")
TS1.marker_unbound_spread()
print("TS1.get_marks_areas()")
TS1.get_marks_areas()




print("TS1.area_marks_shuffle()")
TS1.area_marks_shuffle()
print("fig = plt.figure(figsize=(10, 10))")

exit()

FileName = "B21-166b_lin_cut.tif"
Path0 = "includes/Pictures"
Path = Path0 + "/" + FileName
img_lin = cv2.imread(Path)  # Try houses.jpg or neurons.jpg
img_lin = img_lin[0:2 ** 9 - 1, 0:2 ** 9 - 1]


mask_write_treads('Shapes/Shape1/Shape', TS1.get_masks())
mask_write_treads('Shapes/Shape2/Shape', TS2.get_masks())
mask_write_treads('Shapes/Shape3/Shape', TS3.get_masks())
cv2.imwrite('Shapes/Size_511.tif', img)
exit()

"""
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#l_c_bg = cv2.morphologyEx(l_c_bg, cv2.MORPH_OPEN, kernel, iterations=3)
"""
