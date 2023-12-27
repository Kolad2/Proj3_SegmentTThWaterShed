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
img = img[0:2 ** 10 - 1, 0:2 ** 10 - 1]

from rsf_edges import modelini, get_model_edges
from CannyTest import cannythresh, cannythresh_grad
from ThinSegmentation import ThinSegmentation


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
TS1.get_marker_rsfcanny()
TS1.get_marker_from_background_iter()
TS1.get_marker_from_background_iter()
TS1.marker_unbound_spread()
TS1.get_marks_areas()


fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(2, 2, 1),
      fig.add_subplot(2, 2, 2),
      fig.add_subplot(2, 2, 3),
      fig.add_subplot(2, 2, 4)]
ax[0].imshow(TS1.img)
ax[0].imshow(TS1.area_marks, alpha=0.5, cmap=plt.get_cmap('tab20'))
plt.show()
exit()
print("second seg start")
TS2 = ThinSegmentation(img)
TS2.set_edge_prob(result)
TS2.get_marker_canny()
TS2.get_marker_from_background_iter()
TS2.marker_unbound_spread()
print("thirst seg start")
TS3 = ThinSegmentation(img)
TS3.set_edge_prob(result)
TS3.get_marker_rsfcannygrad()
TS3.get_marker_from_background_iter()
TS3.marker_unbound_spread()
print("seg end")


FileName = "B21-166b_lin_cut.tif"
Path0 = "includes/Pictures"
Path = Path0 + "/" + FileName
img_lin = cv2.imread(Path)  # Try houses.jpg or neurons.jpg
img_lin = img_lin[0:2 ** 9 - 1, 0:2 ** 9 - 1]


fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(2, 2, 1),
      fig.add_subplot(2, 2, 2),
      fig.add_subplot(2, 2, 3),
      fig.add_subplot(2, 2, 4)]
ax[0].imshow(TS1.img, cmap=plt.get_cmap('gray'))
ax[0].imshow(TS1.area_marks, alpha=0.5)
ax[1].imshow(TS2.img, cmap=plt.get_cmap('gray'))
ax[1].imshow(TS2.area_marks, alpha=0.5)
ax[2].imshow(TS3.img, cmap=plt.get_cmap('gray'))
ax[2].imshow(TS3.area_marks, alpha=0.5)
# ax[2].imshow(TS.area_marks, alpha=0.5)
# ax[3].imshow(edges)
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[3].axis('off')
ax[0].set_title("First method")
ax[1].set_title("Second method")
ax[3].set_title("Thirst method")
plt.show()

mask_write_treads('Shapes/Shape1/Shape', TS1.get_masks())
mask_write_treads('Shapes/Shape2/Shape', TS2.get_masks())
mask_write_treads('Shapes/Shape3/Shape', TS3.get_masks())
cv2.imwrite('Shapes/Size_511.tif', img)
exit()

"""
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#l_c_bg = cv2.morphologyEx(l_c_bg, cv2.MORPH_OPEN, kernel, iterations=3)
"""
