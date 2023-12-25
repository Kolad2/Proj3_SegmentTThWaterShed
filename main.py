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




i_num = 0
FileName = "B21-166b_cut.tif"
Path0 = "includes/Pictures"
Path = Path0 + "/" + FileName
img = cv2.imread(Path)  # Try houses.jpg or neurons.jpg
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img[0:2 ** 9 - 1, 0:2 ** 9 - 1]





class ThinSegmentation:
      def __init__(self, img):
            self.img = img
            self.shape = img.shape
            self.area_sure = np.empty(self.shape[0:2], dtype=np.uint8)
            self.area_unknown = np.empty(self.shape[0:2], dtype=np.uint8)
            self.area_bg = np.empty(self.shape[0:2], dtype=np.uint8)
            self.area_marks = np.empty(self.shape[0:2], dtype=np.int32)

      def watershed_iter(self, area_marks):
            area_marks[area_marks == -1] = 0
            area_marks[area_marks == 1] = 0
            area_marks[self.area_bg == 0] = 1
            return cv2.watershed(self.img, area_marks)

      def get_marker_from_background(self):
            area_bg = self.area_bg
            area_dist = cv2.distanceTransform(area_bg, cv2.DIST_L2, 0)
            area_dist = np.uint8((area_dist / area_dist.max()) * 255)
            ret, area_sure = cv2.threshold(
                  area_dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ret, area_marks = cv2.connectedComponents(area_sure)
            area_unknown = cv2.subtract(area_bg, area_sure)
            area_marks = area_marks + 1
            for i in range(10):
                  area_marks = self.watershed_iter(area_marks)


            self.area_marks[:] = area_marks
            self.area_sure[:] = area_sure
            self.area_unknown[:] = area_unknown
            return self.area_marks

      def GetMarkerArea(self):
            lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
            l_c, a, b = cv2.split(lab)
            l_c = cv2.bilateralFilter(l_c, 15, 40, 80)
            area_bg = 255 - cv2.Canny(l_c, 100, 200)
            self.area_bg = area_bg
            area_marks = self.get_marker_from_background()
            return area_marks


from rsf_edges import modelini, get_model_edges
from CannyTest import cannythresh

"""model = modelini()
result = get_model_edges(model, img)
rsf_edges = result
with open("result.pkl", "wb") as fp:
    pickle.dump(result, fp)"""

TS = ThinSegmentation(img)
area_marks = TS.GetMarkerArea()

print("load mask start")
with open("result.pkl", "rb") as fp:
    result = pickle.load(fp)
print("load mask finish")
edges = cannythresh(result)
print(edges.min())
print(edges.max())

fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(2, 2, 1),
      fig.add_subplot(2, 2, 2),
      fig.add_subplot(2, 2, 3),
      fig.add_subplot(2, 2, 4)]
ax[0].imshow(result, cmap=plt.get_cmap('gray'))
ax[1].imshow(edges, cmap=plt.get_cmap('gray'))
ax[2].imshow(img, cmap=plt.get_cmap('gray'))
ax[3].imshow(TS.area_bg, cmap=plt.get_cmap('gray'))
plt.show()
exit()




fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(2, 2, 1),
      fig.add_subplot(2, 2, 2),
      fig.add_subplot(2, 2, 3),
      fig.add_subplot(2, 2, 4)]
ax[0].imshow(TS.img, cmap=plt.get_cmap('gray'))
print(TS.area_marks.min())
ax[0].imshow(TS.area_marks, alpha=0.5)
#ax[1].pcolormesh(TS.area_bg, cmap=plt.get_cmap('PuBu_r'))
ax[1].imshow(cv2.subtract(TS.area_bg, TS.area_sure), cmap=plt.get_cmap('gray'))
ax[2].imshow(TS.area_bg, cmap=plt.get_cmap('gray'))
ax[2].imshow(TS.area_marks, alpha=0.5)
ax[3].imshow(edges)
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[3].axis('off')
ax[0].set_title("markers iteration")
ax[1].set_title("sure $l_c$ foreground")
plt.show()

exit()

masks = []
for i in range(area_marks.max()):
    mask = {'segmentation': area_marks == i, 'bbox': (0, 0, 0, 0)}
    segmentation = np.where(mask['segmentation'])
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))
        mask['bbox'] = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
        masks.append(mask)
print("Длинна: ", len(masks))

mask_write_treads('Shape_test2/Shape', masks)
cv2.imwrite('test.tif', img)


"""
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#l_c_bg = cv2.morphologyEx(l_c_bg, cv2.MORPH_OPEN, kernel, iterations=3)
# Marker labelling

# mask_write_treads('Shape_test2/Shape', masks)
# 

"""
