import os
import sys
import PathCreator
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
img = img[0:2 ** 11 - 1, 0:2 ** 11 - 1]
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_c, a, b = cv2.split(lab)
l_c = cv2.bilateralFilter(l_c, 15, 40, 80)
l_c_bg = 255 - cv2.Canny(l_c, 100, 200)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
l_c_bg = cv2.morphologyEx(l_c_bg, cv2.MORPH_OPEN, kernel, iterations=3)

l_c_dist = cv2.distanceTransform(l_c_bg, cv2.DIST_L2, 0)
l_c_dist = np.uint8(l_c_dist / l_c_dist.max() * 255)
# thresholding
ret, l_c_su = cv2.threshold(l_c_dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# l_c_su = cv2.adaptiveThreshold(l_c_dist, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)

l_c_su = l_c_su.astype(np.uint8)
# Finding unknown region


l_c_unknown = cv2.subtract(l_c_bg, l_c_su)
# Marker labelling

ret, l_c_marks = cv2.connectedComponents(l_c_su)
# l_c_marks = l_c_marks.astype(np.uint32)
l_c_marks = l_c_marks + 1

l_c_marks[l_c_unknown == 255] = 0
l_c_marks2 = cv2.watershed(img, l_c_marks)
# l_c_marks2 = cv2.watershed(img,l_c_marks)
# img2 = cv2.merge((l_c,l_c,l_c))
# img[l_c_marks2 == -1] = [255,0,0]

masks = []
for i in range(l_c_marks2.max()):
      mask = {'segmentation': l_c_marks2 == i, 'bbox': (0, 0, 0, 0)}
      segmentation = np.where(mask['segmentation'])
      if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))
            mask['bbox'] = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
            masks.append(mask)
print("Длинна: ",len(masks))

mask_write_treads('Shape_test2/Shape', masks)

cv2.imwrite('test.tif', img)

fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(2, 2, 1),
      fig.add_subplot(2, 2, 2),
      fig.add_subplot(2, 2, 3),
      fig.add_subplot(2, 2, 4)]
ax[0].imshow(cv2.merge((l_c_bg, l_c_bg, l_c_bg)))
ax[1].imshow(cv2.merge((l_c_su, l_c_su, l_c_su)))
ax[2].imshow(img)
l_c_marks.max()
l_c_marks2[l_c_marks2 != 9] = 0
ax[3].imshow(l_c_marks2)
ax[3].axis('off')
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[0].set_title("background")
ax[1].set_title("sure foreground")

plt.show()
