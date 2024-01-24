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
from rsf_edges import modelini, get_model_edges
from CannyTest import cannythresh, cannythresh_grad
from ThinSegmentation import ThinSegmentation
from scipy.spatial import cKDTree, KDTree


FileName = "B21-166b"
Path0 = "/media/kolad/HardDisk/ThinSection"
Path_dir = Path0 + "/" + FileName + "/"
Path_img = Path_dir + "Picture" + "/" + FileName  + ".tif"
# image loading
img = cv2.imread(Path_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
sh = img.shape
img = img[
      int(sh[0]/2 - sh[0]/3):int(sh[0]/2 + sh[0]/3),
      int(sh[1]/2 - sh[1]/3):int(sh[1]/2 + sh[1]/3)]
# RSF_result load
Path_rsf = Path_dir + "RSF_edges" + "/" + FileName + "_edges_cut.tif"
img_rsf = cv2.imread(Path_rsf)
result_rsf,_,_ = cv2.split(img_rsf)
# Lineaments load
"""Path_line = Path_dir + "Lineaments" + "/" + FileName + "_lin_cut.tif"
img_line = cv2.imread(Path_line)
result_line,_,_ = cv2.split(img_line)"""
#
n = 9
start_img = False
b_shapewrite = False
#
#img = img[0:2 ** n, 0:2 ** n]
#result_rsf = result_rsf[0:2 ** n, 0:2 ** n]
#result_line = result_line[0:2 ** n, 0:2 ** n]

if start_img:
    fig = plt.figure(figsize=(10, 10))
    ax = [fig.add_subplot(2, 2, 1),
          fig.add_subplot(2, 2, 2),
          fig.add_subplot(2, 2, 3)]
    ax[0].imshow(cv2.merge((result_rsf,result_rsf,result_rsf)))
    #ax[1].imshow(cv2.merge((result_line,result_line,result_line)))
    ax[2].imshow(img)
    #plt.show()


TS0 = ThinSegmentation(img, result_rsf)
TS0.method0()
print("TS0.get_marks_areas()")
S = TS0.get_marks_areas()
print("pickle")
with open("temp/" + FileName + "_S.pickle", 'wb') as handle:
    pickle.dump(S, handle, protocol=pickle.HIGHEST_PROTOCOL)



"""

TS1 = ThinSegmentation(img, result_rsf, result_line)
TS2 = ThinSegmentation(img, result_rsf, result_line)
TS3 = ThinSegmentation(img, result_rsf, result_line)



TS1.method1()
TS2.method2()
TS3.method3()

print("fig = plt.figure(figsize=(10, 10))")
fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(2, 2, 1),
      fig.add_subplot(2, 2, 2),
      fig.add_subplot(2, 2, 3),
      fig.add_subplot(2, 2, 4)]
ax[0].imshow(TS0.area_marks)
ax[1].imshow(TS1.area_marks)
ax[2].imshow(TS2.area_marks)
ax[3].imshow(TS3.area_marks)
plt.show()


"""

"""
if b_shapewrite:
    mask_write_treads('Shapes/Shape0/Shape0', TS0.get_masks())
    mask_write_treads('Shapes/Shape1/Shape1', TS1.get_masks())
    mask_write_treads('Shapes/Shape2/Shape2', TS2.get_masks())
    mask_write_treads('Shapes/Shape3/Shape3', TS3.get_masks())
    cv2.imwrite("Shapes/img.tif", img)

"""
exit()