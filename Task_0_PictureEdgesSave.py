import numpy as np
import PathCreator
import matplotlib.pyplot as plt
import cv2
from Shpreader import get_shp_poly
from Bresenham_Algorithm import line as line_BA
import pickle
from rsf_edges import modelini, get_model_edges
from CannyTest import cannythresh, cannythresh_grad
from ThinSegmentation import ThinSegmentation


FileName = "B21-166b"
Path0 = "/media/kolad/HardDisk/ThinSection"
Path_dir = Path0 + "/" + FileName + "/"
Path_img = Path_dir + "Picture" + "/" + FileName  + ".tif"
img = cv2.imread(Path_img)  # Try houses.jpg or neurons.jpg
sh = img.shape

img = img[
      int(sh[0]/2 - sh[0]/3):int(sh[0]/2 + sh[0]/3),
      int(sh[1]/2 - sh[1]/3):int(sh[1]/2 + sh[1]/3)]

img = img[0:2 ** 9 - 1, 0:2 ** 9 - 1].copy()

model = modelini()
result_rsf = get_model_edges(model, img)
result_rsf = np.uint8((result_rsf / result_rsf.max()) * 255)
img_rsf = cv2.merge((result_rsf, result_rsf, result_rsf))



cv2.imwrite(Path_dir + "RSF_edges" + "/" + FileName + "_edges1.tif", img_rsf)

exit()
fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].imshow(img_rsf)
plt.show()