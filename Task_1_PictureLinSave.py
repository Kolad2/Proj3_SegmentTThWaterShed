import numpy as np
import PathCreator
import matplotlib.pyplot as plt
import cv2
from Shpreader import get_shp_poly
from Bresenham_Algorithm import line as line_BA
import pickle

#FileName = "B21-234a"
#FileName = "B21-215b"
#FileName = "B21-215a"
#FileName = "B21-213b"
#FileName = "B21-213a"
#FileName = "B21-189b"
#FileName = "B21-188b_2"
#FileName = "B21-151b"
#FileName = "B21-107b"
#FileName = "64-3"
#FileName = "15-2"
#FileName = "15-1"
#
#FileName = "B21-166b"
#FileName = "B21-122a"
#FileName = "B21-120a"
#FileName = "B21-51a"
#FileName = "B21-200b"
#FileName = "B21-192b"
#FileName = "19-5b"


Path0 = "/media/kolad/HardDisk/ThinSection"
Path_dir = Path0 + "/" + FileName + "/"
Path_shape = Path_dir + "Joined/" + FileName + "_joined"
poly = get_shp_poly(Path_shape)

Path_img = Path_dir + "Picture" + "/" + FileName  + ".tif"
img = cv2.imread(Path_img)
sh = img.shape

img_lines = np.zeros(sh[0:2], dtype=np.uint8)
img_lines = cv2.polylines(img_lines, poly, False, 255, 2)
img_lines = img_lines[
      int(sh[0]/2 - sh[0]/3):int(sh[0]/2 + sh[0]/3),
      int(sh[1]/2 - sh[1]/3):int(sh[1]/2 + sh[1]/3)]

cv2.imwrite(Path_dir + "Lineaments/" + FileName + "_lin_cut.tif", img_lines)


#img_lines = img_lines[0:2 ** 9 - 1, 0:2 ** 9 - 1]

fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].imshow(cv2.merge((img_lines,img_lines,img_lines)))
plt.show()