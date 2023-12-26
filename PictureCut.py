import numpy as np
import matplotlib.pyplot as plt
import cv2
from Shpreader import get_shp_poly

FileName = "B21-166b.tif"
Path0 = "includes/Pictures"
Path = Path0 + "/" + FileName
img = cv2.imread(Path)  # Try houses.jpg or neurons.jpg
sh = img.shape

img = img[
      int(sh[0]/2 - sh[0]/3):int(sh[0]/2 + sh[0]/3),
      int(sh[1]/2 - sh[1]/3):int(sh[1]/2 + sh[1]/3)]

"""path = "includes/Shapes/Shape_B21-166b/B21-166b_joined"
poly = get_shp_poly(path)
img_lin = np.zeros(sh[0:2], dtype=np.uint8)
img_lin = cv2.polylines(img_lin, poly, False, 255, 1)

img_lin = img_lin[
      int(sh[0]/2 - sh[0]/3):int(sh[0]/2 + sh[0]/3),
      int(sh[1]/2 - sh[1]/3):int(sh[1]/2 + sh[1]/3)]
cv2.imwrite(Path0 + "/" + "B21-166b_lin_cut.tif",img_lin)"""

cv2.imwrite(Path0 + "/" + "B21-166b_cut.tif",img)


#plt.imshow(img)
#plt.show()