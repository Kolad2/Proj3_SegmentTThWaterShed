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

path = "includes/Shapes/Shape_B21-166b/B21-166b_joined"
poly = get_shp_poly(path)
img_white = np.zeros(sh[0:2], dtype=np.float32)
img_lines = np.zeros(sh[0:2], dtype=np.float32)
img_lines2 = np.zeros(sh[0:2], dtype=np.float32)


def get_blured_line(img_white, polygon, sigma):
      xmin = polygon[:, 0].min() - 10
      xmax = polygon[:, 0].max() + 1 + 10
      ymin = polygon[:, 1].min() - 10
      ymax = polygon[:, 1].max() + 1 + 10
      img_lin = cv2.polylines(img_white, [polygon], False, 1, 1)
      img_lin[ymin:ymax,xmin:xmax] = cv2.GaussianBlur(img_lin[ymin:ymax,xmin:xmax], (9, 9), sigma)
      return img_lin
i1 = 0
for polygon in poly:
      i1 = i1 + 1
      print(i1)
      xmin = polygon[:, 0].min()-10
      xmax = polygon[:, 0].max()+1+10
      ymin = polygon[:, 1].min()-10
      ymax = polygon[:, 1].max()+1+10
      img_white = np.zeros(sh[0:2], dtype=np.float32)
      L = 0
      for i in range(len(polygon)-1):
            L = L + np.sqrt(np.sum((polygon[i+1] - polygon[i]) ** 2))
      img_line = get_blured_line(img_white, polygon, 3)
      img_lines[ymin:ymax,xmin:xmax] = \
            (img_lines[ymin:ymax,xmin:xmax] + img_line[ymin:ymax,xmin:xmax]
             - img_lines[ymin:ymax,xmin:xmax]*img_line[ymin:ymax,xmin:xmax])
print(img_lines.max())
img_lines = np.uint8(img_lines/img_lines.max()*255)
img_lines2 = cv2.polylines(img_white, poly, False, 255, 1)
img_lines2 = np.uint8(img_lines2)


img_lines = img_lines[
      int(sh[0]/2 - sh[0]/3):int(sh[0]/2 + sh[0]/3),
      int(sh[1]/2 - sh[1]/3):int(sh[1]/2 + sh[1]/3)]
img_lines2 = img_lines2[
      int(sh[0]/2 - sh[0]/3):int(sh[0]/2 + sh[0]/3),
      int(sh[1]/2 - sh[1]/3):int(sh[1]/2 + sh[1]/3)]
cv2.imwrite(Path0 + "/" + "B21-166b_lin_cut.tif",img_lines)

cv2.imwrite(Path0 + "/" + "B21-166b_cut.tif",img)


img_lines = img_lines[0:2 ** 9 - 1, 0:2 ** 9 - 1]
img_lines2 = img_lines2[0:2 ** 9 - 1, 0:2 ** 9 - 1]
img = img[0:2 ** 9 - 1, 0:2 ** 9 - 1]
fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(2, 2, 1),
      fig.add_subplot(2, 2, 2),
      fig.add_subplot(2, 2, 3)]
ax[0].imshow(cv2.merge((img_lines,img_lines, img_lines)))
ax[1].imshow(img)
ax[2].imshow(cv2.merge((img_lines2,img_lines2, img_lines2)))
plt.show()