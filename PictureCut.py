import numpy as np
import PathCreator
import matplotlib.pyplot as plt
import cv2
from Shpreader import get_shp_poly
from Bresenham_Algorithm import line as line_BA
import pickle

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

img_lines = np.zeros(sh[0:2], dtype=np.uint8)
img_lines = cv2.polylines(img_lines, poly, False, 255, 2)
img_lines = img_lines[
      int(sh[0]/2 - sh[0]/3):int(sh[0]/2 + sh[0]/3),
      int(sh[1]/2 - sh[1]/3):int(sh[1]/2 + sh[1]/3)]

cv2.imwrite(Path0 + "/" + "B21-166b_lin_cut.tif", img_lines)
cv2.imwrite(Path0 + "/" + "B21-166b_cut.tif", img)

print("load mask start")
with open("result.pkl", "rb") as fp:
    result = pickle.load(fp)
print("load mask finish")


img_lines = img_lines[0:2 ** 9 - 1, 0:2 ** 9 - 1]
img = img[0:2 ** 9 - 1, 0:2 ** 9 - 1]

result = np.uint8((result / result.max()) * 255)
ret, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
result = np.uint8(result)
kernel = np.ones((2, 2), np.uint8)

result2 = cv2.ximgproc.thinning(result)
result2 = cv2.dilate(result2, kernel, iterations=1)
img_bg = cv2.add(img_lines, result)
img_bg2 = cv2.add(img_lines, result2)


fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(2, 2, 1),
      fig.add_subplot(2, 2, 2),
      fig.add_subplot(2, 2, 3),
      fig.add_subplot(2, 2, 4)]
ax[0].imshow(cv2.merge((img_lines,img_lines, img_lines)))
ax[1].imshow(img)
ax[2].imshow(cv2.merge((img_bg,img_bg,img_bg)))

img_bg2 = cv2.dilate(img_bg2, kernel, iterations=1)
ax[3].imshow(cv2.merge((img_bg2,img_bg2,img_bg2)))

plt.show()