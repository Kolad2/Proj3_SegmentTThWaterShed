import os
import sys
import PathCreator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shapefile

def get_shp_poly(path):
    shapes: list[Any]
    with shapefile.Reader(path) as shp:
        shapes = shp.shapes()
        bbox = shp.bbox
    poly = []
    for i in range(len(shapes)):
        poly.append(np.array(shapes[i].points, np.int32) * [1, -1])
    return poly


"""poly = get_shp_poly(path)

image = np.zeros((11402, 9734), dtype=np.uint8)
image = cv2.polylines(image, poly, False, 255, 5)

fig = plt.figure(figsize=(10, 5))
ax = [fig.add_subplot(1, 1, 1)]
ax[0].imshow(image, cmap=plt.get_cmap('gray'))
ax[0].axis('off')
ax[0].set_title("background")
plt.show()

cv2.imwrite('shape.tif',image)"""