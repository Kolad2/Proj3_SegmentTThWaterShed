import numpy as np
import matplotlib.pyplot as plt
import cv2

FileName = "B21-166b.tif"
Path0 = "includes/Pictures"
Path = Path0 + "/" + FileName
img = cv2.imread(Path)  # Try houses.jpg or neurons.jpg
sh = img.shape
print(sh)
img = img[
      int(sh[0]/2 - sh[0]/3):int(sh[0]/2 + sh[0]/3),
      int(sh[1]/2 - sh[1]/3):int(sh[1]/2 + sh[1]/3)]
cv2.imwrite(Path0 + "/" + "B21-166b_cut.tif",img)


plt.imshow(img)
plt.show()