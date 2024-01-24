import numpy as np
import PathCreator
import matplotlib.pyplot as plt
import cv2
from rsf_edges import modelini, get_model_edges

FileName = "B21-166b"
Path0 = "/media/kolad/HardDisk/ThinSection"
PathDir = Path0 + "/" + FileName + "/"
img_edges = cv2.imread(PathDir + "RSF_edges/" + FileName + "_edges_cut.tif")
img = cv2.imread(PathDir + "Picture/" + FileName + ".tif")
sh = img.shape
img = img[
      int(sh[0]/2 - sh[0]/3):int(sh[0]/2 + sh[0]/3),
      int(sh[1]/2 - sh[1]/3):int(sh[1]/2 + sh[1]/3)]

img_edges = img_edges[0:2 ** 9, 0:2 ** 9,:]
result_rsf,a,b = cv2.split(img_edges)
result_rsf = cv2.GaussianBlur(result_rsf,(5,5),cv2.BORDER_DEFAULT)

img = img[0:2 ** 9, 0:2 ** 9,:]

print(img.shape, result_rsf.shape)

model = modelini()
result_rsf = get_model_edges(model, img)
result_rsf = (result_rsf/result_rsf.max())*255

def piupiu(alpha1,alpha2,edges):
      ret, result_bin = cv2.threshold(result_rsf, 255*alpha1, 255, cv2.THRESH_BINARY)
      result_bin = np.uint8(result_bin)
      ret, result_bin2 = cv2.threshold(result_rsf, 255*alpha2, 255, cv2.THRESH_BINARY)
      result_bin2 = np.uint8(result_bin2)

      result_bin = cv2.add(result_bin, edges)
      kernel = np.ones((5, 5), np.uint8)
      result_bin2 = cv2.morphologyEx(result_bin2, cv2.MORPH_CLOSE, kernel)
      edges = cv2.ximgproc.thinning(result_bin)
      result_bin3 = cv2.subtract(result_bin, result_bin2)
      edges[result_bin3 == 0] = 0
      return edges

edges_0 = np.zeros((2 ** 9,2 ** 9), np.uint8)
edges_0_1 = edges_0.copy()
for i in range(6,21):
      edges = piupiu(0.05*i, 0.05*(i+1), edges_0)
      edges_0 = cv2.add(edges_0, edges)

for i in range(3,10):
      edges = piupiu(0.1*(i/3), 0.1*((i/3)+0.5),edges_0_1)
      edges_0_1 = cv2.add(edges_0_1,edges)

ret, result_bin = cv2.threshold(result_rsf, 255*0.2, 255, cv2.THRESH_BINARY)

#

result = edges_0
result2 = edges_0_1
fig = plt.figure(figsize=(10, 10))
ax = [fig.add_subplot(2, 2, 1),
      fig.add_subplot(2, 2, 2),
      fig.add_subplot(2, 2, 3)]
ax[0].imshow(cv2.merge((result_rsf,result_rsf,result_rsf)))
ax[1].imshow(cv2.merge((result, result, result)))
ax[2].imshow(cv2.merge((result2, result2, result2)))
#ax[1].imshow(img)

plt.show()


#cv2.add(255 - self.area_bg, self.edges_line)