import os
import sys
import PathCreator
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import time
import pickle
import math
from ListFiles import GetFiles
from ShpMaskWriter import mask_write, mask_write_treads
from rsf_edges import modelini, get_model_edges
from CannyTest import cannythresh, cannythresh_grad
from ThinSegmentation import ThinSegmentation
from scipy.spatial import cKDTree, KDTree
from Shpreader import get_shp_poly
import random

Path0 = "/media/kolad/HardDisk/ThinSection"

FileNames = ["B21-234a",    #0
             "B21-215b",    #1
             "B21-215a",    #2
             "B21-213b",    #3
             "B21-213a",    #4
             "B21-189b",    #5
             "B21-188b_2",  #6
             "B21-151b",    #7
             "B21-107b",    #8
             "64-3",        #9
             "15-2",        #10
             "15-1",        #11
             "B21-166b",    #12
             "B21-122a",    #13
             "B21-120a",    #14
             "B21-51a",     #15
             "B21-200b",    #16
             "B21-192b",    #17
             "19-5b"]       #18

n = 10
method = 1
for FileName in FileNames:
	Path_dir = Path0 + "/" + FileName + "/"
	Path_img = Path_dir + "Picture" + "/" + FileName + ".tif"
	# image loading
	img = cv2.imread(Path_img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	sh = img.shape
	img = img[
	      int(sh[0] / 2 - sh[0] / 3):int(sh[0] / 2 + sh[0] / 3),
	      int(sh[1] / 2 - sh[1] / 3):int(sh[1] / 2 + sh[1] / 3)]
	#img = img[0:2 ** n, 0:2 ** n]
	summask = np.zeros((img.shape[0],img.shape[1]))
	intermax = 500
	S = []
	P = []
	for i in range(0, intermax, 1):
		print(FileName, i)
		# RSF_result load
		Path_rsf = Path_dir + "RSF_edges" + "/" + FileName + "_edges_cut.tif"
		img_rsf = cv2.imread(Path_rsf)
		result_rsf,_,_ = cv2.split(img_rsf)
		# Lineaments load
		Path_shape = Path_dir + "Joined/" + FileName + "_joined"
		polys = get_shp_poly(Path_shape)
		result_line = np.zeros(sh[0:2], dtype=np.uint8)
		#print(len(polys))
		polys2 = []
		for poly in polys:
			#L = 0
			#for j in range(1,len(poly)):
			#	L = L + math.sqrt(sum((poly[j] - poly[j-1])**2))
			if random.randint(0, 19) > 6:
				polys2.append(poly)

		result_line = cv2.polylines(result_line, polys2, False, 255, 3)
		result_line = result_line[
		            int(sh[0] / 2 - sh[0] / 3):int(sh[0] / 2 + sh[0] / 3),
		            int(sh[1] / 2 - sh[1] / 3):int(sh[1] / 2 + sh[1] / 3)]
		#result_rsf = result_rsf[0:2 ** n, 0:2 ** n]
		#result_line = result_line[0:2 ** n, 0:2 ** n]
		TS = ThinSegmentation(img, result_rsf, result_line)
		TS.method2()
		lS, lP = TS.get_SP()
		S.append(lS)
		P.append(lP)

		#mask = TS.area_marks
		#mask[mask == -1] = 0
		#mask[mask > 0] = 1
		#summask = summask + mask

	#summask = summask/intermax
	#summask2 = summask.copy()
	#summask2[summask2 > 0.5] = 1
	#summask2[summask2 <= 0.5] = 0
	#summask2 = np.uint8((summask2 / summask2.max()) * 255)
	#kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
	#summask2 = cv2.erode(summask2, kernel, iterations=1)
	#ret, area_marks = cv2.connectedComponents(summask2)
	#
	#mask_write_treads("Shapes/Shape_" + FileName + "/Shape_" + str(method) + "_" + FileName + "/Shape_" + str(method) + "_" + FileName, TS.get_masks(area_marks))
	#cv2.imwrite("Shapes/Shape_" + FileName + "/img_" + FileName + ".tif", img)
	dict = {'S': S, 'P': P}
	if not os.path.exists("temp/StatisticData/" + FileName):
		os.mkdir("temp/StatisticData/" + FileName)
	scipy.io.savemat("temp/StatisticData/" + FileName + "/" + FileName + "_" + str(method) + "_S.mat", dict)

"""fig = plt.figure(figsize=(10, 10))
	ax = [fig.add_subplot(2, 2, 1),
	      fig.add_subplot(2, 2, 2),
	      fig.add_subplot(2, 2, 3)]
	ax[0].imshow(summask)
	ax[1].imshow(summask2)
	print(np.unique(area_marks))
	ax[2].imshow(area_marks)
	plt.show()"""