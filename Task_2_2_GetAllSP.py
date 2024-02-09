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


for FileName in FileNames:
	print("Start", FileName)
	Path_dir = Path0 + "/" + FileName + "/"
	Path_img = Path_dir + "Picture" + "/" + FileName + ".tif"
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
	Path_line = Path_dir + "Lineaments" + "/" + FileName + "_lin_cut.tif"
	img_line = cv2.imread(Path_line)
	result_line,_,_ = cv2.split(img_line)
	TS = ThinSegmentation(img, result_rsf, result_line)
	TS.method2()
	S, P = TS.get_SP()
	dict = {'S': S, 'P': P}
	scipy.io.savemat("temp/" + FileName + "_S.mat", dict)