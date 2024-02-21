import numpy as np
import PathCreator
import matplotlib.pyplot as plt
import cv2
from Shpreader import get_shp_poly
from Bresenham_Algorithm import line as line_BA
import pickle

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

FileNames = ["B21-188b_2",
             "64-3",        #9
             "15-2",     #15
             "B21-200b",    #16
             "B21-192b"]       #18

for FileName in FileNames:
    print(FileName)
    Path_dir = Path0 + "/" + FileName + "/"
    Path_shape = Path_dir + "Joined/" + FileName + "_joined"
    poly = get_shp_poly(Path_shape)

    Path_img = Path_dir + "Picture" + "/" + FileName  + ".tif"
    img = cv2.imread(Path_img)
    sh = img.shape

    img_lines = np.zeros(sh[0:2], dtype=np.uint8)
    img_lines = cv2.polylines(img_lines, poly, False, 255, 1)
    img_lines = img_lines[
                int(sh[0]/2 - sh[0]/3):int(sh[0]/2 + sh[0]/3),
                int(sh[1]/2 - sh[1]/3):int(sh[1]/2 + sh[1]/3)]
    cv2.imwrite(Path_dir + "Lineaments/" + FileName + "_lin_cut.tif", img_lines)