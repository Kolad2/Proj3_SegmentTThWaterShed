import os
import sys
import PathCreator
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import pickle
import math
from random import shuffle
from ListFiles import GetFiles
from ShpMaskWriter import mask_write, mask_write_treads
from rsf_edges import modelini, get_model_edges
from CannyTest import cannythresh, cannythresh_grad

class ThinSegmentation:
    def __init__(self, img):

        self.img = img
        self.shape = img.shape
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        l_c, a, b = cv2.split(lab)
        self.img_gray = l_c
        self.area_sure = np.empty(self.shape[0:2], dtype=np.uint8)
        self.area_unknown = np.empty(self.shape[0:2], dtype=np.uint8)
        self.area_dist = None
        self.area_bg = np.empty(self.shape[0:2], dtype=np.uint8)
        self.area_marks = None
        self.edges_w = None

    def watershed_iter(self, area_marks, area_bg=None):
        if area_bg is None:
            area_bg = self.area_bg
        area_marks = area_marks.copy()
        area_marks[area_marks == -1] = 0
        area_marks[area_marks == 1] = 0
        area_marks[area_bg == 0] = 1
        return cv2.watershed(self.img, area_marks)

    def marker_unbound_spread(self):
        self.area_marks = self.watershed_iter(self.area_marks, self.area_bg * 0 + 255)

    def area_marks_edgeupdate(self, area_marks):
        area_marks[area_marks == -1] = 0
        area_marks[area_marks == 0] = 0
        area_marks[area_marks == 1] = 0
        area_marks[self.area_bg == 0] = 1

    def area_marks_summator(self, area_marks_base, area_marks):
        area_marks_base = area_marks_base.copy()
        self.area_marks_edgeupdate(area_marks_base)
        area_marks_base[area_marks > 1] = area_marks[area_marks > 1] + self.area_marks.max()
        return area_marks_base

    def get_marker_from_background_iter(self, area_bg=None):
        if area_bg is None:
            area_bg = self.area_bg.copy()
        else:
            area_bg = area_bg.copy()

        if self.area_marks is not None:
            area_bg[self.area_marks != 1] = 0
            area_marks = self.get_marker_from_background(area_bg)
            self.area_marks = self.area_marks_summator(self.area_marks, area_marks)
            self.area_marks = self.watershed_iter(self.area_marks)
        else:
            area_marks = self.get_marker_from_background(area_bg)
            self.area_marks = np.empty(self.shape[0:2], dtype=np.int32)
            self.area_marks = area_marks

    def get_marker_from_background(self, area_bg=None):
        if area_bg is None:
            area_bg = self.area_bg
        area_marks = np.empty(self.shape[0:2], dtype=np.int32)
        area_dist = np.empty(self.shape[0:2], dtype=np.float32)
        area_dist[:] = cv2.distanceTransform(area_bg, cv2.DIST_L2, 0)
        area_dist = np.uint8((area_dist / area_dist.max()) * 255)
        ret, area_sure = cv2.threshold(
            area_dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, area_marks[:] = cv2.connectedComponents(area_sure)
        area_unknown = cv2.subtract(area_bg, area_sure)
        area_marks = area_marks + 1

        dS: int
        S: int = sum(area_marks[area_marks == 1])
        for i in range(10):
            area_marks = self.watershed_iter(area_marks, area_bg)
            dS = S - sum(area_marks[area_marks == 1])
            S = S - dS

        self.area_dist = area_dist
        self.area_sure[:] = area_sure
        self.area_unknown[:] = area_unknown
        return area_marks

    def get_edge_prob(self):
        if self.edges_w is None:
            self.edges_w = np.empty(self.shape[0:2], dtype=np.float32)
            model = modelini()
            self.edges_w[:] = get_model_edges(model, self.img)
        return self.edges_w.copy()

    def set_edge_prob(self, edges_w):
        self.edges_w = edges_w.copy()

    def get_bg_canny(self):
        self.img_gray = cv2.bilateralFilter(self.img_gray, 15, 40, 80)
        self.area_bg = 255 - cv2.Canny(self.img_gray, 100, 200)
        self.edges_w = self.get_edge_prob()
        self.area_bg[self.edges_w < 0.1] = 255
        self.get_marker_from_background_iter()
        return self.area_marks

    def get_bg_rsfcanny(self):
        #self.img_gray = cv2.bilateralFilter(self.img_gray, 15, 40, 80)
        self.edges_w = self.get_edge_prob()
        self.area_bg = 255 - cannythresh(self.edges_w)
        self.area_bg[self.edges_w < 0.1] = 255
        self.get_marker_from_background_iter()
        return self.area_marks

    def get_bg_rsfcannygrad(self):
        #self.img_gray = cv2.bilateralFilter(self.img_gray, 15, 40, 80)
        self.edges_w = self.get_edge_prob()
        self.area_bg = 255 - cannythresh_grad(self.edges_w, self.img_gray)
        self.area_bg[self.edges_w < 0.1] = 255
        return self.area_marks

    def get_marks_areas(self):
        l = np.max(self.area_marks)
        S = [0]*l
        for i in range(1, l+1):
            print(i)
            S[i-1] = np.sum(self.area_marks == i)
        return S

    def area_threshold(self, th: int):
        S = self.get_marks_areas()
        B = [0 if x <= th else 1 for x in S]
        for i in range(len(S)):
            if B[i] == 0:
                self.area_marks[self.area_marks == i + 1] = 0
        #plt.hist([x for x in S if x < 200], bins=20)
        #plt.show()

    def area_marks_shuffle(self):
        l = np.max(self.area_marks)
        numlist = [x for x in range(2,l+1)]
        shuffle(numlist)
        for i in range(l-1):
            self.area_marks[self.area_marks == i + 2] = numlist[i]

    def get_masks(self):
        masks = []
        area_marks = self.area_marks
        for i in range(area_marks.max()):
            mask = {'segmentation': area_marks == i, 'bbox': (0, 0, 0, 0)}
            segmentation = np.where(mask['segmentation'])
            if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                x_min = int(np.min(segmentation[1]))
                x_max = int(np.max(segmentation[1]))
                y_min = int(np.min(segmentation[0]))
                y_max = int(np.max(segmentation[0]))
                mask['bbox'] = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
                masks.append(mask)
        return masks