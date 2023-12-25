import os
import utils
import models
import data_loader
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision


def modelini():
    path = os.path.dirname(os.path.realpath(__file__))
    model = models.RCF()
    utils.load_pretrained(model, path + "/models/RCFcheckpoint_epoch12.pth")
    return model

def image_cv2nn(img):
    img = np.float32(img)
    img = data_loader.prepare_image_cv2(img)
    img = torch.unsqueeze(torch.from_numpy(img).cpu(), 0)
    return img

def image_nn2cv(img):
    return torch.squeeze(img.detach()).cpu().numpy()


def get_model_edges(model,img):
    img_nn = image_cv2nn(img)
    results = model(img_nn)
    return image_nn2cv(results[-1])