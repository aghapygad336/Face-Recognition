# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 04:02:20 2019

@author: 20127
"""

import cv2
import os
img_dir = "D:\PR1\ATT"
images = []

def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
load_images_from_folder(img_dir)
print(images)