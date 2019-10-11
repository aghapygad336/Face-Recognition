import cv2
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

img_dir = "D:\PR1\Face-Recognition\ATT" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []

print("Images Loaded")
i=0
for f1 in files:
    i=i+1
    img = cv2.imread(f1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data.append(gray)
    mydata = np.array(data)
    print("Image : \n",mydata)

       
##fixing shapev->400*10304
d=np.reshape(mydata, (400, 10304))
print("Shape od D matrix :",d.shape)