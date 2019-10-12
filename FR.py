import cv2
import os
import glob
import numpy as np
import pandas as pd

img_dir = "D:\PR1\Face-Recognition\ATT" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
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

np.savetxt("bigD.csv", d, delimiter=",")
#prepare the Labels
nameOflabels = []

for i in range(0, 10304):
    personN=1
    person= str(personN)
    nameOflabels.append(person)
    if (personN%40)<1:
            personN = personN +1
    

d = pd.read_csv("bigD.csv", sep=',')
    
xx = d.iloc[:, 0:10304]  # independent columns
xx.columns =nameOflabels
print(xx)