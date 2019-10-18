import numpy as np
import os
import glob
import cv2
import pandas as pd

from numpy import linalg as LA


def LDA(data):
    labels = data.labels.unique()
    feature_mean_for_each_label = []
    l = 0

    # means
    for i in labels:
        means = []
        for col in data.columns[:-1]:
            means.append(np.mean(data[data['labels'] == 1][col]))
        feature_mean_for_each_label.append(means)
    print(len(feature_mean_for_each_label),
          len(feature_mean_for_each_label[0]))

    # b
    temp = []
    i = 0
    for i in range(40):
        for k in range(40):
            temp.append(10*np.matmul((np.subtract(feature_mean_for_each_label[i], feature_mean_for_each_label[k])),
                                     np.transpose(np.subtract(feature_mean_for_each_label[i], feature_mean_for_each_label[k]))))
        print(i+1/40)
        i = i + 1
    B = np.sum(temp)
    print(B)
    '''
    #z[i]
    z = []
    for i in range(40):
        z.append(np.subtract(data[data['labels']==i][:-1],np.transpose(feature_mean_for_each_label[i])))

    print(z[0])         
                
                
    #s[i]
    s = []
    for i in range(40):
        s.append(np.matmul(z[i],np.transpose(z[i])))
    S = np.sum(s)

    #eigenvalues and eigenvectors
    eigenvalues,eigenvectors = LA.eigh(np.matmul(LA.inv(S),B))
                
        '''


img_dir = "./ATT"  # Enter Directory of all images
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)
data = []
i = 0
for f1 in files:
    i = i+1
    img = cv2.imread(f1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data.append(gray)
    mydata = np.array(data)
    # print("Image : \n",mydata)
# fixing shapev->400*10304
d = np.reshape(mydata, (400, 10304))
print("Shape od D matrix :", d.shape)

labels = []
personN = 1
for i in range(1, 401):

    person = str(personN)
    labels.append(person)
    z = i % 10
    if z < 1:
        # print("**IF**",personN)
        personN = personN + 1

# print(labels)

data = pd.DataFrame(data=d)
data['labels'] = labels

ld = LDA()
ld.LDA(data)
