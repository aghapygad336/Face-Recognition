# -*- coding: utf-8 -*-
"""lda.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jn4nwJmYzB0Bup_dsuH29Nb823nQSgOI
"""

# !unzip ATT.zip
import cv2
import os
import glob
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def Knn(train_data,train_label,test_data,test_label):
    best_n  = [1,3,5,7]
    score = []
    for i,neighbour in zip(range(len(best_n )),best_n ):
        KnnTest = KNeighborsClassifier(n_neighbors = neighbour, weights = 'distance') 
        KnnTest.fit(train_data.T, train_label) 
        pred = KnnTest.predict(test_data.T)
        score.append(accuracy_score(pred,test_label)) 
        print("Accuracy score is: " + str(score[i]))
        count = 0
        for i in range(len(pred)):
            print("[" + str(i) + "]" + "Classified as: "+ str(pred[i]) +" Actual is: "+ str(test_label[i]))
            
    print("Number of Misclassified is " + str(count))
    plt.plot(score,best_n)
    plt.show()




img_dir = "./ATT"  # Enter Directory of all images
data_path = os.path.join(img_dir, '*g')
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

#prepare the Labels
nameOflabels = []
label=1

for i in range(1,401):

    labelX= str(label)
    nameOflabels.append(labelX)
    z=i%40
    if z<1:

         print("**IF**",label)
         label = label +1
     
         
df = pd.DataFrame(d, index=nameOflabels)
i_train=0
i_test=0
train_split_value = int(d.shape[0]*(5/10))
test_split_value = d.shape[0] - train_split_value
    
train_data = np.zeros((train_split_value,10304))
train_labels = np.zeros((train_split_value,1)) 
    
test_data = np.zeros((test_split_value,10304))
test_labels = np.zeros((test_split_value,1))
for i in range(d.shape[0]):
        #even
    if i%2==0:
       test_data[i_test,:] = d[i]
       test_labels[i_test] = nameOflabels[i]
       i_test+=1
        #odd
    
    else:
      train_data[i_train,:] = d[i]
      train_labels[i_train] = nameOflabels[i]
      print( nameOflabels[i])
      i_train+=1
    # print("Image : \n",mydata)
# fixing shapev->400*10304
d = np.reshape(mydata, (400, 10304))
print("Shape od D matrix :", d.shape)

labels = []
personN=1
for i in range(1,401):

    person= str(personN)
    labels.append(person)
    z=i%10
    if z<1:
        # print("**IF**",personN)
        personN = personN +1

# print(labels)

data = pd.DataFrame(data=d)
data['labels'] = nameOflabels
data.iloc[[0]]

mu_all = [np.mean(data[col]) for col in data.columns[:-1]]
len(mu_all)

from numpy import linalg as LA 


labels = data.labels.unique()
mu = []  
l =0

#means
for i in labels:
    means=[]
    for col in data.columns[:-1]:
        means.append(np.mean(data[data['labels'] == i][col]))
    mu.append(means)
print(len(mu),len(mu[0]))

# b 
i =0
b = np.zeros((10304,10304))

for i in range(40):
    mat_diff = np.reshape(np.subtract(mu[i],mu_all),(10304,1))
    b = np.add(b,10*np.matmul(mat_diff,np.transpose(mat_diff)))
print(b.shape)
# print(np.transpose(matt_diff).shape)

#z[i]
z = []
for i in range(40):

    d = data[data['labels'] == labels[i]]
    d = d.drop('labels',axis=1)

    class_mean = np.reshape(mu[i],(10304,1))
    mean_by_ones_transpose =np.matmul(np.ones((10,1)),np.transpose(class_mean))
    z.append(np.subtract(d,mean_by_ones_transpose))

# print(data.shape)
# print('data shape',d.shape)
# print(mean_by_ones_transpose.shape)
# z = np.array(z)

#s[i]
S = np.zeros((10304,10304))
for i in range(40):
    mat = np.array(z[i])
    output = np.matmul(np.transpose(mat),mat)
    S = np.add(S,output)
print(S.shape)

E_Values_LDA , E_Vectors_LDA  = LA.eigh(np.matmul(LA.inv(S),b))
idx = E_Values_LDA.argsort()[::-1]##
E_Values_Sorted = E_Values_LDA[idx]
E_Vectors_Sorted = E_Vectors_LDA[:,idx]
projMatrix = E_Vectors_Sorted[:,:39]  # first 39 columns 


U_Train_LDA_projection = np.dot(train_data , projMatrix)
U_Test_LDA_projection = np.dot(test_data , projMatrix)

print(U_Train_LDA_projection.shape)
print(U_T_LDA_projection.shape)


Knn(U_Train_LDA_projection,train_labels,U_Test_LDA_projection,test_labels)    



#eigenvalues and eigenvectors
