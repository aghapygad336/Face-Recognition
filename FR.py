import cv2
import os
import glob
import numpy as np
import pandas as pd


def findAlpha (a,alpha,EigenValuesSorted):
    for i in range(a,10304):
        #Find the Variance 
        B = float(sum(EigenValuesSorted))
        T = float(sum(EigenValuesSorted[:i]))
        if(T/B >= alpha):
            return i


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

#prepare the Labels
nameOflabels = []
personN=1

for i in range(1,401):

    person= str(personN)
    nameOflabels.append(person)
    z=i%40
    if z<1:

         print("**IF**",personN)
         personN = personN +1
     
         
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
      
      
print ("*******************PCA****************")      
#mean
from numpy import linalg as LA 
      
mean_train_data = np.mean(train_data, axis=0).reshape(10304,1)
print
mean_test_data = np.mean(test_data, axis=0).reshape(10304,1)
#Z=data -one.mean
z_Train = train_data - np.ones((200,1)).dot(mean_train_data.T)
z_Test = test_data - np.ones((200,1)).dot(mean_test_data.T)

#covariance matrix
covarianceMatrix_Training = (1/len(z_Train))*np.matmul(np.transpose(z_Train),z_Train)
covarianceMatrix_Testing = (1/len(z_Test))*np.matmul(np.transpose(z_Test),z_Test)

eigenValues_train_data,eigenVectors_train_data = LA.eigh(covarianceMatrix_Training)
eigenValues_test_data,eigenVectors_test_data = LA.eigh(covarianceMatrix_Testing)  
#step Sorting using the train data 
  


idx = eigenValues_train_data.argsort()[::-1]   
eigenValuesSorted = eigenValues_train_data[idx]
eigenVectorsSorted = eigenVectors_train_data[:,idx]

#type(eigenVectorsSorted)
#Out[3]: numpy.ndarray


#select largest 
chosen_Alpha= [0.8,0.85,0.9,0.95]
for a in chosen_Alpha:
    W= findAlpha(0,a,eigenValuesSorted)
    NewW = eigenVectorsSorted[: , 0 : W + 1]
    wTrain = np.dot(NewW.T , z_Train.T)
    wTest = np.dot(NewW.T , z_Test.T)
    print("For Alpha: " + str(a))
    
    



