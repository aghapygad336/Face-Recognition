import numpy as np
import os
import glob
import cv2
import pandas as pd

class LDA:
    def LDA(self,data):
        labels = data.labels.unique()
        feature_mean = []
        for i in labels:
            means = [np.mean(data[data['labels'] == i].drop('labels',axis=1)[col]) for col in data.columns]
            feature_mean = [means]
        print(len(feature_mean),len(feature_mean[0]))
        # mean1 = np.mean(data[data[2] == 1][[0, 1]])
        # mean2 = np.mean(data[data[2] == -1][[0, 1]])
        # b = np.matmul((np.subtract(mean1, mean2)),
        #               np.transpose((np.subtract(mean1, mean2))))
        # z = []S
        # z.append(np.subtract(data[data[2] == 1][[0, 1]], np.matmul(
        #     np.transpose(mean1), [1, 1, 1, 1, 1])))
        # z.append(np.subtract(data[data[2] == -1][[0, 1]],
        #                      np.matmul(np.transpose(mean2), [1, 1, 1, 1, 1])))
        # s = []
        # s.append(np.matmul(np.transpose(z[[0, 1]]), z[[0, 1]]))
        # s.append(np.matmul(np.transpose(z[1]), z[1]))

        # s_tot = np.add([0], s[1])

        # values, vectors = np.linalg.eig(np.array(np.linalg.inv(s_tot))*b)


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
data['labels'] = labels

ld = LDA()
ld.LDA(data)
