# Face-Recognition
 
Face-Recognition using LDA & PCA

# Data Set
I used the ORL dataset which has 10 images per 40 people,Every image is a grayscale image of size 92x112.
For more information about the data set ATT,Non-Face,Nature
https://drive.google.com/open?id=1N1_xV2rIsfcH1JOA1CjzSZudguhj6HdL


# Steps

1-Read the images and convert into a vector of 10304 (92*112) values corresponding to the image size.
2-Split the data to training and testing with a percantage of 50 % for each batch
3-Apply the LDA algorithm steps with a final goal of computing the eigen-values and eigen-vectors
4-Apply the KNN algorithm with different K valuesfor the calssification phase, prediction the values of the test data batch and calculating the accuracy

# Results 
PCA : 95%
![knnaccuracy PCA](https://user-images.githubusercontent.com/46167070/67467615-5b850300-f649-11e9-8114-b8667a2a28f2.PNG)

LDA : we  tried for the values of k and 39 dominant vectors.


