# Face-Recognition
 
Face-Recognition using LDA & PCA

# Data Set
I used the ORL dataset which has 10 images per 40 people,Every image is a grayscale image of size 92x112.
For more information about the data set ATT,Non-Face,Nature
https://drive.google.com/open?id=1N1_xV2rIsfcH1JOA1CjzSZudguhj6HdL
![pr1](https://user-images.githubusercontent.com/46167070/67557730-e6cbca80-f715-11e9-8deb-76ea27ee3c2f.PNG)



# Steps

<li>Read the images and convert into a vector of 10304 (92*112) values corresponding to the image size.</li>
<li>Split the data to training and testing with a percantage of 50 % for each batch.</li>
<li>Apply the LDA algorithm steps with a final goal of computing the eigen-values and eigen-vectors</li>
<li>4-Apply the KNN algorithm with different K valuesfor the calssification phase, prediction the values of the test data batch and calculating the accuracy</li>

#Results

![pca](https://user-images.githubusercontent.com/46167070/67558549-79b93480-f717-11e9-9756-20467f8f79e6.PNG)
![LDA](https://user-images.githubusercontent.com/46167070/67561777-bdaf3800-f71d-11e9-9811-f27d51457bcc.png)



