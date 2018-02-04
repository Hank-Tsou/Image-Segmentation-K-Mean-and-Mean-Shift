import cv2 # opencv-python
import numpy as np # use numpy library as np for array object
from sklearn.cluster import KMeans # import KMeans function from sklearn.cluster


image = cv2.imread('panda.jpg') # read image
cluster_num = input('cluster number for K-means: ') # input the cluster number

row, col, dep = image.shape # get image shape
img_data = image.reshape((row * col, 3)) # re-shape the image data structure for the K-Mean function

km_output = KMeans(n_clusters=cluster_num).fit_predict(img_data) # Calculate result by use K-mean function
cluster = km_output.reshape([row,col]) # re-shape the data structure

pic_new = np.zeros((row, col, 3), dtype = np.uint8) # create a new empty object for result image
for i in range(row):   
    for j in range(col):
		for k in range(3):
			pic_new[i][j][k] = (255/(cluster[i][j]+1)) # fill in pixels value depend on the k-mean cluster

#cv2.imwrite('3_a_1.jpg', pic_new) # output result image name 3_a.jpg           
cv2.imshow('k_mean', pic_new) # show result image
cv2.waitKey(0)                # system pause         


