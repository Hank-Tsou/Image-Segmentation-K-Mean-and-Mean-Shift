import cv2 # opencv-python
import numpy as np # use numpy library as np for array object
from sklearn.cluster import MeanShift, estimate_bandwidth # import MeanShift and estimate_bandwidth function from sklearn.cluster


image = cv2.imread('panda.jpg') # read image
row, col, depth = image.shape   # get image shape

img_data = image.reshape((row * col, 3)) # re-shape the image data structure for the Mean Shift function
bandwidth = estimate_bandwidth(img_data, quantile=.2, n_samples=500) # Estimate bandwidth for Mean Shift function

ms_output = MeanShift(bandwidth, bin_seeding=True).fit(img_data).labels_ # Calculate result by use MeanShift function

cluster = ms_output.reshape([row,col]) # re-shape the data structure


pic_new = np.zeros((row, col, 3), dtype = np.uint8) # create a new empty object for result image
for i in range(row):   
    for j in range(col):
		for k in range(3):
			pic_new[i][j][k] = (255/(cluster[i][j]+1)) # fill in pixels value depend on the Mean Shift cluster
            
#cv2.imwrite('3_b.jpg', pic_new)   # output result image name 3_b.jpg           
cv2.imshow('Mean_Shift', pic_new) # show result image
cv2.waitKey(0)                    # system pause     