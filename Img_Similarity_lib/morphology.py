import cv2
import numpy as np 
img = cv2.imread("Truth_Patches/6.jpg",0)
kernel = np.ones((9,9),np.uint8)  
erosion = cv2.erode(img,kernel,iterations = 1)
# print(erosion.shape)
cv2.imshow("er",erosion)
cv2.imwrite("Truth_Patches/6-mag.jpg",erosion)
cv2.waitKey()