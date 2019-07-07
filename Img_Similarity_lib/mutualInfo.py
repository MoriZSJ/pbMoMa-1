from sklearn import metrics as mr
from scipy.misc import imread
import numpy as np
 
img1 = imread('mag_Videos/btfy/btfymag1.jpg')
img2 = imread('Truth_Patches/6-mag.jpg')
 
img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
 
img1 = np.reshape(img1, -1)
img2 = np.reshape(img2, -1)
# print(img2.shape)
# print(img1.shape)
mutual_infor = mr.mutual_info_score(img1, img2)
 
print("mutual_info: "+ str(mutual_infor))
