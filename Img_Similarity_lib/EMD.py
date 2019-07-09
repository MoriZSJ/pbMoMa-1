import cv2
import numpy as np 
import matplotlib.pyplot as plt

def img_to_sig(arr):
    """Convert a 2D array to a signature for cv2.EMD"""
    
    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i,j]/arr.sum(), i, j])
            count += 1
    return sig


img1='mag_Videos/tri/magtri2.jpg'
img2='Truth_Patches/13.jpg'

im1=cv2.imread(img1,0)
im2=cv2.imread(img2,0)
# print(im1.shape)

h1 = cv2.calcHist([im1],[0],None,[256],[0,256])
h2 = cv2.calcHist([im2],[0],None,[256],[0,256])
# plt.plot(h2)
# plt.show()
# print(h1.shape)
# print(h1.sum())
sig1 = img_to_sig(h1)
sig2 = img_to_sig(h2)
# print(sig1)

dis,_,_ = cv2.EMD(sig1,sig2,cv2.DIST_L2)

print(dis)