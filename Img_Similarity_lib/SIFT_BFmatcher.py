import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = 'magVid_matlab/btfy/magKCF-btfy1.jpg'
imgname2 = 'Truth_Patches/11-mag.jpg'
feature = 'magVid_matlab/btfy/mor/matbtfyfea09.jpg'
out_path = 'magVid_matlab/btfy/mor/matbtfyBF09.jpg'
factor = 0.9

sift = cv2.xfeatures2d.SIFT_create()

img1 = cv2.imread(imgname1)
print(img1.shape)  
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # transform into gray value
kp1, des1 = sift.detectAndCompute(img1,None)   # des = descriptor

img2 = cv2.imread(imgname2)
print(img2.shape)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
kp2, des2 = sift.detectAndCompute(img2,None)  

hmerge = np.hstack((gray1, gray2)) # horizontal concatenate
cv2.imshow("gray", hmerge) # show
cv2.waitKey(0)

img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255)) # visualize feature points in red circle
img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255)) 
hmerge = np.hstack((img3, img4)) 
cv2.imshow("point", hmerge) 
cv2.imwrite(feature,hmerge)
cv2.waitKey(0)
# BFMatcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
# adjust ratio
good = []
for m,n in matches:
    if m.distance < factor*n.distance:
        good.append([m])

print("good: "+str(len(good))+"\nmatch: "+str(len(matches))+"\npercent:"+str(float(len(good))/len(matches)))
img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
cv2.imshow("BFmatch", img5)
cv2.imwrite(out_path,img5)
cv2.waitKey(0)
cv2.destroyAllWindows()