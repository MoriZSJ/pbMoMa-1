import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = 'mag_Videos/square/squmag1.jpg'
imgname2 = 'Truth_Patches/6-mag.jpg'
feature = 'mag_Videos/square/testf.jpg'
out_path = 'mag_Videos/square/testb.jpg'
factor = 0.9
MIN_MATCH_COUNT = 10
ransacReprojThreshold = 4.0

sift = cv2.xfeatures2d.SIFT_create()
img1 = cv2.imread(imgname1,0)
# print(img1.shape)  
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # transform into gray value
kp1, des1 = sift.detectAndCompute(img1,None)   # des = descriptor

img2 = cv2.imread(imgname2,0)
# print(img2.shape)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
kp2, des2 = sift.detectAndCompute(img2,None)  

hmerge = np.hstack((img1, img2)) # horizontal concatenate
cv2.imshow("gray", hmerge) # show
cv2.waitKey(0)

img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255)) # visualize feature points in red circle
img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255)) 
hmerge = np.hstack((img3, img4)) 
cv2.imshow("point", hmerge) 
cv2.imwrite(feature,hmerge)
cv2.waitKey(0)


# BFMatcher
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)

# Flann matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)


# adjust ratio
good = []
for m,n in matches:
    if m.distance < factor*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    ptsA= np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)
    matchMask = status.ravel().tolist()
    # h,w = img1.shape
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv2.perspectiveTransform(pts,H)

    # img2 = cv2.polylines(img2,[np.int32(dst)],True,200,3, cv2.LINE_AA)

else:
    # print("No enough good matches!")
    matchMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchMask, # draw only inliers
                   flags = 2)

print("good: "+str(len(good))+"\nmatch: "+str(len(matches))+"\npercent:"+str(float(len(good))/len(matches)))
img5 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imshow("BFmatch", img5)
cv2.imwrite(out_path,img5)
cv2.waitKey(0)
cv2.destroyAllWindows()