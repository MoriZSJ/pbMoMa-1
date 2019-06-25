import cv2
import numpy as np
import matplotlib.pyplot as plt

img = np.array(cv2.imread("mag_Videos/btfy/tck1/btfy1.jpg"))
saveImg = "mag_Videos/btfy/tck1/btfymag1.jpg"

# =======find center of path patch
y, x, z = np.where(img==[0,0,0])
#print(x,y)
center = [int(np.mean(x)),int(np.mean(y))]
print(center)

# =======locate origin patch
patchsize = 20
p1 = [center[0]-patchsize,center[1]-patchsize]
p2 = [center[0]+patchsize,center[1]-patchsize]
p3 = [center[0]-patchsize,center[1]+patchsize]
p4 = [center[0]+patchsize,center[1]+patchsize]
pts1 = np.float32([p1,p2,p3,p4])
print(p1,p2,p3,p4)
# =======transformed patch size
transsize = 320
pts2 = np.float32([[0,0],[transsize,0],[0,transsize],[transsize,transsize]])

# =======transform
M = cv2.getPerspectiveTransform(pts1,pts2)
res = cv2.warpPerspective(img,M,(320,320)) 
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(res)
plt.show()
cv2.imwrite(saveImg,res)

