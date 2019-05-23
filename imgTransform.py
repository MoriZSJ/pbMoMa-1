import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Path/11amp.jpg")
saveImg = "Path/11amptrans.jpg"

rows, cols = img.shape[:2]
pts1 = np.float32([[152,310],[182,310],[152,340],[182,340]])
pts2 = np.float32([[0,0],[320,0],[0,320],[320,320]])
M = cv2.getPerspectiveTransform(pts1,pts2)
res = cv2.warpPerspective(img,M,(320,320)) 
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(res)
plt.show()
cv2.imwrite(saveImg,res)


