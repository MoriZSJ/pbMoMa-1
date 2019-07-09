import numpy as np
import cv2
import sys
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt

############ image grayvalued #################
# img_path = "../BA_thesis/BachelorArbeitMaterial/imgs/cranecrop-ori.jpg"
# gray_path = "../BA_thesis/BachelorArbeitMaterial/imgs/cranecrop_ori.jpg"

# imGray = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
# cv2.imshow("gray",imGray)
# cv2.waitKey()
# cv2.imwrite(gray_path,imGray)

############# image invert-color  ################
# img_path = "test/2f.jpg"
# out_path = "test/inv2f.jpg"


# img = Image.open(img_path)
# ivt = PIL.ImageOps.invert(img)
# ivt.save(out_path)

############# paint gray-image with color  ################
img_path = "mag_Videos/baby/invert_0.4-0.8.jpg"
out_path  ="mag_Videos/baby/color_0.4-0.8.jpg"
imGray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# print(imgColor[500,500:900])
imgColor = cv2.applyColorMap(imGray,5)#cv2.COLORMAP_OCEAN
cv2.imshow("color",imgColor)
cv2.waitKey()
cv2.imwrite(out_path,imgColor)

########### matplotlib $###############
# plt.figure(figsize=(1920,1080))
# plt.imshow(imGray,cmap='magma')
# plt.colorbar()
# plt.axis('off')
# plt.savefig(out_path)
# plt.show()

 

