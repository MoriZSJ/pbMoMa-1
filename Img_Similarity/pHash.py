import cv2
from PIL import Image

def phash(img):
    img = img.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, img.getdata()) / 64.
    return reduce(
        lambda x, (y, z): x | (z << y),
        enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())),
        0
    )

def hamming_distance(a, b):
    return bin(a^b).count('1')

def is_imgs_similar(img1,img2):
	return True if hamming_distance(phash(img1),phash(img2)) <= 5 else False

truth = "Truth/11.jpg"
img = "Path/9.jpg"
im = Image.open(img)
Truth = Image.open(truth)
print("pHash: "+str(hamming_distance(phash(im),phash(Truth))))