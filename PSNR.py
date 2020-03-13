from numpy import mean
from math import log10, sqrt
from cv2 import imread
original = imread('data/train/scaling_factor_2/target/tt12.bmp')
contrast = imread('test.jpg',1)
def psnr(img1, img2):
    mse = mean( (img1 - img2) ** 2 )
    if mse == 0:
      return 100
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / sqrt(mse))

d=psnr(original,contrast)
print(d)