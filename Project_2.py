import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

for imgName in os.listdir("test_images/"):
	curImage = mpimg.imread('test_images/' + imgName)
	curImage_Output = cv2.cvtColor(curImage, cv2.COLOR_RGBA2BGRA)
	cv2.imwrite(os.path.join('output_images/', 'output_' + imgName), curImage_Output)