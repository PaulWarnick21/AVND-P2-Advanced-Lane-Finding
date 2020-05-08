import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

# TODO If needed turn this calibration into a function

# Initial arrays for camera calibration
objPoints = [] # 3D points in real world space
imgPoints = [] # 2D points in image plane

# Camera Calibration
for curImage_Name in os.listdir("camera_cal/"):
	curImage_Cali = mpimg.imread('camera_cal/' + curImage_Name)

	# Number of chessboard corners in x and y direction
	cornerCount_x = 9
	cornerCount_y = 6

	# Prepare object points
	objP = np.zeros((cornerCount_y*cornerCount_x, 3), np.float32)
	objP[:,:2] = np.mgrid[0:cornerCount_x,0:cornerCount_y].T.reshape(-1,2)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(curImage_Cali, (cornerCount_x, cornerCount_y), None)

	'''
	If found, draw corners. Certain images in the camera_cal directory will return false.
	This highlights the need to calibrate with a group of photos.
	'''
	if ret == True:
		imgPoints.append(corners)
		objPoints.append(objP)

for curImage_Name in os.listdir("camera_cal/"):
	curImage_Cali = mpimg.imread('camera_cal/' + curImage_Name)

	# Convert to grayscale
	curImage_Gray = cv2.cvtColor(curImage_Cali, cv2.COLOR_RGB2GRAY)

	# Calibrate the camera based on the previously determined set of object and image points
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, curImage_Gray.shape[::-1], None, None)

	# Undistort each image based on our previous calculations
	curImage_Undistorted = cv2.undistort(curImage_Cali, mtx, dist, None, mtx)
	cv2.imwrite(os.path.join('output_images/', 'output_' + curImage_Name), curImage_Undistorted)

'''
for imgName in os.listdir("test_images/"):
	curImage = mpimg.imread('test_images/' + imgName)
	curImage_Output = cv2.cvtColor(curImage, cv2.COLOR_RGBA2BGRA)
	cv2.imwrite(os.path.join('output_images/', 'output_' + imgName), curImage_Output)
'''
