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

for curImage_Name in os.listdir("test_images/"):
	curImage = mpimg.imread('test_images/' + curImage_Name)

	# Convert to grayscale
	curImage_Gray = cv2.cvtColor(curImage, cv2.COLOR_RGB2GRAY)

	# Calibrate the camera based on the previously determined set of object and image points
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, curImage_Gray.shape[::-1], None, None)

	# Undistort each image based on our previous calculations
	curImage_Undistorted = cv2.undistort(curImage, mtx, dist, None, mtx)

	# Convert to HLS color space and separate L and S channels
	curImage_HLS = cv2.cvtColor(curImage_Undistorted, cv2.COLOR_RGB2HLS)
	lightness_channel = curImage_HLS[:,:,1]
	saturation_channel = curImage_HLS[:,:,2]

	# Calculate the directional gradient using the L channel
	sobelx = cv2.Sobel(lightness_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
	scaled_sobel = np.uint8(255 * abs_sobelx/np.max(abs_sobelx))

	# Threshold with colour based on S channel and gradient with L channel
	s_channel_bin = np.zeros_like(saturation_channel) # Create and all black binary image
	s_channel_bin[(saturation_channel >= 145) & (saturation_channel <= 255) | (scaled_sobel >= 20) & (scaled_sobel <= 80)] = 1 # Pixels that meet the thresholds are turned white
	colour_gradient_thershold_bin = np.uint8(255 * s_channel_bin/np.max(s_channel_bin))

	cv2.imwrite(os.path.join('output_images/', 'output_' + curImage_Name), colour_gradient_thershold_bin)
