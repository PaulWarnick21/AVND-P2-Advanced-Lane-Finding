import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

# TODO If needed turn this calibration into a function

# Initial arrays for camera calibration
objPoints = [] # 3D points in real world space
imgPoints = [] # 2D points in image plane

# Parameters for sliding window approach to determine lane line position
nwindows = 9 # The number of sliding windows
margin = 100 # The width of the windows +/- margin
minpix = 50 # The minimum number of pixels found to recenter window

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
	s_channel_bin[(saturation_channel >= 180) & (saturation_channel <= 240) | (scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1 # Pixels that meet the thresholds are turned white
	colour_gradient_thershold_bin = np.uint8(255 * s_channel_bin/np.max(s_channel_bin))

	'''
	Source and destination points on our image for transforming the prespective
	so that we see the lane from a birds eye view, this is used to calculate
	lane curvature
	'''
	image_size = (colour_gradient_thershold_bin.shape[1], colour_gradient_thershold_bin.shape[0])
	transform_srcPoints = np.float32([
		[115, image_size[1]],
		[580, 450],
		[685, 450],
		[1050, image_size[1]]])
	transform_dstPoints = np.float32([
		[320, image_size[1]],
		[320, 0],
		[950, 0],
		[950, image_size[1]]])

	# Determine the Perspective Transform "M" given the source and destination points
	M = cv2.getPerspectiveTransform(transform_srcPoints, transform_dstPoints)
	# Warp the image based on the above transform
	curImage_Warped = cv2.warpPerspective(colour_gradient_thershold_bin, M, image_size, flags=cv2.INTER_LINEAR)

	cv2.imwrite(os.path.join('output_images/', 'output_' + curImage_Name), curImage_Warped)

'''
	im = plt.imread('test_images/' + curImage_Name)
	implot = plt.imshow(im)
	plt.scatter([180], [image_size[1]])
	plt.scatter([585], [450])
	plt.scatter([700], [450])
	plt.scatter([1130], [image_size[1]])
	plt.show()


	print(curImage_Name)
	plt.plot(histogram)
	print(leftx_base)
	print(rightx_base)
	plt.show()
'''
