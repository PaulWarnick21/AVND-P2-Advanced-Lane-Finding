import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip

# Initial arrays for camera calibration
objPoints = [] # 3D points in real world space
imgPoints = [] # 2D points in image plane

'''
Source and destination points on our image for transforming the prespective
so that we see the lane from a birds eye view, this is used to calculate
lane curvature
'''
transform_srcPoints = np.float32([
	[115, 720],
	[580, 450],
	[685, 450],
	[1050, 720]])
transform_dstPoints = np.float32([
	[320, 720],
	[320, 0],
	[950, 0],
	[950, 720]])

# Parameters for sliding window approach to determine lane line position
nwindows = 9 # The number of sliding windows
margin = 100 # The width of the windows +/- margin
minpix = 50 # The minimum number of pixels found to recenter window

# Define conversions in x and y from pixels space to meters for lane curvature
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

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

def process_image(curImage):
#for curImage_Name in os.listdir("test_images/"):
#	curImage = mpimg.imread('test_images/' + curImage_Name)

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
	s_channel_bin[(saturation_channel >= 180) & (saturation_channel <= 240) | (scaled_sobel >= 40) & (scaled_sobel <= 80)] = 1 # Pixels that meet the thresholds are turned white
	colour_gradient_thershold_bin = np.uint8(255 * s_channel_bin/np.max(s_channel_bin))

	image_size = (colour_gradient_thershold_bin.shape[1], colour_gradient_thershold_bin.shape[0])

	# Determine the Perspective Transform "M" given the source and destination points
	M = cv2.getPerspectiveTransform(transform_srcPoints, transform_dstPoints)

	# Inverse Perspective Transform for reverting the transformation later
	Minv = cv2.getPerspectiveTransform(transform_dstPoints, transform_srcPoints)

	# Warp the image based on the above transform
	curImage_Warped = cv2.warpPerspective(colour_gradient_thershold_bin, M, image_size, flags=cv2.INTER_LINEAR)

	# Take a histogram of the bottom half of the image
	histogram = np.sum(curImage_Warped[curImage_Warped.shape[0]//2:,:], axis=0)

	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]//2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Set height of windows - based on nwindows above and image shape
	window_height = np.int(curImage_Warped.shape[0]//nwindows)

	# Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
	nonzero = curImage_Warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Current positions to be updated later for each window in nwindows
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = curImage_Warped.shape[0] - (window+1)*window_height
		win_y_high = curImage_Warped.shape[0] - window*window_height

		### Find the four below boundaries of the window ###
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		
		# If you found > minpix pixels, recenter next window (`right` or `leftx_current`) on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices (previously was a list of lists of pixels)
	try:
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)
	except ValueError:
		# Avoids an error if the above is not implemented fully
		pass

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	left_fit_meters = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_meters = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)


	# Generate x and y values for plotting
	ploty = np.linspace(0, curImage_Warped.shape[0]-1, curImage_Warped.shape[0] )
	try:
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	except TypeError:
		# Avoids an error if `left` and `right_fit` are still none or incorrect
		print('The function failed to fit a line!')
		left_fitx = 1*ploty**2 + 1*ploty
		right_fitx = 1*ploty**2 + 1*ploty

	'''
	Calculates the curvature of polynomial functions in meters
	'''

	# Define y-value where we want radius of curvature
	# We'll choose the maximum y-value, corresponding to the bottom of the image
	y_eval = np.max(ploty)

	# The calculation of R_curve (radius of curvature)
	left_curverad = ((1 + (2*left_fit_meters[0]*y_eval*ym_per_pix + left_fit_meters[1])**2)**1.5) / np.absolute(2*left_fit_meters[0])
	right_curverad = ((1 + (2*right_fit_meters[0]*y_eval*ym_per_pix + right_fit_meters[1])**2)**1.5) / np.absolute(2*right_fit_meters[0])
	average_curverad = round((left_curverad + right_curverad) / 2, 2);

	# Create two images for drawing and window selection
	out_img = np.dstack((curImage_Warped, curImage_Warped, curImage_Warped))
	window_img = np.zeros_like(out_img)

	# Colour in the lane line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Create an image to draw on
	curImage_Warped_Zero = np.zeros_like(curImage_Warped).astype(np.uint8)
	coloured_Warp = np.dstack((curImage_Warped_Zero, curImage_Warped_Zero, curImage_Warped_Zero))

	# Cast the points into usable format for fillPoly() and crop the bottom to avoid highlighting the hood of the vehicle
	left_points = np.array([np.transpose(np.vstack([left_fitx[0:710], ploty[0:710]]))])
	right_points = np.array([np.flipud(np.transpose(np.vstack([right_fitx[0:710], ploty[0:710]])))])
	points = np.hstack((left_points, right_points))

	# Draw the lane onto our image
	cv2.fillPoly(coloured_Warp, np.int_([points]), (0,255, 0))

	# Invert the transformation to the original perspective
	curImage_Lane_Boundaries = cv2.warpPerspective(coloured_Warp, Minv, (curImage.shape[1], curImage.shape[0])) 
	# Combine the result with the original image
	curImage_Final = cv2.addWeighted(curImage, 1, curImage_Lane_Boundaries, 0.3, 0)

	# Determine the cars offset in meters when compared to the center of the lane
	laneCenter = ((right_fitx[image_size[1]-1] - left_fitx[image_size[1]-1]) / 2) + left_fitx[image_size[1]-1]
	carPosition = image_size[0]/2
	offset = round((carPosition - laneCenter)*xm_per_pix, 2)

	# Add text and display
	average_rad_string = "Radius of Curvature: %.2f m" % average_curverad
	offset_string = "Offset from Center: %.2f m" % offset

	cv2.rectangle(curImage_Final, (80, 40), (1220, 180), (255, 0, 0), -1) 
	cv2.putText(curImage_Final,average_rad_string , (100, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), thickness=2)
	cv2.putText(curImage_Final, offset_string, (100, 160), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), thickness=2)

	# Fix colour and save to files
	#curImage_Final_Colour = cv2.cvtColor(curImage_Final, cv2.COLOR_BGR2RGB)
	#cv2.imwrite(os.path.join('output_images/', 'output_' + curImage_Name), curImage_Final_Colour)

	return curImage_Final

# Find lines on project video
clip = VideoFileClip("project_video.mp4")
white_clip = clip.fl_image(process_image)
white_clip.write_videofile('test_videos_output/project_video_result.mp4', audio=False)
