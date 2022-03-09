from psutil import OPENBSD
import cv2
import os
import numpy as np
import pdb

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########
	hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
	lower_orange = np.array([5, 150, 150])  # 22-35 90-100 80-100
	upper_orange = np.array([15, 255, 255])
	# lower_orange = np.array([10, 90, 80])
	# upper_orange = np.array([22, 100,100])
	orange_filter = cv2.inRange(hsv_img,lower_orange, upper_orange)
	#cv2.imshow('orange filter',orange_filter)

	# TODO: note this crop the image and only for line following
	shape = orange_filter.shape
	crop_start = int(5*shape[0]/6)
	for i in range(crop_start,shape[0]):
		orange_filter[i,:] =  0
	for i in range(0,int(shape[0])/2):
		orange_filter[i,:] = 0

	#edit this section to generalize beyond the data set
	#Bitwise-AND mask and original image, note, rn not using the opening at all
	res = cv2.bitwise_and(img,img, mask= orange_filter)
	# cv2.imshow('res',res)
	resBGR = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
	imgray = cv2.cvtColor(resBGR, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 127, 255, 0)

	# print(thresh)
	# #opening: erosion followed by dilation
	kernel = np.ones((5,5))
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	# cv2.imshow('opening',opening)

	new_img, contours,hierarchy = cv2.findContours(opening ,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours.sort(key = cv2.contourArea, reverse = True)
	# print(len(contours))
	#top left coordinate of rectangle is (x,y)
	if (len(contours)==0):
		x,y,w,h = 0, 0, 0, 0
	else:
		x,y,w,h = cv2.boundingRect(contours[0])

	bounding_box = ((x,y),(x+w,y+h))

	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return bounding_box
