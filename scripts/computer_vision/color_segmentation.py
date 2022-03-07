import cv2
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

def cd_color_segmentation(img, template):
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

	lower_orange = np.array([20,50,50])
	upper_orange = np.array([40,255,255])
	orange_filter = cv2.inRange(img,lower_orange, upper_orange)
	#cv2.imshow('orange filter',orange_filter)

	#opening: erosion followed by dilation
	#kernel = np.ones(5,5)
	#opening = cv2.morphologyEx(orange_filter, cv2.MORPH_OPEN, kernel)
	#cv2.imshow('opening',opening)


	# Bitwise-AND mask and original image, note, rn not using the opening at all
	res = cv2.bitwise_and(img,img, mask= orange_filter)
	cv2.imshow('res',res)
	contours,hierarchy = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours.sort(key = cv2.contourArea, reverse = True)

	#top left coordinate of rectangle is (x,y)
	x,y,w,h = cv2.boundingRect(contours[0])

	bounding_box = ((x,y),(x+w,y+h))

	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return bounding_box
