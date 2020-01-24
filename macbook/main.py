### Imports ###
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import numpy.polynomial.polynomial as poly
import statistics
import time
import math
from aiohttp import web
import socketio
from flask import Flask, render_template
from flask_socketio import send, emit, SocketIO
from classes.video_capture import VideoCapture 

### Flask Setup ###
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

### Settings & Vars ###
scale_down_factor = 1
dilate_kernel = np.ones((5,5), np.uint8)
prev_mid_points = {}
lookahead_distance = 0.01  # Lookahead 30% of img
# Saving images
image_increment = 1

### Windows ###
cv.namedWindow("output",cv.WINDOW_NORMAL)
cv.resizeWindow("output", 500, 300)
cv.moveWindow("output", 0, 500)
cv.namedWindow("input",cv.WINDOW_NORMAL)
cv.resizeWindow("input", 500,300)
cv.moveWindow("input", 500, 500)

### Video Capture From Camera ###
cap = VideoCapture("http://192.168.1.246:8081")

### Setup For Speed Calculation ###
frame1 = cap.read()
prev_frame = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)

### Calculating Functions ###
def calculate_speed(prev_frame, curr_frame):
	"""
	Description:
		Finds speed that the car is moving using image flow. Takes this frame
		and previous frame to figure out speed moving forwards.
	"""
	flow = cv.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 30, 3, 5, 1.2, 0)
	mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
	mag = mag.flatten()
	ang = ang.flatten()
	flow = np.vstack((mag, ang)).T
	flow = [x for x in flow if (x[1]>math.pi/2-1 and x[1]<math.pi/2+1)]
	speed = float(np.average([x[0] for x in flow]))
	return speed

def find_lane_lines(img, prev_mid_points):
	"""
	Description:
		This function takes in a 1 bit black and white image from the camera and 
		finds the location of the lane lines. 

		First, we find the start_search_point, which is where we start searching 
		for the lane lines. We go from this start_search_point outwards to the 
		left and right to find the lane lines. If it is the bottom row, the 
		start_search_point is the bottom lane midpoint for the last frame. (If
		the last frame doesnt have a lane midpoint, we just make start_search_point
		the middle of the image). If it is not the bottom row, we derive 
		the start_search_point from the row below's lane lines.

		We then start from the start_search_point and go to the right to try
		find the right lane. We do the same but to the left to find the left
		lane. We add the left and right side of each lane line to 
		****_lane_sides. If we dont know a lane side, we add add None instead
		*and stop searching for that lane line in the rows above.
		
		We then use these lane sides to find the middle of each lane line, and
		add it to *****_lane_mid_points. 

		Finally, we extrapolate unknown lane mid points at the bottom of the 
		image (if we the lane line is cut off slightly). Also, if we cant find
		and entire lane line, we pretend that it is at the very edge of the
		image. For example if we cant find the right lane, we pretend the right 
		lane is at the very right of the image.

		You better appreciate these comments because they tooke forever to write
	"""

	### Settings ###

	# If only one lane line is known, the start_search_point will 
	# be derived from that one known lane line. This value is how much 
	# to offset the start_search_point from that known lane line
	offset_from_known_lane_line = img.shape[1]/4

	right_lane_mid_points = {}
	left_lane_mid_points = {}
	right_lane_ended = False
	left_lane_ended = False

	# For each row of img from bot to top
	for row in reversed(range(img.shape[0])):
		# Side of each lane
		right_lane_sides = [] # [left side, right side]
		left_lane_sides = [] # [right side, left side]

		# Row as a python list
		row_as_list = img[row].tolist()

		# If this is the bottom row, start search point is last 
		# frames first mid point if it exists. If it doesnt, it
		# defaults to the center of the image.
		if row == img.shape[0]-1:
			if img.shape[0]-1 in prev_mid_points:
				start_search_point = int(prev_mid_points[img.shape[0]-1])
			else:
				start_search_point = int(img.shape[1]//2)

		# If is not bottom row, derive start_search_point from 
		# the below row's lane lines.
		else:
			# If row below has both lane's mid points
			if row+1 in right_lane_mid_points and row+1 in left_lane_mid_points:
				# Make start search point the middle of below row's lane lines
				start_search_point = int((right_lane_mid_points[row+1]+left_lane_mid_points[row+1])/2)

			# If row below only has one line, use offset_from_known_lane_line
			elif row+1 in right_lane_mid_points:
				start_search_point = int(right_lane_mid_points[row+1]-offset_from_known_lane_line)
			elif row+1 in left_lane_mid_points:
				start_search_point = int(left_lane_mid_points[row+1]+offset_from_known_lane_line)


		# If the start_search_point is outside of img, stop.
		if start_search_point >= img.shape[1] or start_search_point < 0:
			break

		### Right Lane ###
		# If start_search_point is currently on a lane marking
		on_lane = img[row][start_search_point].all() 
		if on_lane:
			# Put start_search_point as left side of right lane
			right_lane_sides.append(start_search_point)

		# From start_search_point to right side of img
		for col in range(start_search_point, img.shape[1]):
			# If previously not on lane, but now is, set on_lane
			# to True and enter col into right_lane_sides as the
			# left side of the right lane.
			if not on_lane and row_as_list[col] == [255,255,255]:
				on_lane = True
				right_lane_sides.append(col)

			# If previously was on lane, but now isn't, put this col
			# into right_lane_sides as right side of right lane and
			# stop searching for right lane.
			elif on_lane and not row_as_list[col] == [255,255,255]:
				right_lane_sides.append(col)
				break

		# Add none to right lane sides if point was not found
		for x in range(0, 2-len(right_lane_sides)):
			right_lane_sides.append(None)

		### Left Lane ###
		# If start_search_point is currently on a lane marking
		on_lane = img[row][start_search_point].all() # If it is currently on a lane marking
		if on_lane:
			# Put start_search_point as left side of right lane
			left_lane_sides.append(start_search_point)

		# From start_search_point to left side of img
		for col in reversed(range(0, start_search_point)):
			# If previously not on lane, but now is, set on_lane
			# to True and enter col into left_lane_sides as the
			# right side of the left lane.
			if not on_lane and row_as_list[col] == [255,255,255]:
				on_lane = True
				left_lane_sides.append(col)

			# If previously was on lane, but now isn't, put this col
			# into left_lane_sides as left side of left lane and
			# stop searching for left lane.
			elif on_lane and not row_as_list[col] == [255,255,255]:
				left_lane_sides.append(col)
				break

		# Add none to left lane sides if point was not found
		for x in range(0, 2-len(left_lane_sides)):
			left_lane_sides.append(None)

		# If right lane has not ended and we know the right lane sides for this 
		# row, calculate right lane's mid point for this row and put it into 
		# right_lane_mid_points.
		if not right_lane_ended and right_lane_sides[0] != None and right_lane_sides[1] != None:
			right_lane_mid_points[row] = (right_lane_sides[0]+right_lane_sides[1])/2
			img[row][int((right_lane_sides[0]+right_lane_sides[1])/2)] = (255, 0, 0)
		# If the right lane has ended, or we dont know the right lane sides, set
		# right_lane_ended to True.
		elif len(right_lane_mid_points) != 0:
			right_lane_ended = True

		# If left lane has not ended and we know the left lane sides for this 
		# row, calculate left lane's mid point for this row and put it into 
		# left_lane_mid_points.
		if not left_lane_ended and left_lane_sides[0] != None and left_lane_sides[1] != None:
			left_lane_mid_points[row] = (left_lane_sides[0]+left_lane_sides[1])/2
			img[row][int((left_lane_sides[0]+left_lane_sides[1])/2)] = (255, 0, 0)
		# If the left lane has ended, or we dont know the left lane sides, set
		# left_lane_ended to True.
		elif len(left_lane_mid_points) != 0:
			left_lane_ended = True

		# Draw lane sides in yellow
		for lane_side in right_lane_sides+left_lane_sides:
			if lane_side == None:
				continue
			img[row][lane_side] = (255, 255, 0)

	### Extrapolate unknown points at bottom of image ###
	# At the bottom of the image, if the lane line is half cut off, no mid point
	# is added. This adds a straight line to fill in those unknown points

	# If right_lane_mid_points is not empty
	if len(right_lane_mid_points.keys()) != 0:
		# First row that is known
		first_known_row = max(right_lane_mid_points, key=int)
		# If we know the mid point for the row 5 above the first one
		if first_known_row-5 in right_lane_mid_points:
			# Find slope 
			slope = (right_lane_mid_points[first_known_row]-right_lane_mid_points[first_known_row-5])/5
			
			# From first known row down to bottom of img, add extrapolated points
			# to right_lane_mid_points
			for row in range(first_known_row, img.shape[0]):
				col = (row-first_known_row)*slope+right_lane_mid_points[first_known_row]
				if col >= img.shape[1] or col < 0:
					break
				right_lane_mid_points[row] = col
				img[row][int(col)] = (255, 0, 0)

	# If we dont know any of the right lane
	else:
		# For each row in the image, set the right_lane_mid_point to the right
		# side of the image
		for row in range(img.shape[0]):
			right_lane_mid_points[row] = img.shape[1]-1

	# If left_lane_mid_points is not empty
	if len(left_lane_mid_points.keys()) != 0:
		# First row that is known
		first_known_row = max(left_lane_mid_points, key=int)
		# If we know the mid point for the row 5 above the first one
		if first_known_row-5 in left_lane_mid_points:
			# Find slope 
			slope = (left_lane_mid_points[first_known_row]-left_lane_mid_points[first_known_row-5])/5
			
			# From first known row down to bottom of img, add extrapolated points
			# to left_lane_mid_points
			for row in range(first_known_row, img.shape[0]):
				col = (row-first_known_row)*slope+left_lane_mid_points[first_known_row]
				if col >= img.shape[1] or col < 0:
					break
				left_lane_mid_points[row] = col
				img[row][int(col)] = (255, 0, 0)
	
	# If we dont know any of the left lane
	else:
		# For each row in the image, set the left_lane_mid_point to the left
		# side of the image
		for row in range(img.shape[0]):
			left_lane_mid_points[row] = 0

	return [right_lane_mid_points, left_lane_mid_points]

def lane_line_of_best_fit(lane_line_mid_points, img):
	"""
	Description:
		Finds line of best fit function for lane lines using lane line mid 
		points.
	"""
	x = []
	y = []

	# Turn dict into array of x and y
	for row, col in lane_line_mid_points.items():
		x.append(row)
		y.append(col)

	z = np.polyfit(x, y, 2)
	best_fit_function = np.poly1d(z)

	# Draw function onto image
	
	for row in range(max(lane_line_mid_points, key=int)):
		col = int(best_fit_function(row))
		if col < 0 or col >= img.shape[1]:
			continue
		img[row][col] = (0, 255, 0)
	

	return best_fit_function

def find_mid_points(img, right_lane_mid_points, left_lane_mid_points, right_best_fit_function, left_best_fit_function):
	"""
	Description:
		Finds mid point of entire lane. 
	"""

	mid_points = {}

	# For row from top to bottom 
	for row in range(img.shape[0]):

		# If this row is below last known col of either lane, extrapolate the
		# mid points with straight line
		if row > max(right_lane_mid_points, key=int) or row > max(left_lane_mid_points, key=int):
			last_known_row = row-1

			# If we dont know the mid point of the row 20 above the last known
			# row, break and give up
			if last_known_row-20 not in mid_points:
				break

			# Find slope
			slope = (mid_points[last_known_row]-mid_points[last_known_row-20])/20
			
			# From last known row down to bottom if image, add in extrapodated
			# mid points.
			for row in range(last_known_row, img.shape[0]):
				col = (row-last_known_row)*slope+mid_points[last_known_row]
				if col >= img.shape[1] or col < 0:
					break
				mid_points[row] = col
				img[row][int(col)] = (255, 0, 0)

			break

		# If actual value for either line does not exist, use line of best fit
		if row in right_lane_mid_points:
			right_mid_point = right_lane_mid_points[row]
		else:
			right_mid_point = right_best_fit_function(row)
		
		if row in left_lane_mid_points:
			left_mid_point = left_lane_mid_points[row]
		else:
			left_mid_point = left_best_fit_function(row)

		mid_point = int((right_mid_point+left_mid_point)/2)

		# If mid point is outside of image, ignore it
		if mid_point >= img.shape[1] or mid_point < 0:
			continue

		mid_points[row] = mid_point
		img[row][mid_point] = (0, 0, 255)

	return mid_points

def find_turn_value(mid_points, lookahead_distance_pixels):
	### Create line of best fit for mid_points ###
	x = []
	y = []

	# Turn dict into array of x and y
	for row, col in mid_points.items():
		x.append(row)
		y.append(col)

	z = np.polyfit(x, y, 3)
	mid_points_best_fit_function = np.poly1d(z)
	f_der = np.polyder(mid_points_best_fit_function)

	# Draw function onto image
	"""
	for row in range(max(mid_points, key=int)):
		col = int(f(row))
		if col < 0 or col >= img.shape[1]:
			continue
		img[row][col] = (0, 255, 0)
	"""

	# Use derivative to find how much to turn
	turn_value = -math.atan(f_der(lookahead_distance_pixels))/1.5707963268

	return [turn_value, mid_points_best_fit_function]

def find_center_offset(img, mid_points_best_fit_function, lookahead_distance_pixels):
	center_offset = (mid_points_best_fit_function(lookahead_distance_pixels)/img.shape[1])-0.5

	return center_offset


### Drawing Functions ###
def draw_lanes(img):
	pass

### Main Function ###
@socketio.on("new_frame")
def process_frame():
	### Global Vars ###
	global prev_mid_points
	global prev_frame

	### Read Img From Video ###
	orig_img = cap.read()
	orig_img = cv.flip(orig_img, 0)
	orig_img = cv.flip(orig_img, 1)
	img = cv.cvtColor(orig_img, cv.COLOR_RGB2GRAY)
	img_to_save = img
	
	### Calculate Speed Using Image Flow ###
	speed = calculate_speed(prev_frame, img)
	prev_frame = img

	### Img Manipulation ###
	img = cv.resize(img, (int(orig_img.shape[1]*scale_down_factor), int(orig_img.shape[0]*scale_down_factor)))
	img = cv.bitwise_not(img)
	img = cv.blur(img, (3, 3))
	img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, -10)
	#_, img = cv.threshold(img, 130, 255, cv.THRESH_BINARY)
	img = cv.dilate(img, dilate_kernel, iterations=1) 
	img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

	### Lookahead pixels ###
	lookahead_distance_pixels = img.shape[0]*(1-lookahead_distance)

	### Find lane lines ###
	right_lane_mid_points, left_lane_mid_points = find_lane_lines(img, prev_mid_points)

	### Line of best fit ###
	right_best_fit_function = lane_line_of_best_fit(right_lane_mid_points, img);
	left_best_fit_function = lane_line_of_best_fit(left_lane_mid_points, img);

	### Find mid points of lane ###
	mid_points = find_mid_points(img, right_lane_mid_points, left_lane_mid_points, right_best_fit_function, left_best_fit_function)
	# If mid_points is empty, return
	if len(mid_points.keys()) == 0: 
		data = {
			"turn_value": 0,
			"center_offset": 0,
			"speed": 0,
		}
		print(data)
		emit("driving_data", data)
		return

	prev_mid_points = mid_points

	### Find if turning left or right ###
	turn_value, mid_points_best_fit_function = find_turn_value(mid_points, lookahead_distance_pixels)

	### Calculate offset from center of lane ###
	center_offset = find_center_offset(img, mid_points_best_fit_function, lookahead_distance_pixels)

	### Send data to car ###
	data = {
		"turn_value": turn_value,
		"center_offset": center_offset,
		"speed": speed,
	}
	print(data)
	emit("driving_data", data)

	### Draw Look ahead distance ###
	for col in range(img.shape[1]):
		img[int(lookahead_distance_pixels)][col] = (127, 127, 127)

	### Draw Lanes ###

	### Display orig img with overlay and stuff ###
	for row in reversed(range(0, img.shape[0], 2)):
		for col in range(int(left_best_fit_function(row)), int(right_best_fit_function(row)), 2):
			orig_img_row = int(row/scale_down_factor)
			orig_img_col = int(col/scale_down_factor)

			if orig_img_col >= orig_img.shape[1] or orig_img_col < 0:
				continue

			orig_img[orig_img_row][orig_img_col] = (0, 255, 0)

	cv.imshow("input", orig_img)
	cv.imshow("output", img)

	### Save Images For Deep Learning ###
	cv.imwrite("trainingImages/{}-{}.png".format(image_increment, center_offset), img_to_save)

	if cv.waitKey(1) & 0xFF == ord('q'):
		return

if __name__ == '__main__':
	socketio.run(app, host="0.0.0.0", port=8080)

cap.release()
cv.destroyAllWindows()