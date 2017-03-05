
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC, SVC
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label

import os.path
import pickle
from utils import *

color_space    = 'YCrCb'    # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient         = 9          # HOG orientations
pix_per_cell   = 8          # HOG pixels per cell
cell_per_block = 2          # HOG cells per block
hog_channel    = [0, 1, 2]   # array of 0, 1, 2
spatial_size   = (32, 32) # Spatial binning dimensions
hist_bins      = 32       # Number of histogram bins

spatial_feat   = True         # Spatial features on or off
hist_feat      = True         # Histogram features on or off
hog_feat       = True         # HOG features on or off
y_start_stop   = [None, None] # Min and max in y to search in slide_window()

frame_counter = 0
buffer_size   = 20
buffer_boxes  = []

X_scaler = None
svc      = None

# train features, return classifier and scaler
def train_features(car_images,
				   notcar_images,
				   color_space,
				   spatial_size,
				   hist_bins,
				   orient,
				   pix_per_cell,
				   cell_per_block,
				   hog_channel,
				   spatial_feat,
				   hist_feat,
				   hog_feat
				   ):
	# extract car features
	car_features = extract_features( car_images,  
                                 color_space    = color_space, 
                                 spatial_size   = spatial_size,
                                 hist_bins      = hist_bins, 
                                 orient         = orient,
                                 pix_per_cell   = pix_per_cell, 
                                 cell_per_block = cell_per_block, 
                                 hog_channel    = hog_channel,
                                 spatial_feat   = spatial_feat, 
                                 hist_feat      = hist_feat,
                                 hog_feat       = hog_feat )
	# extract notcar features
	notcar_features = extract_features( notcar_images,
                                    color_space    = color_space, 
                                    spatial_size   = spatial_size,
                                    hist_bins      = hist_bins, 
                                    orient         = orient,
                                    pix_per_cell   = pix_per_cell, 
                                    cell_per_block = cell_per_block, 
                                    hog_channel    = hog_channel,
                                    spatial_feat   = spatial_feat, 
                                    hist_feat      = hist_feat,
                                    hog_feat       = hog_feat )

	X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
	# compute mean and std
	scaler = StandardScaler().fit(X)
	# apply standarization to X
	scaled_X = scaler.transform(X)

	# label array
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

	# random shuffle training data set
	scaled_X, y = shuffle(scaled_X, y)

	# Split up data into randomized training and test sets
	randst = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split( scaled_X, y, test_size = 0.2,
                                                     random_state = randst )

	print('Using:', orient,'orientations', pix_per_cell,
      	  'pixels per cell and', cell_per_block,'cells per block')
	print('Feature vector length:', len(X_train[0]))

	# Use a linear SVC 
	svc = LinearSVC(C = 0.1, max_iter=2000)
	# Check the training time for the SVC
	t = time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
	
	# saved classifier, scaler and parameters to a pickle file

	pickle_file = 'svc.pickle'
	try:
		with open(pickle_file, 'wb') as pf:
			pickle.dump(
            	{
                	'svc': svc,
                	'scaler': scaler,
                	'color_space' : color_space,
                	'spatial_size': spatial_size,
                	'hist_bins' : hist_bins,
                	'orient' : orient,
                	'pix_per_cell' : pix_per_cell,
                	'cell_per_block' : cell_per_block,
                	'hog_channel' : hog_channel,
                	'spatial_feat' : spatial_feat,
                	'hist_feat' : hist_feat,
                	'hog_feat' : hog_feat
            	},
            	pf, pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('Unable to save svc to', pickle_file, ':', e)
		raise
	
	return svc, scaler


# detect car from an image
def detect_cars(image):
	global frame_counter, buffer_size, buffer_boxes
	# convert to scale 0 to 1, comment out this line if image already in 0 to 1
	draw_image = np.copy(image)
	
	image = image.astype(np.float32)/255
	heat = np.zeros_like(image[:,:,0]).astype(np.float)
	
	window_sizes = [64, 96]
	box_list = []
	
	for ws in window_sizes:
		windows = slide_window(image,
                               x_start_stop = [None, None],
                               y_start_stop = [400, 656], # search only lower half of the image
                               xy_window    = (ws, ws),
                               xy_overlap   = (0.75, 0.75))
        
		hot_windows = search_windows(image,
                                     windows,
                                     svc,
                                     X_scaler,
                                     color_space    = color_space, 
                                     spatial_size   = spatial_size,
                                     hist_bins      = hist_bins,
                                     orient         = orient,
                                     pix_per_cell   = pix_per_cell, 
                                     cell_per_block = cell_per_block, 
                                     hog_channel    = hog_channel,
                                     spatial_feat   = spatial_feat, 
                                     hist_feat      = hist_feat,
                                     hog_feat       = hog_feat)
        
		for i in range(len(hot_windows)):
			if((hot_windows[i][1][0] -hot_windows[i][0][0]) > 50):
				box_list.append(hot_windows[i])
    
	bufIndex = frame_counter % buffer_size
	if frame_counter < buffer_size:
		buffer_boxes.append(box_list)
	else:
		buffer_boxes[bufIndex] = box_list
        
	frame_counter += 1
    
	# flatten the list of lists
	if frame_counter < buffer_size -1:
		heat = add_heat(heat, box_list)
		heat = apply_threshold(heat, 8)
	else:
		box_lists = [i for b in buffer_boxes for i in b]
		heat = add_heat(heat, box_lists)
		head = apply_threshold(heat, 20)
	
	heat_map = np.clip(heat, 0, 255)
    
	labels = label(heat_map)
	draw_image = draw_labeled_bboxes(draw_image, labels)
    
	return draw_image
	
if __name__ == "__main__":
	# calculate camera and distortion coefficients first
	from moviepy.editor import VideoFileClip
	
	if os.path.exists("svc.pickle") is False:
		car_images   = glob.glob('train_images/vehicles/*/*.png')
		notcar_images = glob.glob('train_images/non-vehicles/*/*.png')

		train_features(car_images,
				   notcar_images,
				   color_space,
				   spatial_size,
				   hist_bins,
				   orient,
				   pix_per_cell,
				   cell_per_block,
				   hog_channel,
				   spatial_feat,
				   hist_feat,
				   hog_feat
				   )

	pickle_data = pickle.load( open("svc.pickle", "rb" ) )
	svc = pickle_data["svc"]
	X_scaler = pickle_data["scaler"]
	color_space = pickle_data["color_space"]
	spatial_size = pickle_data["spatial_size"]
	hist_bins = pickle_data["hist_bins"]
	orient = pickle_data["orient"]
	pix_per_cell = pickle_data["pix_per_cell"]
	cell_per_block = pickle_data["cell_per_block"]
	spatial_feat = pickle_data["spatial_feat"]
	hist_feat = pickle_data["hist_feat"]
	hog_feat = pickle_data["hog_feat"]
	
	input_file = 'project_video.mp4'
	clip = VideoFileClip(input_file)

	output_file = 'out_project_video.mp4'
	out_clip = clip.fl_image(detect_cars)
	out_clip.write_videofile(output_file, audio=False)	
	
	
	
	

