import ipdb
import numpy as np
import scipy as sp
import csv
import copy
import cv2
import glob
import tensorflow
from PIL import Image

# Define constants
CONTOUR_AREA_THRESH = 100
VIGNETTE_THRESH = 90
BLACK_THRESH = 50
CIRCLE_THRESH = 20
# FILT_THRESH = 170
GAUSS_THRESH = 0.02


# create a Matlab like struct class
class MATLABstruct():
	pass


###################### FUNCTIONS #########################
def make2DGaussian(size, radius = 3, center = None):
	""" Make a square gaussian kernel.
	size is the length of a side of the square
	radius is full-width-half-maximum, which
	can be thought of as an effective radius.
	"""
	x = np.arange(0, size, 1, float)
	y = x[:,np.newaxis]

	if center is None:
		x0 = y0 = size // 2
	else:
		x0 = center[0]
		y0 = center[1]

	gauss = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / radius**2)
	return gauss

def circleCrop(img, orig, radiusOption = 0):
	# radius = 0 means vignetting is present, radius = 1 means no vignetting so do not circle crop
	# create circle around center of image to get rid of vignette and other noise 
	img_size = np.shape(img)
	center = (img_size[1]//2,img_size[0]//2)
	use_thresh = GAUSS_THRESH
	if radiusOption == 0:
		# check to see if there is vignetting or if the picture is a dermoscopy image in which case it will be a circular image with the corners all black. 
		# In this case we need to removes the corners
		find_black_circle = np.where(img[img_size[0]/2,:] > BLACK_THRESH)
		c1 = np.min(find_black_circle)
		c2 = np.max(find_black_circle)

		if c1 > CIRCLE_THRESH:
			radius = (c2 - c1)//2
			use_thresh = .095
		else:
			radius = np.min(center)

	else:
		radius = np.max(center)*1.1

	# create gaussian hump square matrix 
	gauss_hump = make2DGaussian(np.max(img_size),radius)	
	py1 = (gauss_hump.shape[0] - np.min(img_size))//2
	py2 = py1 + np.min(img_size)
	circle_mask = gauss_hump[py1:py2,:]

	# apply a binary filter to the guassian hump at a certain level to take a cross section (which will be a circle to get the circle crop)
	circle_mask[circle_mask < use_thresh] = 0
	circle_mask[circle_mask >= use_thresh] = 1

	# generate the image with the cropped section retaining original values and everything else being black
	circle_crop = np.multiply(circle_mask[:,:,None], orig)
	circle_crop[circle_crop == 0] = 255
	circle_crop = np.array(circle_crop, 'uint8')	# change matrix type to integers because pixel values are 8-bit integers from 0-255 not float

	return circle_crop

def detectLesionUsingBlobs(gray_crop):
	# BLOB DETECTION

	ipdb.set_trace()	

	# Blur image to remove noise
	frame = cv2.GaussianBlur(gray_crop, (3, 3), 0)

	# generate filter threshold thats custom to each picture
	FILT_THRESH = np.max((150, np.max(gray_crop) + np.min(gray_crop) - np.mean(gray_crop))) + np.std(gray_crop)/4		# divide by a randomly chosen factor

	# define range of blob color in grayscale
	blobMin = 0
	blobMax = FILT_THRESH

	# Sets pixels to white if in purple range, else will be set to black
	mask = cv2.inRange(frame, blobMin, blobMax)

	# Bitwise-AND of mask and purple only image - only used for display
	res = cv2.bitwise_and(frame, frame, mask= mask)

	#    mask = cv2.erode(mask, None, iterations=1)
	# commented out erode call, detection more accurate without it

	# dilate makes the in range areas larger
	mask = cv2.dilate(mask, None, iterations=1)
	
	# Set up the SimpleBlobdetector with default parameters.
	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	params.minThreshold = 0;
	params.maxThreshold = 256;

	# Filter by Area.
	params.filterByArea = True
	params.minArea = 1000

	# Filter by Circularity
	params.filterByCircularity = False
	params.minCircularity = 0.1

	# Filter by Convexity
	params.filterByConvexity = False
	params.minConvexity = 0.5

	# Filter by Inertia. 0 is close to a line/stretched out ellipse and 1 is a circle
	params.filterByInertia = False
	params.minInertiaRatio = 0.5

	detector = cv2.SimpleBlobDetector_create(params)

	# Detect blobs.
	reversemask=255-mask
	keypoints = detector.detect(reversemask)

	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	im_with_keypoints = cv2.drawKeypoints(gray_crop, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	# Show keypoints
	cv2.imshow("Keypoints", im_with_keypoints)
	cv2.waitKey(0)

def getContours(thresh, orig):
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

	# calculate the area of each contour
	area = np.zeros(len(cnts))
	for i, contour in enumerate(cnts):
		area[i] = cv2.contourArea(contour)
		# print 'area: ', area[i]

	# filter contours by area
	cnts_filt_indx = [i for i,v in enumerate(area) if v > CONTOUR_AREA_THRESH]

	
	# draw contours on original image
	for cont_i in enumerate(cnts_filt_indx):
		cnt = cnts[cont_i[1]]
		cv2.drawContours(orig, [cnt], 0, (0,255,0), 3)


def formatImageForNN(img, dim=(256,256)):
	# Take input image (x,y,3) and make it a square image then resize the image to be 256x256 pixels for the neural network input
	# (eventually make the function crop out a square of the convex hull of the mole)
	
	# input image size
	img_size = img.shape[0:2]

	# make a new blank/black sqaure image with the larger of the two sides
	sqr_side_lenth = np.max(img_size)
	sqr_img = np.zeros((sqr_side_lenth,sqr_side_lenth,3),'uint8')

	# fill in the top right corner of the new black image with the lesion image
	sqr_img[0:img_size[0],0:img_size[1],:] = img 

	# resize the image to be a 256 x 256 
	resized = cv2.resize(sqr_img,dim,interpolation = cv2.INTER_AREA)
	# cv2.imshow('t',resized)

	return resized





def preProcessImage(original, plotting = False):
	# import plotting library
	if plotting:
		from matplotlib import pyplot as plt

	# check if the picture is in landscape, if not make it landscape by rotating in 90 degrees
	rows,cols,colors = original.shape
	if rows > cols:
		orig = np.zeros([cols,rows,colors])
		for f in range(original.shape[2]):
			orig[:,:,f] = np.fliplr(original[:,:,f].T)	# rotate 90 degrees clockwise
		orig = np.array(orig[1:-1,1:-1], 'uint8')	# get rid of black border on most images and change to correct format type
	else:
		orig = original[1:-1,1:-1]	# get rid of black border on most images

	# convert to gray scale
	gray_img = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)

	# check to see if corners of the image are dark which would indicate vignetting. If so, then apply the circle crop
	box1 = 5
	box2 = 10
	if np.mean(gray_img[box1:box2,box1:box2]) < VIGNETTE_THRESH or np.mean(gray_img[-box2:-box1,-box2:-box1]) < VIGNETTE_THRESH:
	# if gray_img[0,0] > VIGNETTE_THRESH or gray_img[-1,-1] > VIGNETTE_THRESH:
		# crop circle using 2D guassian with mean zero variable variance (radius)
		circle_crop = circleCrop(gray_img, orig, 0)
	else:
		circle_crop = circleCrop(gray_img, orig, 1)

	# gray scale the new circle cropped image
	gray_crop = cv2.cvtColor(circle_crop,cv2.COLOR_BGR2GRAY)

	# # Blob detection method
	# detectLesionUsingBlobs(gray_crop)


	# # increase contrast locally
	# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	# contra1 = clahe.apply(gray_crop)
	# # increase contrast globally
	# glob_contra = cv2.equalizeHist(gray_crop)


	# generate filter threshold thats custom to each picture
	FILT_THRESH = np.max((150, np.max(gray_crop) + np.min(gray_crop) - np.mean(gray_crop))) + np.std(gray_crop)/10		# divide by a randomly chosen factor

	# apply binary filter to extract darker skin
	thresh = cv2.threshold(gray_crop, FILT_THRESH, 255, 0)[1]		# same thing as:	filt[filt>FILT_THRESH] = 255, filt[filt<=FILT_THRESH] = 0

	# # find contours
	# cnts, cnts_filt_indx = getContours(thresh, orig)

	# apply binary filter to the original colored image
	lesion = copy.deepcopy(circle_crop)
	lesion[thresh == 255] = (0,0,0)
	lesion = cv2.cvtColor(lesion, cv2.COLOR_BGR2RGB)

	# run the canny edge detector
	lesion_gray = cv2.cvtColor(lesion, cv2.COLOR_RGB2GRAY)
	lesion_edges = cv2.Canny(lesion_gray, 100,200)

	# format image shape and size for the neural network
	final_processed = formatImageForNN(lesion)


	if plotting:
		# plot some stuff
		plt.subplot(171),plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
		plt.title('Original'), plt.xticks([]), plt.yticks([])
		plt.subplot(172),plt.imshow(cv2.cvtColor(circle_crop, cv2.COLOR_BGR2RGB))
		plt.title('Crop'), plt.xticks([]), plt.yticks([])
		plt.subplot(173),plt.imshow(gray_crop)
		plt.title('Contrast'), plt.xticks([]), plt.yticks([])
		plt.subplot(174),plt.imshow(thresh)
		plt.title('Filter'), plt.xticks([]), plt.yticks([])
		plt.subplot(175), plt.imshow(lesion_edges)
		plt.title('Edges'), plt.xticks([]), plt.yticks([])
		plt.subplot(176),plt.imshow(lesion)
		plt.title('Processed'), plt.xticks([]), plt.yticks([])
		plt.subplot(177),plt.imshow(final_processed)
		plt.title('Resized'), plt.xticks([]), plt.yticks([])
		plt.show()

		# plt.savefig('figures/segmentation_' + str(np.round(np.random.rand()*100)) + '.png', bbox_inches='tight')
		# cv2.waitKey(0)

	# ipdb.set_trace()

	return final_processed, thresh

def getDataLabels(options):
	# Data files
	train_data_truth_filename = 'data/dataset_' + options.dataset + '/raw/ISIC_' + options.year + '_Training_GroundTruth.csv'
	test_data_truth_filename = 'data/dataset_' + options.dataset + '/raw/ISIC_' + options.year + '_Test_GroundTruth.csv'

	# import data from CSV (train)
	open_train = open(train_data_truth_filename,'rb')
	open_test = open(test_data_truth_filename,'rb')
	train_data = np.asarray(list(csv.reader(open_train)))
	test_data = np.asarray(list(csv.reader(open_test)))

	# extract the image labels and convert benign and malignant strings to binary  {0,1}
	img_names_train = copy.deepcopy(train_data[:,0])
	y_train = copy.deepcopy(train_data[:,1])
	y_train[y_train == 'benign'] = 0
	y_train[y_train == 'malignant'] = 1
	y_train[y_train == '0.0'] = 0
	y_train[y_train == '1.0'] = 1
	labels_train = np.array(y_train,'uint8')

	img_names_test = copy.deepcopy(test_data[:,0])
	y_test = copy.deepcopy(test_data[:,1])
	y_test[y_test == '0.0'] = 0
	y_test[y_test == '1.0'] = 1
	y_test[y_test == 'benign'] = 0
	y_test[y_test == 'malignant'] = 1
	labels_test = np.array(y_test,'uint8')

	# # save formatted labels
	# np.savetxt("labels.csv", y, delimiter=',')

	labels = MATLABstruct()
	img_names = MATLABstruct()

	labels.train = labels_train
	labels.test = labels_test
	img_names.train = img_names_train
	img_names.test = img_names_test

	return img_names, labels


def getBalancedDataLabels(trainOrTest, options):
	# define trainOrTest as 'test' or 'train'
	all_img_names, all_labels = getDataLabels(options)
	path = 'data/dataset_' + options.dataset + '/formatted/' + trainOrTest + '/processed/balanced/*.jpeg'

	if trainOrTest == 'train':
		labels = all_labels.train
		names = all_img_names.train 
	elif trainOrTest == 'test':
		labels = all_labels.test
		names = all_img_names.test 

	balanced_labels = np.array([])
	for j, img_name in enumerate(glob.glob(path)):
		# img_names[i] = img
		indx = [i for i in range(names.shape[0]) if names[i]==img_name[-17:-5]]
		add_label = labels[indx]

		if balanced_labels.shape[0] == 0:
			balanced_labels = add_label
		else:
			balanced_labels = np.vstack((balanced_labels,add_label))

	balanced_labels = balanced_labels.reshape(balanced_labels.shape[0],)
	return balanced_labels


def getSegmentationError(seg_truth_file_path, seg_estimate, error_thresh=.2):
	# calculate the area of the segmentation truth and estimate and if they're within a certain error threshold of each other 
	# consider the image properly segmented (return error=0) and if not (return error=1) 

	# import truth binary mask as gray scale and invert it (by subtracting it from 255) to match format of seg_estimate
	seg_truth = 255 - cv2.imread(seg_truth_file_path,0)	

	# normalize both images
	seg_truth = seg_truth/255
	seg_estimate = seg_estimate/255

	# calculate what 5% of the total number of pixels in the image is and use that number as the error threshold
	pixel_thresh = np.prod(seg_truth.shape)*error_thresh

	# calculate the number of white pixels in each image 
	seg_truth_area = int(np.sum(seg_truth))
	seg_estimate_area = int(np.sum(seg_estimate))

	# calculate the difference in area between ground truth and preprocessed estimate of segmentation
	area_diff = np.absolute(seg_truth_area - seg_estimate_area)

	# two methods for calculating error
	error = 0 if area_diff < pixel_thresh else 1

	return error 





######################## MAIN #############################
def processAll(options):

	path = options.data_input_path
	names,labels = getDataLabels(options)
	if options.name == 'train':
		names = names.train
		labels = labels.train
	elif options.name == 'test':
		names = names.test
		labels = labels.test


	# import all images from the training/test set
	dataset = []
	img_names = dict()
	seg_error = 0
	for i, img in enumerate(glob.glob(options.data_input_path)):
		orig = cv2.imread(img)	# load image i
		img_names[i] = img

		if options.clean:
			p1, seg_estimate = preProcessImage(orig, options.show_images)	# show_images is a global variables

			# calculate segmentation error
			if options.calculate_error:
				seg_truth_file_path = options.data_input_path[:-5] + img_names[i][-16:-4] + '_Segmentation.png'
				seg_error += getSegmentationError(seg_truth_file_path, seg_estimate)

			folder_processed_name = 'processed/'
		else:
			orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
			p1 = formatImageForNN(orig)
			folder_processed_name = 'unprocessed/'

		# save image as jpeg in the folder corresponing to its label. This is for tensorflow-slim format purposes
		if names[i]==img_names[i][-16:-4] and options.separate_imgs_by_label:
			if labels[i] == 0: 
				folder_label_name = 'label_0/'
			elif labels[i] == 1:
				folder_label_name = 'label_1/'
		else:
			folder_label_name = ''

		if options.save:
			# save the image
			im = Image.fromarray(p1)
			im.save(options.save_path + options.name + '/' + folder_processed_name + folder_label_name + img_names[i][-16:-4] + ".jpeg")

	# lesion segmentation error
	error = seg_error/float(len(img_names))
	print error



def main():
	# define variable clean as (clean=True: processing/cleaning/resizing images) or (c;ean=False: just resizing images)
	# separate_imgs_by_label (False: images are all put in one folder, True: images are put into 2 separate folders based on their associated label)
	train_options = MATLABstruct()
	train_options.name = 'train'
	train_options.clean = True
	train_options.separate_imgs_by_label = True 
	train_options.save = True		
	train_options.dataset = '2'
	train_options.year = '2016' if train_options.dataset == '1' else '2017' if train_options.dataset == '2' else '' 
	train_options.data_input_path = 'data/dataset_' + train_options.dataset + '/raw/ISIC_' + train_options.year + '_Training_Data/*.jpg'
	train_options.save_path = 'data/dataset_' + train_options.dataset + '/cnn_data/balanced/'		# "data/formatted/"
	# train_options.data_input_path = "scraping/scraped_data/*.jpg"
	# train_options.save_path = 'scraping/formatted/'
	train_options.show_images = False
	train_options.calculate_error = False

	test_options = MATLABstruct()
	test_options.name = 'test'
	test_options.clean = True
	test_options.separate_imgs_by_label = True 		
	test_options.save = False
	test_options.dataset = train_options.dataset
	test_options.year = train_options.year
	test_options.data_input_path = 'data/dataset_' + test_options.dataset + '/raw/ISIC_' + test_options.year + '_Test_Data/*.jpg'
	test_options.save_path = 'data/dataset_' + test_options.dataset + '/cnn_data/balanced/' 		#"data/formatted/balanced/"
	test_options.show_images = False
	test_options.calculate_error = False

	# processAll(train_options)
	processAll(test_options)



if __name__ == "__main__":
    main()



