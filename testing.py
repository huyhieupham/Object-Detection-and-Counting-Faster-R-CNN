import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
import itertools
import operator
from outils import config
import outils.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from outils import roi_helpers
import datetime
import re
import subprocess

parser = OptionParser()
parser.add_option("-i", "--input_file", dest="input_file", help="Path to input video file.")
parser.add_option("-o", "--output_file", dest="output_file", help="Path to output video file.", default="output.mp4")
parser.add_option("-d", "--input_dir", dest="input_dir", help="Path to input working directory.", default="C:/Users/Huy-Hieu.Pham/Desktop/keras-frcnn-master")
parser.add_option("-u", "--output_dir", dest="output_dir", help="Path to output working directory.", default="C:/Users/Huy-Hieu.Pham/Desktop/keras-frcnn-master/outputs")
parser.add_option("-r", "--frame_rate", dest="frame_rate", help="Frame rate of the output video.", default=25.0)

(options, args) = parser.parse_args()
if not options.input_file:   # if filename is not given
	parser.error('Error: path to video input_file must be specified. Pass --input-file to command line')

input_video_file = options.input_file
output_video_file = options.output_file
#img_path = os.path.join(options.input_dir, '')
img_path = 'C:/Users/Huy-Hieu.Pham/Desktop/keras-frcnn-master/outputs'
output_path = os.path.join(options.output_dir, '')
num_rois = 32
frame_rate = float(options.frame_rate)

def cleanup():
	print("cleaning up...")
	os.popen('rm -f ' + img_path + '*')
	os.popen('rm -f ' + output_path + '*')

def get_file_names(search_path):
	for (dirpath, _, filenames) in os.walk(search_path):
		for filename in filenames:
			yield filename#os.path.join(dirpath, filename)

def convert_to_images():
	cam = cv2.VideoCapture(input_video_file)
	counter = 0
	while True:
		flag, frame = cam.read()
		if flag:
			cv2.imwrite(os.path.join(img_path, str(counter) + '.jpg'),frame)
			counter = counter + 1
		else:
			break
		if cv2.waitKey(1) == 27:
			break
			# press esc to quit
	cv2.destroyAllWindows()

def save_to_video():
	list_files = sorted(get_file_names(output_path), key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
	#list_files = sorted(get_file_names(img_path), key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
	img0 = cv2.imread(os.path.join(output_path,'0.jpg'))
	height , width , layers =  img0.shape

	# fourcc = cv2.cv.CV_FOURCC(*'mp4v')
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	#fourcc = cv2.cv.CV_FOURCC(*'XVID')
	videowriter = cv2.VideoWriter(output_video_file,fourcc, frame_rate, (width,height))
	for f in list_files:
		print("saving..." + f)
		img = cv2.imread(os.path.join(output_path, f))
		videowriter.write(img)
	videowriter.release()
	cv2.destroyAllWindows()

def format_img(img, C):
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape

	if width <= height:
		f = img_min_side/width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side/height
		new_width = int(f * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def accumulate(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
            yield key, sum(item[1] for item in subiter)

def main():
	cleanup()
	sys.setrecursionlimit(40000)
	config_output_filename = 'config.pickle'

	with open(config_output_filename, 'rb') as f_in:
		C = pickle.load(f_in)

	# turn off any data augmentation at test time
	C.use_horizontal_flips = False
	C.use_vertical_flips = False
	C.rot_90 = False
	class_mapping = C.class_mapping

	if 'bg' not in class_mapping:
		class_mapping['bg'] = len(class_mapping)

	# class_mapping = {v: k for k, v in class_mapping.iteritems()} 
	class_mapping = {v: k for k, v in class_mapping.items()}
	print(class_mapping)
	class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
	C.num_rois = num_rois

	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
		input_shape_features = (1024, None, None)
	else:
		input_shape_img = (None, None, 3)
		input_shape_features = (None, None, 1024)


	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(C.num_rois, 4))
	feature_map_input = Input(shape=input_shape_features)

	# define the base network (resnet here, can be VGG, Inception, etc)
	shared_layers = nn.nn_base(img_input, trainable=True)

	# define the RPN, built on the base layers
	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn_layers = nn.rpn(shared_layers, num_anchors)

	classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

	model_rpn = Model(img_input, rpn_layers)
	model_classifier_only = Model([feature_map_input, roi_input], classifier)

	model_classifier = Model([feature_map_input, roi_input], classifier)

	model_rpn.load_weights(C.model_path, by_name=True)
	model_classifier.load_weights(C.model_path, by_name=True)

	model_rpn.compile(optimizer='sgd', loss='mse')
	model_classifier.compile(optimizer='sgd', loss='mse')

	all_imgs = []

	classes = {}

	bbox_threshold = 0.8

	visualise = True

	print("Converting video to images..")
	convert_to_images()
	print("anotating...")

	list_files = sorted(get_file_names(img_path), key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
	for img_name in list_files:
		if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
			continue
		print(img_name)
		st = time.time()
		filepath = os.path.join(img_path,img_name)
		img = cv2.imread(filepath)
		X = format_img(img, C)

		img_scaled = np.transpose(X.copy()[0, (2, 1, 0), :, :], (1, 2, 0)).copy()
		img_scaled[:, :, 0] += 123.68
		img_scaled[:, :, 1] += 116.779
		img_scaled[:, :, 2] += 103.939

		img_scaled = img_scaled.astype(np.uint8)

		if K.image_dim_ordering() == 'tf':
			X = np.transpose(X, (0, 2, 3, 1))

		# get the feature maps and output from the RPN
		[Y1, Y2, F] = model_rpn.predict(X)


		R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

		# convert from (x1,y1,x2,y2) to (x,y,w,h)
		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]

		# apply the spatial pyramid pooling to the proposed regions
		bboxes = {}
		probs = {}

		for jk in range(R.shape[0]//C.num_rois + 1):
			ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
			if ROIs.shape[1] == 0:
				break

			if jk == R.shape[0]//C.num_rois:
				#pad R
				curr_shape = ROIs.shape
				target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
				ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
				ROIs_padded[:, :curr_shape[1], :] = ROIs
				ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
				ROIs = ROIs_padded

			[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

			for ii in range(P_cls.shape[1]):

				if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
					continue

				cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

				if cls_name not in bboxes:
					bboxes[cls_name] = []
					probs[cls_name] = []

				(x, y, w, h) = ROIs[0, ii, :]

				cls_num = np.argmax(P_cls[0, ii, :])
				try:
					(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
					tx /= C.classifier_regr_std[0]
					ty /= C.classifier_regr_std[1]
					tw /= C.classifier_regr_std[2]
					th /= C.classifier_regr_std[3]
					x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
				except:
					pass
				bboxes[cls_name].append([16*x, 16*y, 16*(x+w), 16*(y+h)])
				probs[cls_name].append(np.max(P_cls[0, ii, :]))

		all_dets = []
		all_objects = []

		for key in bboxes:
			bbox = np.array(bboxes[key])

			new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
			for jk in range(new_boxes.shape[0]):
				(x1, y1, x2, y2) = new_boxes[jk,:]

				#cv2.rectangle(img_scaled,(x1, y1), (x2, y2), int(class_to_color[key]),2)
				cv2.rectangle(img_scaled,(x1, y1), (x2, y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

				textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
				all_dets.append((key,100*new_probs[jk]))
				all_objects.append((key, 1))

				(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
				textOrg = (x1, y1-0)

				cv2.rectangle(img_scaled, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
				cv2.rectangle(img_scaled, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
				cv2.putText(img_scaled, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
		print('Elapsed time = {}'.format(time.time() - st))
		height, width, channels = img_scaled.shape
		cv2.rectangle(img_scaled, (0,0), (width, 30), (0, 0, 0), -1)
		cv2.putText(img_scaled, "Object counting (Name & Number of Objects): " + str(list(accumulate(all_objects))), (5, 19), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
		cv2.imwrite(os.path.join(output_path, img_name), img_scaled)
		print(all_dets)
	print("saving to video..")
	save_to_video()

if __name__ == '__main__':
	main()
