'''
Author: Nathan LaPierre

Modified version of the keras implementation of VGG16.
The keras code is open-source under the MIT license,
	& the VGG architecture and weights are 
	available under the Creative Commons License
	(https://creativecommons.org/licenses/by/4.0/).
We use TensorFlow as our backend, which is under the
	Apache license.
Model adapted for regression and methods for generating
	inputs and gathering results added.
'''


from __future__ import absolute_import

import argparse, glob, os, sys, time
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC, SVR
from sklearn.externals import joblib
import pandas as pd

import warnings
import json

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras import optimizers


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
start = time.time()


def echo(msg):
	global start
	seconds = time.time() - start
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	hms = "%02d:%02d:%02d" % (h, m, s)
	print '['+hms+'] ' + msg


def parseargs():
	parser = argparse.ArgumentParser(description='Create pileups from .paf read-to-read mapping and fastq reads.')
	parser.add_argument('--baseline', action='store_true', help='Compare with baseline: SVM trained on # of minimizers matched.')
	parser.add_argument('--classify', default=-1.0, type=float, help='Turn into classification problem; specify threshold.')
	parser.add_argument('--debug', default=0, type=int, help='Number of examples to use in debug mode. If 0, non-debug mode (default).')
	parser.add_argument('--epochs', default=5, type=int, help='Number of epochs to train the network. Default: 5.')
	parser.add_argument('--extra', default=0, type=int, help='Number of fully connected layers to add to VGG. Default: 0')
	parser.add_argument('--input', default='./', help='Directory with png pileup images. Default: current directory.')
	parser.add_argument('--labels', default='labels.txt', help='Path to image labels file.')
	parser.add_argument('--load', default='NONE', help='Path to keras model file to load. Default: do not load.')
	parser.add_argument('--loadsvm', default='NONE', help='Path to saved pickled SVM file. Default: do not load.')
	parser.add_argument('--output', default='NONE', help='File to write model to. Default: no output.')
	parser.add_argument('--outputsvm', default='NONE', help='File to write svm model to. Default: no output.')
	parser.add_argument('--plot', default='NONE', help='Where to save a plot of predicted vs actual. Default: no plot.')
	parser.add_argument('--segment_size', type=int, default=100, help='Size of read segments to evaluate.')
	parser.add_argument('--streaming', action='store_true', help='Streaming loading of images to reduce memory footprint.')
	parser.add_argument('--test_input', default='NONE', help='Directory of serparate set of images to test model on.')
	parser.add_argument('--test_labels', default='NONE', help='Path to file with labels for images in test set.')
	parser.add_argument('--window_size', type=int, default=200, help='Window size for VGG. Window >= segment size.')
	args = parser.parse_args()
	return args


def _obtain_input_shape(input_shape, default_size, min_size, data_format, include_top):
	if data_format == 'channels_first':
		default_shape = (3, default_size, default_size)
	else:
		default_shape = (default_size, default_size, 3)
	if include_top:
		if input_shape is not None:
			if input_shape != default_shape:
				raise ValueError('When setting`include_top=True`, '
								 '`input_shape` should be ' + str(default_shape) + '.')
		input_shape = default_shape
	else:
		if data_format == 'channels_first':
			if input_shape is not None:
				if len(input_shape) != 3:
					raise ValueError('`input_shape` must be a tuple of three integers.')
				if input_shape[0] != 3:
					raise ValueError('The input must have 3 channels; got '
									 '`input_shape=' + str(input_shape) + '`')
				if ((input_shape[1] is not None and input_shape[1] < min_size) or
				   (input_shape[2] is not None and input_shape[2] < min_size)):
					raise ValueError('Input size must be at least ' +
									 str(min_size) + 'x' + str(min_size) + ', got '
									 '`input_shape=' + str(input_shape) + '`')
			else:
				input_shape = (3, None, None)
		else:
			if input_shape is not None:
				if len(input_shape) != 3:
					raise ValueError('`input_shape` must be a tuple of three integers.')
				if input_shape[-1] != 3:
					raise ValueError('The input must have 3 channels; got '
									 '`input_shape=' + str(input_shape) + '`')
				if ((input_shape[0] is not None and input_shape[0] < min_size) or
				   (input_shape[1] is not None and input_shape[1] < min_size)):
					raise ValueError('Input size must be at least ' +
									 str(min_size) + 'x' + str(min_size) + ', got '
									 '`input_shape=' + str(input_shape) + '`')
			else:
				input_shape = (None, None, 3)
	return input_shape


def VGG16(include_top=True, weights='imagenet',
		  input_tensor=None, input_shape=None,
		  pooling=None, extra=0, classify=-1.0):

	if weights not in {'imagenet', None}:
		raise ValueError('The `weights` argument should be either '
						 '`None` (random initialization) or `imagenet` '
						 '(pre-training on ImageNet).')

	# Determine proper input shape
	input_shape = _obtain_input_shape(input_shape,
									  default_size=224,
									  min_size=48,
									  data_format=K.image_data_format(),
									  include_top=include_top)

	if input_tensor is None:
		img_input = Input(shape=input_shape)
	else:
		if not K.is_keras_tensor(input_tensor):
			img_input = Input(tensor=input_tensor, shape=input_shape)
		else:
			img_input = input_tensor

	x = BatchNormalization()(img_input)
	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)#(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
	#x = BatchNormalization()(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
	#x = BatchNormalization()(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
	#x = BatchNormalization()(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
	#x = BatchNormalization()(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

	if include_top:
		# Classification block
		x = Flatten(name='flatten')(x)
		x = Dense(4096, activation='relu', name='fc1')(x)
		x = Dropout(0.5)(x)
		x = Dense(4096, activation='relu', name='fc2')(x)
		x = Dropout(0.5)(x)
		x = Dense(1000, activation='softmax', name='predictions')(x)
	else:
		if pooling == 'avg':
			x = GlobalAveragePooling2D()(x)
		elif pooling == 'max':
			x = GlobalMaxPooling2D()(x)

	# Ensure that the model takes into account
	# any potential predecessors of `input_tensor`.
	if input_tensor is not None:
		inputs = get_source_inputs(input_tensor)
	else:
		inputs = img_input
	# Create model.
	model = Model(inputs, x, name='vgg16')

	# load weights
	if weights == 'imagenet':
		if include_top:
			weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
									WEIGHTS_PATH,
									cache_subdir='models')
		else:
			weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
									WEIGHTS_PATH_NO_TOP,
									cache_subdir='models')
		model.load_weights(weights_path)
		if K.backend() == 'theano':
			layer_utils.convert_all_kernels_in_model(model)

		if K.image_data_format() == 'channels_first':
			if include_top:
				maxpool = model.get_layer(name='block5_pool')
				shape = maxpool.output_shape[1:]
				dense = model.get_layer(name='fc1')
				layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

			if K.backend() == 'tensorflow':
				warnings.warn('You are using the TensorFlow backend, yet you '
							  'are using the Theano '
							  'image data format convention '
							  '(`image_data_format="channels_first"`). '
							  'For best performance, set '
							  '`image_data_format="channels_last"` in '
							  'your Keras config '
							  'at ~/.keras/keras.json.')

	# model.layers.pop()
	#x = Flatten(name='flatten')(model.layers[-1].output)
	#x = BatchNormalization()(x)
	x = Dense(4096, activation='relu', name='fc1')(model.layers[-1].output)#(x)
	x = Dropout(0.5)(x)
	#x = BatchNormalization()(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dropout(0.5)(x)
	for i in range(extra):
		#x = BatchNormalization()(x)
		x = Dense(4096, activation='relu', name='fc'+str(i+3))(x)
		x = Dropout(0.5)(x)	
	#x = BatchNormalization()(x)
	if classify == -1.0:
		x = Dense(1, activation='relu', name='predictions_new')(x)
	else:
		x = Dense(1, activation='sigmoid', name='predictions_new')(x)
	new_model = Model(inputs, x, name='vgg16_new')
	return new_model


def eval_preds_classify(actual, predicted, val, baseline=False):
	if not baseline:
		predicted = np.array([i[0] for i in predicted])
	tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
	for i in range(len(actual)):
		if actual[i] >= val and predicted[i] >= val:
			tp += 1.0
		elif actual[i] < val and predicted[i] >= val:
			fp += 1.0
		elif actual[i] < val and predicted[i] < val:
			tn += 1.0
		elif actual[i] >= val and predicted[i] < val:
			fn += 1.0
	accuracy, precision, recall, f1, specificity, aucroc = ['nan' for i in range(6)]
	if tp + fp + tn + fn > 0:
		accuracy = (tp + tn) / (tp + fp + tn + fn)
	if tp + fp > 0:
		precision = tp / (tp + fp)
	if tp + fn > 0:
		recall = tp / (tp + fn)
	if precision != 'nan' and recall != 'nan' and precision + recall > 0:
		f1 = 2 * precision * recall / (precision + recall)
	if fp + tn > 0:
		specificity = tn / (tn + fp)
	binary_actual = [1 if actual[i] > val else 0 for i in range(len(actual))]
	if not (sum(binary_actual) == 0 or sum(binary_actual) == len(binary_actual)):
		aucroc = roc_auc_score(binary_actual, predicted)
	print 'Classification Threshold: ' + str(val)
	print 'Accuracy: ' + str(accuracy)
	print 'Precision: ' + str(precision)
	print 'Recall/Sensitivity: ' + str(recall)
	print 'Specificity: ' + str(specificity)
	print 'AUC-ROC: ' + str(aucroc)


def eval_preds(args, actual, predicted, baseline=False):
	if not baseline:
		predicted = np.array([i[0] for i in predicted])
	errors = [abs(actual[i]-predicted[i]) for i in range(len(actual))]
	print 'Average error: ' + str(np.mean(errors))
	mse = np.mean([i**2 for i in errors])
	print 'Mean squared error: ' + str(mse) + '\n'
	percented, within1, within5, within10 = 100.0 / float(len(actual)), 0.0, 0.0, 0.0
	for i in range(len(actual)):
		if errors[i] < 0.01:
			within1 += percented
		if errors[i] < 0.05:
			within5 += percented
		if errors[i] < 0.1:
			within10 += percented
	print str(within1) + ' percent of predictions within 1.0 of actual'
	print str(within5) + ' percent of predictions within 5.0 of actual'
	print str(within10) + ' percent of predictions within 10.0 of actual'		
	print str(100.0 - within10) + ' percent of predictions outside 10.0 from actual'
	print '\nPearson correlation: ' + str(pearsonr(actual, predicted)[0])
	print 'Spearman rank correlation: ' + str(spearmanr(actual, predicted)[0])
	print '\nClassification metrics for various cutoff thresholds:\n'
	cutoffs, df = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], {}
	for val in cutoffs:
		tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
		for i in range(len(actual)):
			if actual[i] >= val and predicted[i] >= val:
				tp += 1.0
			elif actual[i] < val and predicted[i] >= val:
				fp += 1.0
			elif actual[i] < val and predicted[i] < val:
				tn += 1.0
			elif actual[i] >= val and predicted[i] < val:
				fn += 1.0
		accuracy, precision, recall, f1, specificity, aucroc = ['nan' for i in range(6)]
		if tp + fp + tn + fn > 0:
			accuracy = (tp + tn) / (tp + fp + tn + fn)
		if tp + fp > 0:
			precision = tp / (tp + fp)
		if tp + fn > 0:
			recall = tp / (tp + fn)
		if precision != 'nan' and recall != 'nan' and precision + recall > 0:
			f1 = 2 * precision * recall / (precision + recall)
		if fp + tn > 0:
			specificity = tn / (tn + fp)
		binary_actual = [1 if actual[i] > val else 0 for i in range(len(actual))]
		if not (sum(binary_actual) == 0 or sum(binary_actual) == len(binary_actual)):
			aucroc = roc_auc_score(binary_actual, predicted)
		df[val] = [accuracy, precision, recall, specificity, aucroc]
	df = pd.DataFrame.from_dict(df, orient='index')
	df = df.sort_index()
	df.index.name = 'Cutoff'
	df.columns = ['Accuracy', 'Precision', 'Recall/Sensitivity', 'Specificity', 'AUC-ROC']
	print df

	if args.plot != 'NONE':
		print '\n'; echo('Creating plot...')
		ordered_actual = [[actual[i], i] for i in range(len(actual))]
		ordered_actual.sort(key=lambda x: x[0])
		ordered_predicted = np.array([predicted[i[1]] for i in ordered_actual])
		ordered_actual = np.array([i[0] for i in ordered_actual])
		#X = np.arange(0,len(ordered_actual),1)
		plt.figure(figsize=(5,5), dpi=1000)
		#plt.plot(X, ordered_actual)
		#plt.plot(X, ordered_predicted)
		plt.scatter(ordered_actual, ordered_predicted)
		plt.xlim(0.5, 1.03)#len(ordered_actual))
		plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
		plt.xlabel('Actual pct. id')
		plt.ylim(0.5, 1.03)
		plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
		plt.ylabel('Predicted pct. id')
		plt.title('Predicted (Y-axis) vs. Actual (X-axis) pct. id')
		plt.plot(np.arange(0.5,1.0,0.01), np.arange(0.5,1.0,0.01),'k--')  # dashed diagonal line
		plt.savefig(args.plot, dpi=1000)


def process_images(args, labels_dict, generating=False):
	names = []
	data, svmdata, labels, endpoints = [], [], [], []  # endpoints is position where full reads end
	segments = 0
	for fname in glob.glob(args.input+'*.png'):
		imname = fname.split('/')[-1][:-4]
		if imname not in labels_dict:
			continue
		imlabels = labels_dict[imname]
		zero_segments, pos = [1, 1], 0
		for i in imlabels:  # here we determine the 0 identity segments on the end of reads, which are junk
			if i == 0:
				zero_segments[pos] += 1
			else:
				pos=1
		imarray = ndimage.imread(fname, mode='RGB')

		empty, emptycount = True, 0
		for i in range(1, len(imarray)):
			for pix in imarray[i]:
				for val in pix:
					if val != 0.0:
						empty = False
						break
				if empty == False:
					break
			if empty == False:
				break
		if empty:  # ignore reads with no matching reads
			continue

		names.append(fname)

		# break read into windows, excluding junk 0s at the ends
		sidelen = (args.window_size - args.segment_size) / 2  # extra space on each side of segment in window
		blanks = [[0,0,0]] * ((48 - args.window_size) / 2)
		for i in range(zero_segments[0], len(imlabels)-zero_segments[1]):
			startpos, endpos = (i*args.segment_size)-sidelen, ((i+1)*args.segment_size)+sidelen
			if startpos < 0:
				continue
			if endpos > len(imarray[0]):
				break
			window = imarray[:,startpos:endpos]

			if args.baseline:
				vec, counts = [], 0
				for col in range(len(window[0])):
					counts = 0
					for row in range(len(window)):
						if window[row][col][0] > 128.0:
							counts += 1
					vec.append(counts)
				svmdata.append(vec)

			if generating == True:
				segments += 1
				label = imlabels[i]
				labels.append(label)
				continue
			
			if len(blanks) > 0:
				window = list(window)
				for j in range(len(window)):
					window[j] = list(window[j])
					window[j] = np.concatenate((blanks, window[j], blanks), axis=0)
				window = np.array(window)

			label = imlabels[i]
			if len(window) < 48:
				blankrows = [[[0,0,0]] * len(window[0])] * (48 - len(window))
				window = np.concatenate((window, blankrows), axis=0)
			data.append(window)
			labels.append(label)
		endpoints.append(len(labels))
		if args.debug > 0 and len(endpoints) >= args.debug:
			break

	if len(labels) == 0:
		print 'Error: no data found.'
		sys.exit()

	data = np.array(data)
	svmdata = np.array(svmdata)
	labels = np.array(labels)
	if args.classify != -1.0:
		for i in range(len(labels)):
			if labels[i] < args.classify:
				labels[i] = 0.0
			else:
				labels[i] = 1.0
	if not generating:
		segments = len(data)
	#print names[int(0.8*len(endpoints)):]
	#print('Process images test: ', str(len(names[int(0.8*len(endpoints)):])))
	return data, svmdata, labels, endpoints, segments


def get_data(args, testing=False, generating=False):
	if testing == True:  # get data for the test images provided
		args.labels = args.test_labels
		args.input = args.test_input

	labels_dict = {}
	labels_file = open(args.labels, 'r')
	for line in labels_file:
		splits = line.strip().split(' ')
		if len(splits) < 2:
			continue
		labels_dict[splits[0]] = [float(i) for i in splits[1].split(',')]
	labels_file.close()

	data, svmdata, labels, endpoints, segments = process_images(args, labels_dict, generating)

	# we set the indices to split the data into train/validation/test
	# and round these up to the end of the nearest read to prevent overfitting
	'''train_index, valid_index = int(0.6*segments), int(0.8*segments)
	for point in endpoints:
		if point >= train_index:
			train_index = point
			break
	for point in endpoints:
		if point >= valid_index:
			valid_index = point
			break'''
	train_index, valid_index = endpoints[int(0.6*len(endpoints))], endpoints[int(0.8*len(endpoints))]
	#print(endpoints, len(endpoints), len(labels))
	#print(train_index, valid_index)

	return data, svmdata, labels, train_index, valid_index


def run_network(args, data, svmdata, labels, train_index, valid_index):
	vgg = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=data[0].shape, pooling='max', extra=args.extra, classify=args.classify)
	if args.classify == -1.0:
		opt = optimizers.Adam(lr=0.0001)
		vgg.compile(loss='mean_squared_error', optimizer=opt)
	else:
		opt = optimizers.Adam(lr=0.00001)
		vgg.compile(loss='binary_crossentropy', optimizer=opt)
	echo('Fitting model...')
	vgg.fit(data[:train_index], labels[:train_index], epochs=args.epochs, validation_data=(data[train_index:valid_index], labels[train_index:valid_index]), batch_size=64)
	print '\n'; echo('Predicting...')
	predictions = vgg.predict(data[valid_index:], batch_size=64)
	if args.classify == -1.0:
		eval_preds(args, labels[valid_index:], predictions)
		if args.baseline:
			print '\n\n'; echo('SVM Baseline:')
			svm = SVR()
			svm.fit(svmdata[:train_index], labels[:train_index])
			svm_predictions = svm.predict(svmdata[valid_index:])
			eval_preds(args, labels[valid_index:], svm_predictions, baseline=True)
	else:
		eval_preds_classify(labels[valid_index:], predictions, args.classify)
		if args.baseline:
			print '\n\n'; echo('SVM Baseline:')
			svm = SVC()
			svm.fit(svmdata[:train_index], labels[:train_index])
			svm_predictions = svm.predict(svmdata[valid_index:])
			eval_preds_classify(labels[valid_index:], svm_predictions, args.classify, baseline=True)

	if args.output != 'NONE':
		if not args.output.endswith('.hd5'):
			args.output += '.hd5'
		vgg.save(args.output)

	if args.baseline and args.outputsvm != 'NONE':
		if not args.outputsvm.endswith('.pkl'):
			args.outputsvm += '.pkl'
		joblib.dump(svm, args.outputsvm)


def load_and_test(args):
	if args.test_input == 'NONE':
		print 'No data to test on. Exiting...'
		sys.exit()
	echo('Processing test data...')
	if args.loadsvm != 'NONE':
		args.baseline = True
	data, svmdata, labels, train_index, valid_index = get_data(args, testing=True)
	echo('Done processing test data. Loading models...')

	if args.load != 'NONE':
		vgg = load_model(args.load)
		echo('Neural network model loaded successfully. Predicting...\n')
		predictions = vgg.predict(data, batch_size=64)
		if args.classify == -1.0:
			eval_preds(args, labels, predictions)
		else:
			eval_preds_classify(labels, predictions, args.classify)

	if args.loadsvm != 'NONE':
		svm = joblib.load(args.loadsvm)
		print '\n\n'; echo('SVM model loaded successfully. Predicting...\n')
		predictions = svm.predict(svmdata)
		echo('SVM Predictions:\n')
		if args.classify == -1.0:
			eval_preds(args, labels, predictions, baseline=True)
		else:
			eval_preds_classify(labels, predictions, args.classify)


def setup_generator(args):  # creates train/valid/test sets from filenames, and dict mapping reads to labels
	labels_dict = {}
	labels_file = open(args.labels, 'r')
	for line in labels_file:
		splits = line.strip().split(' ')
		if len(splits) < 2:
			continue
		labels_dict[splits[0]] = [float(i) for i in splits[1].split(',')]
	labels_file.close()

	names, labels, endpoints = [], [], []  # endpoints is position where full reads end
	segments = 0
	for fname in glob.glob(args.input+'*.png'):
		imname = fname.split('/')[-1][:-4]
		if imname not in labels_dict:
			continue
		imlabels = labels_dict[imname]
		zero_segments, pos = [1, 1], 0
		for i in imlabels:  # here we determine the 0 identity segments on the end of reads, which are junk
			if i == 0:
				zero_segments[pos] += 1
			else:
				pos=1
		imarray = ndimage.imread(fname, mode='RGB')

		empty, emptycount = True, 0
		for i in range(1, len(imarray)):
			for pix in imarray[i]:
				for val in pix:
					if val != 0.0:
						empty = False
						break
				if empty == False:
					break
			if empty == False:
				break
		if empty:  # ignore reads with no matching reads
			continue
		names.append(fname)

		# break read into windows, excluding junk 0s at the ends
		sidelen = (args.window_size - args.segment_size) / 2  # extra space on each side of segment in window
		blanks = [[0,0,0]] * ((48 - args.window_size) / 2)
		for i in range(zero_segments[0], len(imlabels)-zero_segments[1]):
			startpos, endpos = (i*args.segment_size)-sidelen, ((i+1)*args.segment_size)+sidelen
			if startpos < 0:
				continue
			if endpos > len(imarray[0]):
				break
			window = imarray[:,startpos:endpos]
			segments += 1
			labels.append(imlabels[i])

		endpoints.append(len(labels))
		if args.debug > 0 and len(names) >= args.debug:
			break

	if len(labels) == 0:
		print 'Error: no data found.'
		sys.exit()

	labels = np.array(labels)
	if args.classify != -1.0:
		for i in range(len(labels)):
			if labels[i] < args.classify:
				labels[i] = 0.0
			else:
				labels[i] = 1.0

	train_index, valid_index = endpoints[int(0.6*len(endpoints))], endpoints[int(0.8*len(names))-1]
	train, valid, test = names[:int(0.6*len(names))], names[int(0.6*len(names)):int(0.8*len(names))], names[int(0.8*len(names)):]
	return train, valid, test, train_index, valid_index, labels_dict, labels


def generate_batches(args, filenames, labels_dict, batch_size):
	data, labels, imcount, yielded = [], [], 0, False
	while 1:
		for fname in filenames:
			imname = fname.split('/')[-1][:-4]
			if imname not in labels_dict:
				continue
			imlabels = labels_dict[imname]
			zero_segments, pos = [1, 1], 0
			for i in imlabels:  # here we determine the 0 identity segments on the end of reads, which are junk
				if i == 0:
					zero_segments[pos] += 1
				else:
					pos=1
			imarray = ndimage.imread(fname, mode='RGB')

			empty, emptycount = True, 0
			for i in range(1, len(imarray)):
				for pix in imarray[i]:
					for val in pix:
						if val != 0.0:
							empty = False
							break
					if empty == False:
						break
				if empty == False:
					break
			if empty:  # ignore reads with no matching reads
				continue

			# break read into windows, excluding junk 0s at the ends
			sidelen = (args.window_size - args.segment_size) / 2  # extra space on each side of segment in window
			blanks = [[0,0,0]] * ((48 - args.window_size) / 2)
			for i in range(zero_segments[0], len(imlabels)-zero_segments[1]):
				startpos, endpos = (i*args.segment_size)-sidelen, ((i+1)*args.segment_size)+sidelen
				if startpos < 0:
					continue
				if endpos > len(imarray[0]):
					break
				window = imarray[:,startpos:endpos]
				
				if len(blanks) > 0:
					window = list(window)
					for j in range(len(window)):
						window[j] = list(window[j])
						window[j] = np.concatenate((blanks, window[j], blanks), axis=0)
					window = np.array(window)

				label = imlabels[i]
				if len(window) < 48:
					blankrows = [[[0,0,0]] * len(window[0])] * (48 - len(window))
					window = np.concatenate((window, blankrows), axis=0)
				data.append(window)
				labels.append(label)
				if len(data) == batch_size:
					if args.classify != -1.0:
						labels = [1.0 if i >= args.classify else 0.0 for i in labels]
					yield np.array(data), np.array(labels)
					yielded = True
					data, labels = [], []
			imcount += 1
			if args.debug > 0 and imcount >= args.debug:
				break
			if imcount == len(filenames) and yielded == False:
				print 'Error: no data found.'
				sys.exit()

		if len(data) != 0:  # send remainder batch to the neural network
			if args.classify != -1.0:
				labels = [1.0 if i >= args.classify else 0.0 for i in labels]
			yield np.array(data), np.array(labels)
			yielded = True
			data, labels = [], []


def run_network_generator(args):
	echo('Setting up generator...')
	train, valid, test, train_index, valid_index, labels_dict, labels = setup_generator(args)
	#print(test)
	#print('Generator test: ', str(len(test)))
	vgg = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=(48,args.window_size,3), pooling='max', extra=args.extra, classify=args.classify)
	if args.classify == -1.0:
		opt = optimizers.Adam(lr=0.0001)
		vgg.compile(loss='mean_squared_error', optimizer=opt)
	else:
		opt = optimizers.Adam(lr=0.00001)
		vgg.compile(loss='binary_crossentropy', optimizer=opt)

	#echo('Determining number of batches...')
	#data, svmdata, labels, train_index, valid_index = get_data(args, generating=True)

	echo('Fitting model...')
	vgg.fit_generator(generate_batches(args, train, labels_dict, 64), steps_per_epoch=int(train_index/64)+1, epochs=args.epochs, validation_data=generate_batches(args, valid, labels_dict, 64), validation_steps=int((valid_index-train_index)/64)+1)
	print '\n'; echo('Predicting...')
	predictions = vgg.predict_generator(generate_batches(args, test, labels_dict, 64), int((len(labels)-valid_index)/64)+1)

	print(len(predictions), len(labels[valid_index:]), len(labels))

	if args.classify == -1.0:
		eval_preds(args, labels[valid_index:], predictions)
	else:
		eval_preds_classify(labels[valid_index:], predictions, args.classify)

	if args.output != 'NONE':
		if not args.output.endswith('.hd5'):
			args.output += '.hd5'
		vgg.save(args.output)


def main():
	args = parseargs()
	if not args.input.endswith('/'):
		args.input += '/'
	if not (args.window_size >= args.segment_size):
		print 'Error: window size must be >= segment size.'
		sys.exit()
	if args.classify != -1.0 and (args.classify < 0.0 or args.classify > 1.0):
		print 'Error: Classification threshold must be a value from 0.0 to 1.0'
		sys.exit()
	if (args.test_input == 'NONE') ^ (args.test_labels == 'NONE'):
		print 'Error: Must specify both --test_input and --test_labels or neither.'
		sys.exit() 
	if args.segment_size % 2 != 0 or args.window_size % 2 != 0:
		print 'Error: segment_size and window_size must be even.'

	if args.load == 'NONE' and args.loadsvm == 'NONE':
		if args.streaming:
			run_network_generator(args)
		else:
			echo('Gathering input data...')
			data, svmdata, labels, train_index, valid_index = get_data(args)
			if len(data[:train_index]) == 0 or len(data[train_index:valid_index]) == 0 or len(data[valid_index:]) == 0:
				print 'Not enough input images for train/validation/test split. Use more data.'
				sys.exit()
			echo('Data gathering complete. Compiling CNN model...')
			run_network(args, data, svmdata, labels, train_index, valid_index)
	else:
		load_and_test(args)
	print ''; echo('Done')


if __name__== '__main__':
	main()
#