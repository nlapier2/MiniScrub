'''
Author: Nathan LaPierre

Modified version of the keras implementation of VGG16.
The keras code is open-source under the MIT license,
	& the VGG architecture and weights are 
	available under the Creative Commons License.
Model adapted for regression and methods for generating
	inputs and gathering results added.
'''

#from __future__ import print_function
from __future__ import absolute_import

import argparse, glob, os, sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC, SVR
from sklearn.externals import joblib
import pandas as pd

import warnings

import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras import optimizers

from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from imagenet_utils import _obtain_input_shape


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


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
	parser.add_argument('--segment_size', type=int, default=100, help='Size of read segments to evaluate.')
	parser.add_argument('--test_input', default='NONE', help='Directory of serparate set of images to test model on.')
	parser.add_argument('--test_labels', default='NONE', help='Path to file with labels for images in test set.')
	parser.add_argument('--window_size', type=int, default=200, help='Window size for VGG. Window >= segment size.')
	args = parser.parse_args()
	return args


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

    # we now strip off the old 1000 class layer and add a single relu output for regression
    # model.layers.pop()
    #'''
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
    #'''
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


def eval_preds(actual, predicted, baseline=False):
	if not baseline:
		predicted = np.array([i[0] for i in predicted])
	#print 'Actual:'
	#print actual
	#print '\nPredicted:'
	#print predicted
	#print
	errors = [abs(actual[i]-predicted[i]) for i in range(len(actual))]
	print 'Average error: ' + str(np.mean(errors))
	mse = np.mean([i**2 for i in errors])
	print 'Mean squared error: ' + str(mse)
	percented, within1, within5, within10 = 100.0 / float(len(actual)), 0.0, 0.0, 0.0
	for i in range(len(actual)):
		if errors[i] < 0.01:
			within1 += percented
		if errors[i] < 0.05:
			within5 += percented
		if errors[i] < 0.1:
			within10 += percented
	print
	print str(within1) + ' percent of predictions within 1.0 of actual'
	print str(within5) + ' percent of predictions within 5.0 of actual'
	print str(within10) + ' percent of predictions within 10.0 of actual'		
	print str(100.0 - within10) + ' percent of predictions outside 10.0 from actual'
	print
	print 'Pearson correlation: ' + str(pearsonr(actual, predicted)[0])
	print 'Spearman rank correlation: ' + str(spearmanr(actual, predicted)[0])
	print
	print 'Classification metrics for various cutoff thresholds:\n'
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


def process_images(args, labels_dict):
	data, svmdata, labels, endpoints = [], [], [], []  # endpoints is position where full reads end
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

		# break read into windows, excluding junk 0s at the ends
		sidelen = (args.window_size - args.segment_size) / 2  # extra space on each side of segment in window
		prev_end = 0
		for i in range(zero_segments[0], len(imlabels)-zero_segments[1]):
			startpos = max(prev_end,(i*args.segment_size))-sidelen
			endpos = startpos + args.segment_size + 2*sidelen

			place = startpos
			while place < endpos:
				if imarray[place][0][0] == 0.0:
					endpos += 1
				place += 1
			prev_end = endpos

			if startpos < 0:
				continue
			if endpos > len(imarray[0]):
				break
			window = imarray[:,startpos:endpos]
			label = imlabels[i]#+1]
			#if label < 0.7:
			#	label *= (1-(0.7-label))**(label*10)
			data.append(window)
			labels.append(label)
			if args.baseline:
				vec, counts = [], 0
				for col in range(len(window[0])):
					counts = 0
					for row in range(len(window)):
						if window[row][col][0] > 128.0:
							counts += 1
					vec.append(counts)
				svmdata.append(vec)
		endpoints.append(len(data))
		if args.debug > 0 and len(labels) >= args.debug:
			break

	if len(data) == 0 or len(labels) == 0:
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
	return data, svmdata, labels, endpoints


def get_data(args, testing=False):
	if testing == True:  # get data for the test images provided
		args.labels = args.test_labels
		args.input = args.test_input

	labels_dict = {}
	labels_file = open(args.labels, 'r')
	for line in labels_file:
		splits = line.strip().split(' ')
		if len(splits) < 2:
			continue
		#labels_dict[splits[0]] = [float(i)/100.0 for i in splits[1].split(',')]
		labels_dict[splits[0]] = [float(i) for i in splits[1].split(',')]
	labels_file.close()

	data, svmdata, labels, endpoints = process_images(args, labels_dict)

	# we set the indices to split the data into train/validation/test
	# and round these up to the end of the nearest read to prevent overfitting
	train_index, valid_index = int(0.6*len(data)), int(0.8*len(data))
	for point in endpoints:
		if point >= train_index:
			train_index = point
			break
	for point in endpoints:
		if point >= valid_index:
			valid_index = point
			break

	return data, svmdata, labels, train_index, valid_index


def run_network(args, data, svmdata, labels, train_index, valid_index):
	#vgg = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=data[0].shape, pooling='max')
	vgg = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=data[0].shape, pooling='max', extra=args.extra, classify=args.classify)
	if args.classify == -1.0:
		opt = optimizers.Adam(lr=0.0001)
		vgg.compile(loss='mean_squared_error', optimizer=opt)  #, optimizer='adadelta')
		#vgg.compile(loss='mean_squared_error', optimizer='adadelta')
	else:
		#opt = optimizers.Adadelta(lr=0.1)
		opt = optimizers.Adam(lr=0.00001)
		vgg.compile(loss='binary_crossentropy', optimizer=opt)
	vgg.fit(data[:train_index], labels[:train_index], epochs=args.epochs, validation_data=(data[train_index:valid_index], labels[train_index:valid_index]), batch_size=64)
	print '\nPredicting...'
	predictions = vgg.predict(data[valid_index:], batch_size=64)
	if args.classify == -1.0:
		eval_preds(labels[valid_index:], predictions)
		if args.baseline:
			print '\n\nSVM Baseline:'
			svm = SVR()
			svm.fit(svmdata[:train_index], labels[:train_index])
			svm_predictions = svm.predict(svmdata[valid_index:])
			eval_preds(labels[valid_index:], svm_predictions, baseline=True)
	else:
		eval_preds_classify(labels[valid_index:], predictions, args.classify)
		if args.baseline:
			print '\n\nSVM Baseline:'
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
	print 'Processing test data...'
	if args.loadsvm != 'NONE':
		args.baseline = True
	data, svmdata, labels, train_index, valid_index = get_data(args, testing=True)
	if args.debug <= 0:
		args.debug = len(data)

	if args.load != 'NONE':
		vgg = load_model(args.load)
		print 'Neural network model loaded successfully.'
		print 'Predicting...'
		predictions = vgg.predict(data[:args.debug], batch_size=64)
		if args.classify == -1.0:
			eval_preds(labels[:args.debug], predictions)
		else:
			eval_preds_classify(labels[:args.debug], predictions, args.classify)

	if args.loadsvm != 'NONE':
		svm = joblib.load(args.loadsvm)
		print '\n\nSVM model loaded successfully.'
		predictions = svm.predict(svmdata[:args.debug])
		print 'SVM Predictions:'
		if args.classify == -1.0:
			eval_preds(labels[:args.debug], predictions, baseline=True)
		else:
			eval_preds_classify(labels[:args.debug], predictions, args.classify)


def main():
	args = parseargs()
	if not args.input.endswith('/'):
		args.input += '/'
	if not (args.window_size >= args.segment_size):# and args.segment_size % 100 == 0):
		print 'Error: window size must be >= segment size.'#' and segment size % 100 must = 0.'
		sys.exit()
	if args.classify != -1.0 and (args.classify < 0.0 or args.classify > 1.0):
		print 'Error: Classification threshold must be a value from 0.0 to 1.0'
		sys.exit()
	if (args.test_input == 'NONE') ^ (args.test_labels == 'NONE'):
		print 'Must specify both --test_input and --test_labels or neither.'
		sys.exit() 

	if args.load == 'NONE' and args.loadsvm == 'NONE':
		data, svmdata, labels, train_index, valid_index = get_data(args)
		if len(data[:train_index]) == 0 or len(data[train_index:valid_index]) == 0 or len(data[valid_index:]) == 0:
			print 'Not enough input images for train/validation/test split. Use more data.'
			sys.exit()
		run_network(args, data, svmdata, labels, train_index, valid_index)
	else:
		load_and_test(args)


if __name__=='__main__':
	main()
#