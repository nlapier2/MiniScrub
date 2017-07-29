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

import argparse, glob, sys
import numpy as np
from scipy import ndimage

import warnings

import keras
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, MaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K

from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from imagenet_utils import _obtain_input_shape


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def parseargs():
	parser = argparse.ArgumentParser(description='Create pileups from .paf read-to-read mapping and fastq reads.')
	parser.add_argument('--input', default='./', help='Directory with png pileup images. Default: current directory.')
	parser.add_argument('--labels', default='labels.txt', help='Path to image labels file.')
	parser.add_argument('--segment_size', type=int, default=100, help='Size of read segments to evaluate.')
	parser.add_argument('--window_size', type=int, default=200, help='Window size for VGG. Window >= segment size.')
	args = parser.parse_args()
	return args


def VGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None):

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

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

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
    x = Dense(4096, activation='softmax', name='fc1')(model.layers[-1].output)#(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='tanh', name='fc2')(x)
    x = Dropout(0.5)(x)
    #'''
    x = Dense(1, activation='tanh', name='predictions_new')(x)
    new_model = Model(inputs, x, name='vgg16_new')
    return new_model


def eval_preds(actual, predicted):
	print 'Actual:'
	print actual
	print '\nPredicted:'
	print predicted
	print
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
	outside10 = 100.0 - within1 - within5 - within10
	print str(within1) + ' percent of predictions within 1.0 of actual'
	print str(within5) + ' percent of predictions within 5.0 of actual'
	print str(within10) + ' percent of predictions within 10.0 of actual'		
	print str(outside10) + ' percent of predictions outside 10.0 from actual'


def main():
	args = parseargs()
	if not args.input.endswith('/'):
		args.input += '/'
	if not (args.window_size >= args.segment_size and args.segment_size % 100 == 0):
		print 'Error: window size must be >= segment size and segment size % 100 must = 0.'
		sys.exit()

	labels_dict = {}
	labels_file = open(args.labels, 'r')
	for line in labels_file:
		splits = line.strip().split(' ')
		if len(splits) < 2:
			continue
		labels_dict[splits[0]] = [float(i)/100.0 for i in splits[1].split(',')]
	labels_file.close()

	data, labels = [], []
	for fname in glob.glob(args.input+'*.png'):
		imname = fname.split('/')[-1][:-4]
		if imname not in labels_dict:
			continue
		imlabels = labels_dict[imname]
		zero_segments = 0
		for i in imlabels:
			if i == 0:
				zero_segments += 1
			else:
				break
		imarray = ndimage.imread(fname)
		for i in range(zero_segments, len(imlabels)-2):
			sidelen = args.segment_size / 2  # extra space on each side of segment in window
			if (i+2)*args.segment_size+sidelen > len(imarray[0]):
				break
			window = imarray[:,i*args.segment_size+sidelen:(i+2)*args.segment_size+sidelen]
			label = imlabels[i+1]
			data.append(window)
			labels.append(label)
	data, labels = data[:64], labels[:64]  # debug mode
	data = np.array(data)
	labels = np.array(labels)
	train_index, valid_index = int(0.6*len(data)), int(0.8*len(data))

	avg = [0.0, 0.0, 0.0]
	for image in data:
		for row in image:
			for pixel in row:
				avg = [avg[i]+pixel[i] for i in range(len(pixel))]
	numpixels = float(data.shape[0] * data.shape[1] * data.shape[2])
	avg = [avg[i]/numpixels for i in range(len(avg))]

	for image in data:
		for row in image:
			for pixel in row:
				pixel = [pixel[i]-avg[i] for i in range(len(pixel))]

	#vgg = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=data[0].shape, pooling='max')
	vgg = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=data[0].shape, pooling='max')
	vgg.compile(optimizer='adadelta', loss='mean_squared_error')
	vgg.fit(data[:train_index], labels[:train_index], epochs=50, validation_data=(data[train_index:valid_index], labels[train_index:valid_index]), batch_size=64)
	print 'Predicting...'
	predictions = vgg.predict(data[valid_index:], batch_size=64)
	eval_preds(labels[valid_index:], predictions)


if __name__=='__main__':
	main()
#