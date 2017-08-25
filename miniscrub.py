import argparse, glob, gzip, math, os, sys
import numpy as np
from scipy import ndimage

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras
from keras.models import load_model


def parseargs():
	parser = argparse.ArgumentParser(description='Use saved keras model to scrub reads.')
	parser.add_argument('--compression', default='none', choices=['none', 'gzip'], help='Compression of reads file ("none" or "gzip").')
	parser.add_argument('--cutoff', default=0.0, type=float, help='Scrub read segments below this percent identity.')
	parser.add_argument('--debug', default=0, type=int, help='Number of images to use in debug mode. If <= 0, non-debug mode (default).')
	parser.add_argument('--input', default='./', help='Directory with png pileup images. Default: current directory.')
	parser.add_argument('--labels', default='NONE', help='Path to image labels file. If provided, will NOT trim. Labels must correspond with segment_size.')
	parser.add_argument('--load', required=True, help='Path to keras model file to load. Required.')
	parser.add_argument('--output', default='scrubbed-reads.fastq', help='File to write scrubbed reads to.')
	parser.add_argument('--reads', default='NONE', help='Path to reads file. Default: do not trim (output statsitics instead).')
	parser.add_argument('--segment_size', default=100, help='Neural net segment size to predict. Keep as default unless network retrained.')
	parser.add_argument('--window_size', default=200, help='Neural net window size to predict. Keep as default unless network retrained.')
	args = parser.parse_args()
	return args


def process_images(args, labels_dict, testing=False):
	data, svmdata, labels, locations, endpoints = [], [], [], [], []  # endpoints is position where full reads end
	for fname in glob.glob(args.input+'*.png'):
		imname = fname.split('/')[-1][:-4]
		zero_segments, pos = [1, 1], 0
		if testing == True and imname not in labels_dict:
			continue
		elif testing == True:
			imlabels = labels_dict[imname]
			for i in imlabels:  # here we determine the 0 identity segments on the end of reads, which are junk
				if i == 0:
					zero_segments[pos] += 1
				else:
					pos=1
		imarray = ndimage.imread(fname, mode='RGB')

		# break read into windows, excluding junk 0s at the ends
		sidelen = (args.window_size - args.segment_size) / 2  # extra space on each side of segment in window
		prev_end, num_segments = 0, int(math.ceil(float(len(imarray[0])) / float(args.segment_size)))
		for i in range(zero_segments[0], num_segments-zero_segments[1]):
			startpos, endpos = (i*args.segment_size)-sidelen, ((i+1)*args.segment_size)+sidelen
			if startpos < 0:
				continue
			if endpos > len(imarray[0]):
				break
			window = imarray[:,startpos:endpos]
			if testing == True:
				label = imlabels[i]#+1]
				labels.append(label)
			#if label < 0.7:
			#	label *= (1-(0.7-label))**(label*10)
			data.append(window)
			'''if imname in locations:
				locations[imname].append(str(startpos)+'-'+str(endpos))
			else:
				locations[imname] = [str(startpos)+'-'+str(endpos)]'''
			locations.append(str(imname)+' | '+str(startpos)+' | '+str(endpos))
		endpoints.append(len(data))
		if args.debug > 0 and len(endpoints) >= args.debug:
			break

	if len(data) == 0:
		print 'Error: no data found.'
		sys.exit()

	data = np.array(data)
	svmdata = np.array(svmdata)
	labels = np.array(labels)
	return data, svmdata, labels, endpoints, locations


def get_data(args, testing=False):
	labels_dict = {}
	if testing == True:
		labels_file = open(args.labels, 'r')
		for line in labels_file:
			splits = line.strip().split(' ')
			if len(splits) < 2:
				continue
			#labels_dict[splits[0]] = [float(i)/100.0 for i in splits[1].split(',')]
			labels_dict[splits[0]] = [float(i) for i in splits[1].split(',')]
		labels_file.close()

	data, svmdata, labels, endpoints, locations = process_images(args, labels_dict, testing)
	return data, svmdata, labels, locations


def eval_preds(actual, predicted, baseline=False):
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


def load_and_test(args):
	print 'Processing input data...'
	if args.labels != 'NONE':
		data, svmdata, labels, locations = get_data(args, testing=True)
	else:
		data, svmdata, labels, locations = get_data(args, testing=False)

	print 'Loading model...'
	model = load_model(args.load)
	print 'Model loaded successfully. Predicting...'
	predictions = model.predict(data, batch_size=64)
	predictions = np.array([i[0] for i in predictions])

	if args.labels != 'NONE':
		print 'Evaluating predictions on provided labels...'
		eval_preds(labels[:args.debug], predictions)
	return model, predictions, locations


def output_statistics(args, predictions):
	print '\nTotal number of predictions made: ' + str(len(predictions))
	print 'Average prediction (of percentage of correct bases per 100bp segment): ' + str(100.0*np.mean(predictions))
	print 'Median prediction: ' + str(100.0*np.median(predictions))
	print '\nPercentage of 100bp segments to be scrubbed at different cutoff points:'
	cutoffs = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
	for cutoff in cutoffs:
		print str(cutoff) + ': ' + str(100.0*float(sum([1 if pred <= cutoff else 0 for pred in predictions])) / float(len(predictions)))


def locate_predictions(predictions, locations):
	pred_locs = {}
	for loc in range(len(locations)):
		name, start, end = locations[loc].split(' | ')
		if name not in pred_locs:
			pred_locs[name] = [[predictions[loc], int(start), int(end)]]
		else:
			pred_locs[name].append([predictions[loc], int(start), int(end)])
	return pred_locs


def scrub_read(read, pred_locs, cutoff):
	scrubbed_read = ''
	if len(scrubbed_read) < pred_locs[0][1]:
		scrubbed_read = read[:pred_locs[0][1]]
	for i in range(len(pred_locs)):
		pred, start, end = pred_locs[i]
		if pred > cutoff:
			scrubbed_read += read[start:end]
	if len(read) > pred_locs[-1][2]:
		scrubbed_read += read[pred_locs[-1][2]:]
	return scrubbed_read


def output_reads(args, pred_locs):
	if args.compression == 'none':
		f = open(args.reads, 'r')
	else:
		f = gzip.open(args.reads, 'r')
	outfile = open(args.output, 'w')

	cur_read, read_line, scrubbed_read, scrubbed_quals, line3, num = '', '', '', '', '', -1
	for line in f:
		num = (num + 1) % 4
		if num == 0:
			cur_read = line[1:].split(' ')[0].strip()
			read_line = line.strip()
		elif num == 1:
			if cur_read in pred_locs:
				scrubbed_read = scrub_read(line.strip(), pred_locs[cur_read], args.cutoff)
			else:
				scrubbed_read = ''
		elif num == 2:
			line3 = line.strip()
		elif len(scrubbed_read) > args.segment_size:
			scrubbed_quals = scrub_read(line.strip(), pred_locs[cur_read], args.cutoff)
			outfile.write('\n'.join([read_line, scrubbed_read, line3, scrubbed_quals]) + '\n')
	f.close(); outfile.close()


def main():
	args = parseargs()
	if not args.input.endswith('/'):
		args.input += '/'
	if args.reads == 'NONE':
		print 'Reads to trim not provided. Will print some statsitics.'
	elif args.cutoff <= 0.0 or args.cutoff > 1.0:
		print 'If --reads specified, must specify --cutoff in range [0.0, 1.0)'
		sys.exit()

	model, predictions, locations = load_and_test(args)
	pred_locs = locate_predictions(predictions, locations)

	if args.labels == 'NONE':
		if args.reads == 'NONE':
			output_statistics(args, predictions)
		else:
			print 'Scrubbing reads...'
			output_reads(args, pred_locs)


if __name__ == '__main__':
	main()
#