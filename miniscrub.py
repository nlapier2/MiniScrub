import argparse, glob, gzip, math, os, sys, time
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras
from keras.models import load_model


start = time.time()


def echo(msg):
	global start
	seconds = time.time() - start
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	hms = "%02d:%02d:%02d" % (h, m, s)
	print '['+hms+'] ' + msg


def parseargs():
	parser = argparse.ArgumentParser(description='Use saved keras model to scrub reads.')
	parser.add_argument('--compression', default='none', choices=['none', 'gzip'], help='Compression of reads file ("none" or "gzip").')
	parser.add_argument('--cutoff', default=0.0, type=float, help='Scrub read segments below this percent identity.')
	parser.add_argument('--debug', default=0, type=int, help='Number of images to use in debug mode. If <= 0, non-debug mode (default).')
	parser.add_argument('--input', default='./', help='Directory with png pileup images. Default: current directory.')
	parser.add_argument('--labels', default='NONE', help='Path to image labels file. If provided, will NOT trim. Labels must correspond with segment_size.')
	parser.add_argument('--limit_length', default=0, type=int, help='Optionally do not include reads above a certain length.')
	parser.add_argument('--limit_paf', default=0, type=int, help='Optionally limit the number of reads from paf file (if --paf is used).')
	parser.add_argument('--load', required=True, help='Path to keras model file to load. Required.')
	parser.add_argument('--min_length', default=500, type=int, help='Minimum length of reads to keep.')
	parser.add_argument('--mode', default='minimizers', choices=['minimizers', 'whole'], help='Whether pileups are minimizers-only or whole reads.')
	parser.add_argument('--output', default='scrubbed-reads.fastq', help='File to write scrubbed reads to.')
	parser.add_argument('--paf', default='NONE', help='Path to paf file; required if --mode=minimizers and --reads is specified.')
	parser.add_argument('--reads', default='NONE', help='Path to reads file. Default: do not trim (output statsitics instead).')
	parser.add_argument('--segment_size', default=48, type=int, help='Neural net segment size to predict. Keep as default unless network retrained.')
	parser.add_argument('--streaming', action='store_true', help='Streaming loading of images to reduce memory footprint.')
	parser.add_argument('--window_size', default=72, type=int, help='Neural net window size to predict. Keep as default unless network retrained.')
	args = parser.parse_args()
	return args


def read_paf(fname, compression, limit_paf, limit_length):
	if compression == 'none':
		paf = open(fname, 'r')
	else:
		paf = gzip.open(fname, 'r')

	linecount = 0
	minimizers = {}
	for line in paf:
		splits = line.strip().split('\t')
		if splits[0] == splits[5]:  # read mapped against itself
			if splits[0] in minimizers or (limit_length > 0 and int(splits[1]) > limit_length):
				continue
			if splits[-2][5] == 'I':
				minimizers[splits[0]] = [int(i) for i in splits[-2][6:].split(',')]
			else:
				minimizers[splits[0]] = [int(i) for i in splits[-2][5:].split(',')]

			linecount += 1
			if linecount % 10000 == 0:
				echo('Done reading ' + str(linecount) + ' lines from paf')
			if limit_paf > 0 and linecount % limit_paf == 0:
				paf.close()
				return minimizers

	paf.close()
	return minimizers


def process_images(args, labels_dict, testing=False, fnames=None):
	data, svmdata, labels, locations, endpoints = [], [], [], [], []  # endpoints is position where full reads end
	if fnames == None:
		fnames = glob.glob(args.input+'*.png')
	for fname in fnames:
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
		blanks = [[0,0,0]] * ((48 - args.window_size) / 2)
		for i in range(zero_segments[0], num_segments-zero_segments[1]):
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
			if len(window) < 48:
				blankrows = [[[0,0,0]] * len(window[0])] * (48 - len(window))
				window = np.concatenate((window, blankrows), axis=0)

			if testing == True:
				label = imlabels[i]
				labels.append(label)
			data.append(window)
			locations.append(str(imname)+' | '+str(startpos+sidelen)+' | '+str(endpos-sidelen-1))
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


def get_data(args, testing=False, fnames=None):
	labels_dict = {}
	if testing == True:
		labels_file = open(args.labels, 'r')
		for line in labels_file:
			splits = line.strip().split(' ')
			if len(splits) < 2:
				continue
			labels_dict[splits[0]] = [float(i) for i in splits[1].split(',')]
		labels_file.close()

	data, svmdata, labels, endpoints, locations = process_images(args, labels_dict, testing, fnames)
	return data, svmdata, labels, locations


def eval_preds(actual, predicted, baseline=False):
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


def load_and_test(args):
	if not args.streaming:
		echo('Processing input images...')
		if args.labels != 'NONE':
			data, svmdata, labels, locations = get_data(args, testing=True)
		else:
			data, svmdata, labels, locations = get_data(args, testing=False)

		echo('Loading model...')
		model = load_model(args.load)
		echo('Model loaded successfully. Predicting...')
		predictions = model.predict(data, batch_size=64)
		predictions = np.array([i[0] for i in predictions])
	else:
		fnames = glob.glob(args.input+'*.png')
		if args.debug > 0:
			fnames = fnames[:args.debug]
		counter, predictions, locations = 0, [], []
		echo('Loading model...')
		model = load_model(args.load)
		echo('Model loaded successfully. Predicting...')
		while counter < len(fnames):
			batch = fnames[counter:counter+100]
			if args.labels != 'NONE':
				data, svmdata, labels, locs = get_data(args, testing=True, fnames=batch)
			else:
				data, svmdata, labels, locs = get_data(args, testing=False, fnames=batch)
			locations.extend(locs)
			
			preds = model.predict(data, batch_size=64)
			preds = np.array([i[0] for i in preds])
			predictions.extend(preds)
			counter += 100

	if args.labels != 'NONE':
		echo('Evaluating predictions on provided labels...')
		eval_preds(labels, predictions)
	return model, predictions, locations


def locate_predictions(predictions, locations, minimizers):
	pred_locs, deletions = {}, 0
	for loc in range(len(locations)):
		loc -= deletions
		name, start, end = locations[loc].split(' | ')
		start, end = int(start), int(end)
		if minimizers != {} and name not in minimizers:
			del predictions[loc], locations[loc]
			deletions += 1
			continue
		elif minimizers != {}:
			try:
				start, end = minimizers[name][start], minimizers[name][end]
			except:
				print minimizers[name]
				print name, start, end, len(minimizers[name])
				sys.exit()
		if name not in pred_locs:
			pred_locs[name] = [[predictions[loc], start, end]]
		else:
			pred_locs[name].append([predictions[loc], start, end])
	return pred_locs, predictions, locations


def output_statistics(args, predictions, pred_locs):
	print '\nTotal number of predictions made: ' + str(len(predictions))
	print 'Average prediction (of percentage of correct bases per read segment): ' + str(100.0*np.mean(predictions))
	print 'Median prediction: ' + str(100.0*np.median(predictions))
	print '\nEstimated percentage of read segments to be scrubbed at different cutoff points:'
	print '(Takes into account --min_length='+str(args.min_length)+')'
	cutoffs = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
	for cutoff in cutoffs:
		cur_segments, cur_length, total_segments = 0, 0, 0
		for readname in pred_locs:
			for segment in pred_locs[readname]:
				pred, start, end = segment
				if pred > cutoff:
					cur_segments += 1
					cur_length += (end - start)
				else:
					if cur_length >= args.min_length:
						total_segments += cur_segments
					cur_segments, cur_length = 0, 0
			if cur_length >= args.min_length:
				total_segments += cur_segments
			cur_segments, cur_length = 0, 0
		print str(cutoff) + ': ' + str(100.0 - (100.0 * (float(total_segments) / float(len(predictions)))))


def scrub_read(args, read, pred_locs, cutoff):
	scrubbed_reads, locs = [''], [[-1, -1]]
	if len(scrubbed_reads[0]) < pred_locs[0][1]:
		scrubbed_reads[0] = read[:pred_locs[0][1]]
		locs[0] = [0, pred_locs[0][1]]
	for i in range(len(pred_locs)):
		pred, start, end = pred_locs[i]
		if pred > cutoff:
			scrubbed_reads[-1] += read[start:end]
			if locs[-1][0] == -1:
				locs[-1][0] = start
			locs[-1][1] = end
		else:
			scrubbed_reads.append('')
			locs.append([-1, -1])
	if len(read) > pred_locs[-1][2]:
		scrubbed_reads[-1] += read[pred_locs[-1][2]:]
		if locs[-1][0] == 0:
			locs[-1][0] = pred_locs[-1][2]
		locs[-1][1] = len(read)
	selections = [i for i in range(len(scrubbed_reads)) if len(scrubbed_reads[i]) >= args.min_length]
	scrubbed_reads = [scrubbed_reads[i] for i in selections]
	locs = [locs[i] for i in selections]
	#scrubbed_reads = [i for i in scrubbed_reads if len(i) >= args.min_length]
	return scrubbed_reads, locs


def output_reads(args, pred_locs):
	if args.compression == 'none':
		f = open(args.reads, 'r')
	else:
		f = gzip.open(args.reads, 'r')
	outfile = open(args.output, 'w')

	cur_read, read_line, readlen, line3, num = '', '', 0, '', -1
	scrubbed_reads, scrubbed_quals, locs = [], [], []
	for line in f:
		num = (num + 1) % 4
		if num == 0:
			cur_read = line[1:].split(' ')[0].strip()
			read_line = line[len(cur_read)+1:].strip()
		elif num == 1:
			if cur_read in pred_locs:
				readlen = len(line.strip())
				scrubbed_reads, locs = scrub_read(args, line.strip(), pred_locs[cur_read], args.cutoff)
			else:
				scrubbed_reads = []
		elif num == 2:
			line3 = line.strip()
		elif len(scrubbed_reads) > 0:
			scrubbed_quals, locs = scrub_read(args, line.strip(), pred_locs[cur_read], args.cutoff)
			for i in range(len(scrubbed_reads)) or len(scrubbed_reads[0]) < readlen:
				if len(scrubbed_reads) > 1:
					line1 = '@' + cur_read + '_bases-' + str(locs[i][0]) + '-to-' + str(locs[i][1]) + ' ' + read_line
				else:
					line1 = '@' + cur_read + ' ' + read_line
				outfile.write('\n'.join([line1, scrubbed_reads[i], line3, scrubbed_quals[i]]) + '\n')
	f.close(); outfile.close()


def main():
	args = parseargs()
	if not args.input.endswith('/'):
		args.input += '/'
	if args.reads == 'NONE':
		print 'Reads to trim not provided. Will print some statsitics.'
	elif args.cutoff <= 0.0 or args.cutoff > 1.0:
		print 'Error: If --reads specified, must specify --cutoff in range (0.0, 1.0]'
		sys.exit()
	elif args.labels == 'NONE' and args.mode =='minimizers' and args.paf == 'NONE':
		print 'Error: If --mode=minimizers (default), --paf must be specified.'
		sys.exit()
	if args.min_length < 1:
		print 'Error: Minimum read length to keep must be at least 1.'
		sys.exit()

	minimizers = {}
	if args.labels == 'NONE' and args.mode == 'minimizers':
		echo('Minimizers mode selected. Reading paf file...')
		minimizers = read_paf(args.paf, args.compression, args.limit_paf, args.limit_length)

	model, predictions, locations = load_and_test(args)
	if args.labels == 'NONE':
		echo('Predictions made. Locating segments to cut...')
		predictions = list(predictions)
		pred_locs, predictions, locations = locate_predictions(predictions, locations, minimizers)
		if args.reads == 'NONE':
			output_statistics(args, predictions, pred_locs)
		else:
			echo('Scrubbing reads...')
			output_reads(args, pred_locs)
	echo('Done.')


if __name__ == '__main__':
	main()
#