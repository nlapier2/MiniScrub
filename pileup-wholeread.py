import argparse, gc, gzip, multiprocessing, random, sys
import numpy as np
import scipy.misc


def parse_args():  # handle user arguments
	parser = argparse.ArgumentParser(description='Create pileups from .paf read-to-read mapping and fastq reads.')
	parser.add_argument('--avgdepth', type=int, default=1000, help='Average coverage depth (not currently used).')
	parser.add_argument('--compression', default='none', choices=['none', 'gzip'], help='Compression format, or none')
	parser.add_argument('--debug', action='store_true', help='Debug mode.')
	parser.add_argument('--limit_fastq', type=int, default=0, help='Limit number of reads to scan from fastq file.')
	parser.add_argument('--limit_reads', type=int, default=0, help='Limit number of reads to generate pileups for.')
	parser.add_argument('--mapping', required=True, help='Path to the .paf file of read-to-read mappings.')
	parser.add_argument('--maxdepth', type=int, default=48, help='Maximum number of matched reads per pileup image.')
	parser.add_argument('--plotdir', default='./', help='If --saveplots is used, directory path to save plots in.')
	parser.add_argument('--processes', type=int, default=1, help='Number of multiple processes to run concurrently.')
	parser.add_argument('--reads', required=True, help='Path to the .fastq reads file.')
	parser.add_argument('--saveplots', action='store_true', help='If used, will plot the pileups and save them.')
	parser.add_argument('--verbose', action='store_true', help='Verbose output option.')
	args = parser.parse_args()
	return args


def process_reads(reads, compression, limit, verbose):  # using fastq file, map read names to sequence and quality scores
	if verbose:
		print('Reading reads data from: ' + str(reads))
	if compression == 'gzip':
		reads_file = gzip.open(reads, 'r')
	else:
		reads_file = open(reads, 'r')

	reads_df, num, current = {}, -1, ''
	read_count = 0
	for line in reads_file:
		if args.compression == 'gzip':
			line = line.decode('utf8').strip()
		num = (num + 1) % 4
		if num == 0:
			current = line[1:].split(' ')[0]
			read_count += 1
			if read_count % 10000 == 0 and verbose:
				print('Done with ' + str(read_count) + ' reads')
		#elif num == 1:
		#	reads_df[current] = [line.upper()]
		elif num == 3:
			#scores = [int((ord(ch)-33)*2.75) for ch in line]
			#scores = [(ord(ch)-33) for ch in line]
			scores = [ord(ch) for ch in line]
			reads_df[current] = [scores, np.mean(scores)]
			if limit > 0 and read_count > limit:
				break
	reads_file.close()

	if verbose:
		print('Done reading: ' + str(reads))
	return reads_df


def make_pileup_bw(pid, readname, readqual, readlen, matches, args):
	try: 
		#print('Process ' + str(pid) + ' starting.')
		maxdepth, saveplots, plotdir, debug, pileup = args.maxdepth, args.saveplots, args.plotdir, args.debug, []
		minimizers = matches[0][12]
		pileup.append([128.0 + readqual] * readlen)  # the reference read
		for i in range(maxdepth):
			pileup.append([0.0])  # fill in placeholder lines
		#print('Process ' + str(pid) + ' at step 1.')

		selections, depth_order, depth_index, num = list(range(1,len(matches))), list(range(1, maxdepth+1)), 0, 0
		random.shuffle(selections); random.shuffle(depth_order)
		selections = selections[:maxdepth]
		#print('Process ' + str(pid) + ' at step 2.')

		for s in selections:
			selection = matches[s]
			prefix = [0.0] * int(selection[2])
			suffix = [0.0] * int(readlen-int(selection[3]))
			pixels = [selection[-2]] * (readlen - len(prefix) - len(suffix))
			seq = prefix + pixels + suffix

			#print('Process ' + str(pid) + ' at step 2.1.')
			for i in range(len(minimizers)):
				if minimizers[i] < selection[12][0] or minimizers[i] > selection[12][-1]:  # read does not cover this minimizer
					continue
				if minimizers[i] in selection[12]:  # if that minimizer matched by this read
					#print('Process ' + str(pid) + ' at step 2.1.1.')
					if i+1 < len(minimizers):
						seq[minimizers[i]:minimizers[i+1]] = [i+128.0 for i in seq[minimizers[i]:minimizers[i+1]]]
					elif minimizers[i]+15 < len(seq):
						seq[minimizers[i]:minimizers[i]+15] = [i+128.0 for i in seq[minimizers[i]:minimizers[i]+15]]
					else:
						seq[minimizers[i]:len(seq)] = [i+128.0 for i in seq[minimizers[i]:len(seq)]]
			pileup[depth_order[depth_index]] = seq
			depth_index += 1
			if depth_index >= maxdepth:
				break
		#print('Process ' + str(pid) + ' at step 3.')

		for line in range(len(pileup)):
			pileup[line].extend([0.0] * (readlen - len(pileup[line])))
		pileup = np.array(pileup)
		if saveplots:
			scipy.misc.toimage(pileup, cmin=0.0, cmax=255.0).save(plotdir+readname+'.png')
		#print('Process ' + str(pid) + ' done.')
		return 0
	except:
		print('Error in process ' + str(pid))
		return 1


def make_pileup_rgb(pid, readname, readqual, readlen, matches, args):
	avgdepth, maxdepth, saveplots, plotdir = args.avgdepth, args.maxdepth, args.saveplots, args.plotdir
	pileup, total, pixel = [], 0, [100.0, readqual, 100.0]
	seq = [pixel] * readlen
	pileup.append(seq)
	for i in range(maxdepth):
		pileup.append([[0.0,0.0,0.0]])  # fill in placeholder lines

	selections, depth_order, depth_index, num = range(len(matches)), range(1, maxdepth+1), 0, 0
	random.shuffle(selections); random.shuffle(depth_order)
	selections = selections[:maxdepth]

	for s in selections:
		selection = matches[s]
		prefix = [[0.0,0.0,0.0]] * int(selection[2])
		suffix = [[0.0,0.0,0.0]] * int(readlen-int(selection[3]))
		seq = [list(selection[13:16])] * int(selection[3] - selection[2])
		pileup[depth_order[depth_index]] = prefix + seq + suffix
		depth_index += 1
		if depth_index >= maxdepth:
			break

	for line in range(len(pileup)):
		pileup[line].extend([[0.0,0.0,0.0]] * (readlen - len(pileup[line])))
	pileup = np.array(pileup)
	if saveplots:
		scipy.misc.toimage(pileup, cmin=0.0, cmax=255.0, mode='RGB').save(plotdir+readname+'.png')
	return 0


if __name__ == "__main__":
	args = parse_args()
	if not args.plotdir.endswith('/'):
		args.plotdir += '/'
	args.maxdepth -= 1
	if args.debug:
		args.limit_fastq, args.limit_reads, args.saveplots = 10000, 10, True
	read_count, line_count, window_size = 0, 0, 200
	reads_df = process_reads(args.reads, args.compression, args.limit_fastq, args.verbose)
	reads_list = list(reads_df)

	context = multiprocessing.get_context("spawn")
	pool = context.Pool(processes=args.processes)#, maxtasksperchild=100)
	#pool = multiprocessing.Pool(processes=args.processes, maxtasksperchild=100)

	if args.verbose:
		print('Generating pileups...')
	read_data, cur_read = {}, ''
	if args.compression == 'gzip':
		f = gzip.open(args.mapping, 'r')
	else:
		f = open(args.mapping, 'r')

	for line in f:
		if args.compression == 'gzip':
			line = line.decode('utf8')
		splits = line.strip().split('\t')
		splits = splits[:12] + [splits[13]]
		for i in (1,2,3,6,7,8,9,10,11):
			splits[i] = float(splits[i])
		splits[12] = [int(i) for i in splits[12][5:].split(',')]
		if splits[0] not in reads_df or splits[5] not in reads_df:
			continue
		splits += [splits[9] / splits[10] * 100.0, reads_df[splits[5]][1], ((splits[3]-splits[2])/(splits[8]-splits[7]))*50.0]#100.0]
		if args.limit_fastq > 0 and (splits[0] not in reads_list or splits[5] not in reads_list):
			continue
		if read_data != {} and cur_read != splits[0]:
			readqual, readlen = reads_df[cur_read][1], len(reads_df[cur_read][0])
			pool.apply_async(make_pileup_bw, (read_count, cur_read, readqual, readlen, read_data, args,))
			read_count += 1
			if read_count % 1000 == 0 and args.verbose:
				print('Finished pileups for ' + str(read_count) + ' lines')
			if args.limit_reads > 0 and read_count >= args.limit_reads:
				break
			read_data, line_count = {}, 0

		read_data[line_count] = splits
		cur_read = splits[0]
		line_count += 1

	if read_data != {} and (read_count < args.limit_reads or args.limit_reads == 0):
		readqual, readlen = reads_df[cur_read][1], len(reads_df[cur_read][0])
		pool.apply_async(make_pileup_bw, (read_count, cur_read, readqual, readlen, read_data, args,))
		read_count += 1
		if read_count % 1000 == 0 and args.verbose:
			print('Finished pileups for ' + str(read_count) + ' lines')	

	f.close()
	pool.close()
	pool.join()
	if args.verbose:
		print('Done generating pileups.')
#