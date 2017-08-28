import argparse, gc, gzip, multiprocessing, random, sys, traceback
import numpy as np
import scipy.misc


def parse_args():  # handle user arguments
	parser = argparse.ArgumentParser(description='Create pileups from .paf read-to-read mapping and fastq reads.')
	parser.add_argument('--avgdepth', type=int, default=1000, help='Average coverage depth (not currently used).')
	parser.add_argument('--color', default='rgb', choices=['bw', 'rgb'], help='Color mode (bw or rgb).')
	parser.add_argument('--compression', default='none', choices=['none', 'gzip'], help='Compression format, or none')
	parser.add_argument('--debug', action='store_true', help='Debug mode.')
	parser.add_argument('-k', type=int, default=13, help='K-mer size of minimizers. Required.')
	parser.add_argument('--limit_fastq', type=int, default=0, help='Limit number of reads to scan from fastq file.')
	parser.add_argument('--limit_reads', type=int, default=0, help='Limit number of reads to generate pileups for.')
	parser.add_argument('--mapping', required=True, help='Path to the .paf file of read-to-read mappings.')
	parser.add_argument('--maxdepth', type=int, default=48, help='Maximum number of matched reads per pileup image.')
	parser.add_argument('--mode', default='whole', choices=['whole', 'minimizers'], help='Whole read or minimizers-only.')
	parser.add_argument('--plotdir', default='./', help='If --saveplots is used, directory path to save plots in.')
	parser.add_argument('--processes', type=int, default=1, help='Number of multiple processes to run concurrently.')
	parser.add_argument('--reads', required=True, help='Path to the .fastq reads file.')
	parser.add_argument('--saveplots', action='store_true', help='If used, will plot the pileups and save them.')
	parser.add_argument('--verbose', action='store_true', help='Verbose output option.')
	args = parser.parse_args()
	return args


def process_reads(reads, compression, limit, verbose):  # using fastq file, map read names to sequence and quality scores
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
				print('Finished scanning ' + str(read_count) + ' reads')
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
	return reads_df


def stretch_factor_whole(startpos, line, all_mins, selection):
	ref_mins, match_mins = selection[12], selection[13]
	start_index = len([i for i in ref_mins if i < startpos]) - 1
	if start_index >= len(ref_mins) or start_index < 0:
		return startpos, 1.0
	ref_start, ref_end = ref_mins[start_index], ref_mins[start_index+1] 
	match_start, match_end = match_mins[start_index], match_mins[start_index+1]
	stretch = (match_end - match_start) / (ref_end - ref_start)
	return ref_end, stretch


def stretch_factor_minimizers(startpos, line, all_mins, selection):
	ref_mins, match_mins = selection[12], selection[13]
	ref_start, ref_end, match_start, match_end = selection[2], selection[3], selection[7], selection[8]

	for endpos in range(startpos+1, len(line)):
		if line[endpos][0] == 255.0:
			allstart, allend = all_mins[startpos-1], all_mins[endpos]
			ind = [ref_mins.index(allstart), ref_mins.index(allend)]
			stretch = float(abs(match_mins[ind[1]]-match_mins[ind[0]])) / float(abs(ref_mins[ind[1]]-ref_mins[ind[0]]))
			return endpos, stretch

	mini_start = match_mins[ref_mins.index(all_mins[startpos-1])]  # if no endpos was found (no more minimizers matched)
	stretch = float(match_end - mini_start) / float(ref_end - all_mins[startpos-1]) 
	return len(all_mins), stretch


def make_pileup_bw_whole(pid, readname, readqual, readlen, matches, args):
	try: 
		k = args.k
		readqual = np.mean(readqual)  # TEMP
		maxdepth, saveplots, plotdir, debug, pileup = args.maxdepth, args.saveplots, args.plotdir, args.debug, []
		minimizers = matches[0][12]
		del matches[0]
		pileup.append([128.0 + readqual] * readlen)  # the reference read
		for i in range(maxdepth):
			pileup.append([0.0])  # fill in placeholder lines

		depth_order, depth_index, num = list(range(1, maxdepth+1)), 0, 0
		for s in matches:
			selection = matches[s]
			prefix = [0.0] * int(selection[2])
			suffix = [0.0] * int(readlen-int(selection[3]))
			readqual = np.mean(selection[14])
			pixels = [readqual] * (readlen - len(prefix) - len(suffix))
			seq = prefix + pixels + suffix

			for i in range(len(minimizers)):
				if minimizers[i] < selection[12][0] or minimizers[i] > selection[12][-1]:  # read does not cover this minimizer
					continue
				if minimizers[i] in selection[12]:  # if that minimizer matched by this read
					if minimizers[i]+k < len(seq):
						seq[minimizers[i]:minimizers[i]+k] = [i+128.0 if i < 128.0 else i for i in seq[minimizers[i]:minimizers[i]+k]]
					else:
						seq[minimizers[i]:len(seq)] = [i+128.0 if i < 128.0 else i for i in seq[minimizers[i]:len(seq)]]
			pileup[depth_order[depth_index]] = seq
			depth_index += 1
			if depth_index >= maxdepth:
				break

		for line in range(len(pileup)):
			pileup[line].extend([0.0] * (readlen - len(pileup[line])))
		pileup = np.array(pileup)
		if saveplots:
			scipy.misc.toimage(pileup, cmin=0.0, cmax=255.0).save(plotdir+readname+'.png')
		return 0
	except:
		print('Error in process ' + str(pid))
		err = sys.exc_info()
		tb = traceback.format_exception(err[0], err[1], err[2])
		print(''.join(tb) + '\n')
		return 1


def make_pileup_bw_minimizers(pid, readname, readqual, readlen, matches, args):
	try: 
		k = args.k
		readqual = np.mean(readqual)
		maxdepth, saveplots, plotdir, pileup = args.maxdepth, args.saveplots, args.plotdir, []
		minimizers = matches[0][12]
		del matches[0]
		pileup.append([128.0 + readqual] * len(minimizers))  # the reference read
		for i in range(maxdepth):
			pileup.append([0.0])  # fill in placeholder lines

		depth_order, depth_index, num = list(range(1, maxdepth+1)), 0, 0
		for s in matches:
			selection = matches[s]
			seq = [readqual] * len(minimizers)
			for i in range(len(minimizers)):
				if minimizers[i] < selection[12][0] or minimizers[i] > selection[12][-1]:  # read does not cover this minimizer
					seq[i] = 0.0
				elif minimizers[i] in selection[12]:  # if that minimizer matched by this read
					seq[i] += 128.0
			pileup[depth_order[depth_index]] = seq
			depth_index += 1
			if depth_index >= maxdepth:
				break

		for line in range(len(pileup)):
			pileup[line].extend([0.0] * (len(minimizers) - len(pileup[line])))
		pileup = np.array(pileup)
		if saveplots:
			scipy.misc.toimage(pileup, cmin=0.0, cmax=255.0).save(plotdir+readname+'.png')
		return 0
	except:
		print('Error in process ' + str(pid))
		err = sys.exc_info()
		tb = traceback.format_exception(err[0], err[1], err[2])
		print(''.join(tb) + '\n')
		return 1


def make_pileup_rgb_whole(pid, readname, readqual, readlen, matches, args):
	try: 
		k = args.k
		maxdepth, saveplots, plotdir, pileup = args.maxdepth, args.saveplots, args.plotdir, []
		minimizers = matches[0][12]
		del matches[0]
		avg_stretch = 128.0
		seq = [[255.0, i*2.0, avg_stretch] for i in readqual]
		pileup.append(seq)
		for i in range(maxdepth):
			pileup.append([[0.0,0.0,0.0]])  # fill in placeholder lines

		depth_order, depth_index, num = list(range(1, maxdepth+1)), 0, 0
		for s in matches:
			selection = matches[s]
			meanqual = np.mean(selection[14])*2.0
			prefix = [[0.0,0.0,0.0]] * int(selection[2])
			suffix = [[0.0,0.0,0.0]] * int(readlen-int(selection[3]))
			pixels = [[70.0, meanqual, avg_stretch] for i in range(int(selection[2]), int(selection[3]))]
			seq = prefix + pixels + suffix

			for i in range(len(minimizers)):
				if minimizers[i] < selection[12][0] or minimizers[i] > selection[12][-1]:  # read does not cover this minimizer
					continue
				if minimizers[i] in selection[12]:  # if that minimizer matched by this read
					match_loc = selection[13][selection[12].index(minimizers[i])]
					if minimizers[i]+k < len(seq):
						readquals = [selection[14][j] * 2.0 if j < len(selection[14]) else 0.0 for j in list(range(match_loc, match_loc+k))]
						seq[minimizers[i]:minimizers[i]+k] = [[255.0, readquals[j-minimizers[i]], seq[j][2]] if seq[j][0] == 70.0 else seq[j] for j in list(range(minimizers[i],minimizers[i]+k))]
					else:
						readquals = [selection[14][j] * 2.0 if j < len(selection[14]) else 0.0 for j in list(range(match_loc, match_loc+k))]
						seq[minimizers[i]:len(seq)] = [[255.0, readquals[j-minimizers[i]], seq[j][2]] if seq[j][0] == 70.0 else seq[j] for j in list(range(minimizers[i],len(seq)))]

			pix = 0
			while pix < len(seq):
				if seq[pix][0] != 255.0 and seq[pix][2] != 0.0:# and (pix+1==len(seq) or seq[pix+1] != 255.0):
					endpos, stretch = stretch_factor_whole(pix, seq, minimizers, selection)
					stretch = min(255.0, avg_stretch * (stretch ** 5))
					seq[pix:endpos] = [[i[0], i[1], stretch] for i in seq[pix:endpos]]
					if endpos <= pix:
						pix += 1
					else:
						pix = endpos
				else:
					pix += 1

			pileup[depth_order[depth_index]] = seq
			depth_index += 1
			if depth_index >= maxdepth:
				break

		for line in range(len(pileup)):
			pileup[line].extend([[0.0,0.0,0.0]] * (readlen - len(pileup[line])))

		pileup = np.array(pileup)
		if saveplots:
			scipy.misc.toimage(pileup, cmin=0.0, cmax=255.0, mode='RGB').save(plotdir+readname+'.png')
		return 0
	except:
		print('Error in process ' + str(pid))
		err = sys.exc_info()
		tb = traceback.format_exception(err[0], err[1], err[2])
		print(''.join(tb) + '\n')
		return 1


def make_pileup_rgb_minimizers(pid, readname, readqual, readlen, matches, args):
	try: 
		k = args.k
		maxdepth, saveplots, plotdir, pileup = args.maxdepth, args.saveplots, args.plotdir, []
		minimizers = matches[0][12]
		del matches[0]
		avg_stretch = 128.0
		seq = [[255.0, np.mean(readqual[i:i+k])*2.0, avg_stretch] if i+k <= len(readqual) else [255.0, np.mean(readqual[i:len(readqual)])*2.0, avg_stretch] for i in minimizers]
		pileup.append(seq)
		for i in range(maxdepth):
			pileup.append([[0.0,0.0,0.0]])  # fill in placeholder lines

		depth_order, depth_index, num = list(range(1, maxdepth+1)), 0, 0
		for s in matches:
			selection = matches[s]
			meanqual = np.mean(selection[14])*2.0
			match_start, match_end = minimizers.index(selection[12][0]), minimizers.index(selection[12][-1])
			seq = [[255.0, meanqual, avg_stretch] if minimizers[i] in selection[12] else [70.0, meanqual, avg_stretch] for i in range(match_start, match_end+1)]
			seq = ([[0.0, 0.0, 0.0]] * match_start) + seq + ([[0.0, 0.0, 0.0]] * (len(minimizers) - match_end - 1))

			for pix in range(len(seq)):
				if seq[pix][0] == 255.0 and seq[pix][2] != 0.0:
					matchind = selection[13][selection[12].index(minimizers[pix])]
					seq[pix][1] = np.mean(selection[14][matchind:matchind+k])*2.0 if matchind+k < len(selection[14]) else np.mean(selection[14][matchind:len(selection[14])])*2.0

			pix = 0
			while pix < len(seq):
				if seq[pix][0] != 255.0 and seq[pix][2] != 0.0:# and (pix+1==len(seq) or seq[pix+1] != 255.0):
					endpos, stretch = stretch_factor_minimizers(pix, seq, minimizers, selection)
					stretch = min(255.0, avg_stretch * (stretch ** 5))
					seq[pix:endpos] = [[i[0], i[1], stretch] for i in seq[pix:endpos]]
					if endpos <= pix:
						pix += 1
					else:
						pix = endpos
				else:
					pix += 1

			pileup[depth_order[depth_index]] = seq
			depth_index += 1
			if depth_index >= maxdepth:
				break

		for line in range(len(pileup)):
			pileup[line].extend([[0.0,0.0,0.0]] * (len(minimizers) - len(pileup[line])))
		pileup = np.array(pileup)
		if saveplots:
			scipy.misc.toimage(pileup, cmin=0.0, cmax=255.0, mode='RGB').save(plotdir+readname+'.png')
		return 0
	except:
		print('Error in process ' + str(pid))
		err = sys.exc_info()
		tb = traceback.format_exception(err[0], err[1], err[2])
		print(''.join(tb) + '\n')
		return 1


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

	read_data, cur_read = {}, ''
	if args.compression == 'gzip':
		f = gzip.open(args.mapping, 'r')
	else:
		f = open(args.mapping, 'r')
	for line in f:
		if args.compression == 'gzip':
			line = line.decode('utf8')
		splits = line.strip().split('\t')
		splits = splits[:12] + splits[13:15]
		for i in (1,2,3,6,7,8,9,10,11):
			splits[i] = float(splits[i])
		if splits[12][5] == 'I':
			splits[12] = splits[12][1:]; splits[13] = splits[13][1:]
		splits[12] = [int(i) for i in splits[12][5:].split(',')]
		splits[13] = [int(i) for i in splits[13][5:].split(',')]
		if splits[0] not in reads_df or splits[5] not in reads_df:
			continue
		splits.append(reads_df[splits[5]][0])
		if args.limit_fastq > 0 and (splits[0] not in reads_list or splits[5] not in reads_list):
			continue
		if read_data != {} and cur_read != splits[0]:
			readqual, readlen = reads_df[cur_read][0], len(reads_df[cur_read][0])
			selections = list(range(1,len(read_data)))
			random.shuffle(selections)
			selections = selections[:args.maxdepth] + [0]
			read_data = {i:read_data[i] for i in selections}

			if args.debug:
				if args.mode == 'whole':
					if args.color == 'bw':
						make_pileup_bw_whole(read_count, cur_read, readqual, readlen, read_data, args)
					else:
						make_pileup_rgb_whole(read_count, cur_read, readqual, readlen, read_data, args)
				else:
					if args.color == 'bw':
						make_pileup_bw_minimizers(read_count, cur_read, readqual, readlen, read_data, args)
					else:
						make_pileup_rgb_minimizers(read_count, cur_read, readqual, readlen, read_data, args)
			else:
				if args.mode == 'whole':
					if args.color == 'bw':
						pool.apply_async(make_pileup_bw_whole, (read_count, cur_read, readqual, readlen, read_data, args,))
					else:
						pool.apply_async(make_pileup_rgb_whole, (read_count, cur_read, readqual, readlen, read_data, args,))
				else:
					if args.color == 'bw':
						pool.apply_async(make_pileup_bw_minimizers, (read_count, cur_read, readqual, readlen, read_data, args,))
					else:
						pool.apply_async(make_pileup_rgb_minimizers, (read_count, cur_read, readqual, readlen, read_data, args,))
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
		readqual, readlen = reads_df[cur_read][0], len(reads_df[cur_read][0])
		if args.debug:
			if args.mode == 'whole':
				if args.color == 'bw':
					make_pileup_bw_whole(read_count, cur_read, readqual, readlen, read_data, args)
				else:
					make_pileup_rgb_whole(read_count, cur_read, readqual, readlen, read_data, args)
			else:
				if args.color == 'bw':
					make_pileup_bw_minimizers(read_count, cur_read, readqual, readlen, read_data, args)
				else:
					make_pileup_rgb_minimizers(read_count, cur_read, readqual, readlen, read_data, args)
		else:
			if args.mode == 'whole':
				if args.color == 'bw':
					pool.apply_async(make_pileup_bw_whole, (read_count, cur_read, readqual, readlen, read_data, args,))
				else:
					pool.apply_async(make_pileup_rgb_whole, (read_count, cur_read, readqual, readlen, read_data, args,))
			else:
				if args.color == 'bw':
					pool.apply_async(make_pileup_bw_minimizers, (read_count, cur_read, readqual, readlen, read_data, args,))
				else:
					pool.apply_async(make_pileup_rgb_minimizers, (read_count, cur_read, readqual, readlen, read_data, args,))
		read_count += 1
		if read_count % 1000 == 0 and args.verbose:
			print('Finished pileups for ' + str(read_count) + ' lines')	

	f.close()
	pool.close()
	pool.join()
	print('Done')
#