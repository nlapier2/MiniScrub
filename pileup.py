import argparse, gzip, multiprocessing, sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():  # handle user arguments
	parser = argparse.ArgumentParser(description='Create pileups from .paf read-to-read mapping and fastq reads.')
	parser.add_argument('--compression', default='none', choices=['none', 'gzip'], help='Compression format, or none')
	parser.add_argument('--limit_fastq', type=int, default=0, help='Limit number of reads to scan from fastq file.')
	parser.add_argument('--limit_matches', type=int, default=0, help='Limit number of best matches kept per read.')
	parser.add_argument('--limit_reads', type=int, default=0, help='Limit number of reads to generate pileups for.')
	parser.add_argument('--mapping', required=True, help='Path to the .paf file of read-to-read mappings.')
	parser.add_argument('--processes', type=int, default=1, help='Number of multiple processes to run concurrently.')
	parser.add_argument('--reads', required=True, help='Path to the .fastq reads file.')
	parser.add_argument('--saveplots', action='store_true', help='If used, will plot the pileups and save them.')
	args = parser.parse_args()
	return args


def process_reads(reads, compression, limit):  # using fastq file, create dict mapping read names to sequence and quality scores
	if compression == 'gzip':
		reads_file = gzip.open(reads, 'r')
	else:
		reads_file = open(reads, 'r')
	reads_df, num, current = {}, -1, ''
	read_count = 0
	for line in reads_file:
		line = line.strip()
		num = (num + 1) % 4
		if num == 0:
			current = line[1:].split(' ')[0]
		elif num == 1:
			reads_df[current] = [line.upper()]
			read_count += 1
		elif num == 3:
			#scores = [int((ord(ch)-33)*2.75) for ch in line]
			scores = [(ord(ch)-33) for ch in line]
			reads_df[current].append(scores)
			reads_df[current].append(np.mean(scores))
			if limit > 0 and read_count > limit:
				break
	reads_file.close()
	reads_df = pd.DataFrame(reads_df)
	return reads_df


def make_pileup_rgb(pid, readqual, readlen, matches, limit, plot_opt):
	return str(pid) + ' done'
	pileup, pixel = [], [100.0, readqual, 100.0]
	seq = [pixel for i in range(min(end,readlen)-start)]
	maxlen, total, avgdepth, cutoff = len(seq), 0, 20, 10
	pileup.append(seq)

	# select the reads that cover at least half of the window
	matches = matches[((matches[2]-start).clip(0,None) + (end-matches[3]).clip(0,None))  < (float(end-start)/2.0)]
	matches = matches.head(limit)

	for match in matches.iterrows():
		match = match[1]
		prefix = [[0.0,0.0,0.0] for i in range(int(match[2])-start)]
		r = match['match_pct']
		#g = reads_df[match[5]][2] #np.mean(reads_df[match[5]][1])
		#b = ((match[3] - match[2]) / (match[8] - match[7])) * 100.0
		g = match['green']
		b = match['blue']
		seq = prefix + [[r,g,b] for i in range(min(end,int(match[3]))-start-len(prefix))]
		pileup.append(seq)
		if len(seq) > maxlen:
			maxlen = len(seq)
		if match[3] < minend:
			minend = match[3]
		if len(pileup) > limit + 1:
			break

	while len(pileup) < limit:
		pileup.append([[0.0,0.0,0.0]])
	for line in range(len(pileup)):
		pileup[line].extend([[0.0,0.0,0.0] for i in range(maxlen - len(pileup[line]))])

	if plot_opt:  # will have to modify to use different windows
		saveplots_rgb(pileup, name)
	#return np.array(pileup), minend
	return 0


def saveplots_rgb(pileups):
	for p in range(len(pileups)):
		plt.imsave(pileups[p][1]+'.png', pileups[p][0], vmin=0, vmax=255, format='png', dpi=300)
	return


def main():
	args = parse_args()
	read_count, line_count, window_size = 0, 0, 200
	reads_df = process_reads(args.reads, args.compression, args.limit_fastq)
	reads_list = list(reads_df)
	pool = multiprocessing.Pool(processes=args.processes)

	#'''
	read_data, cur_read = {}, ''
	if args.compression == 'gzip':
		f = gzip.open(args.mapping, 'r')
	else:
		f = open(args.mapping, 'r')
	for line in f:
		splits = line.strip().split('\t')
		for i in (1,2,3,6,7,8,9,10,11):
			splits[i] = float(splits[i])
		if args.limit_fastq > 0 and (splits[0] not in reads_list or splits[5] not in reads_list):
			continue
		if read_data != {} and cur_read != splits[0]:
			cur_chunk = pd.DataFrame.from_dict(read_data, orient='index')
			cur_chunk['match_pct'] = cur_chunk[9] / cur_chunk[10] * 100.0
			cur_chunk['green'] = reads_df[cur_chunk[5]].iloc[2]
			cur_chunk['blue'] = ((cur_chunk[3] - cur_chunk[2]) / (cur_chunk[8] - cur_chunk[7])) * 100.0
			#cur_chunk = cur_chunk.sort_values('match_pct', ascending=False)
			res = pool.apply_async(make_pileup_rgb, (read_count, reads_df[cur_read][2], len(reads_df[cur_read][0]), cur_chunk, args.limit_matches, args.saveplots))
			#print res
			read_count += 1
			if read_count % 10 == 0:
				print read_count
			if read_count == args.limit_reads:
				break
			read_data, line_count = {}, 0
		#read_data[line_count] = splits
		cur_read = splits[0]
		if splits[5] not in read_data:
			read_data[splits[5]] = splits
		else:
			prev_match = read_data[splits[5]]
			prev_pct = prev_match[9] / prev_match[10]
			if (splits[9] / splits[10]) > prev_pct:
				read_data[splits[5]] = splits
		line_count += 1
	f.close()

	print 'done'
	pool.close()
	pool.join()


if __name__ == "__main__":
	main()
#