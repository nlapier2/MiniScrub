import argparse, gzip, multiprocessing, random, sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():  # handle user arguments
	parser = argparse.ArgumentParser(description='Create pileups from .paf read-to-read mapping and fastq reads.')
	parser.add_argument('--avgdepth', type=int, default=10, help='Average coverage depth in each pileup image.')
	parser.add_argument('--compression', default='none', choices=['none', 'gzip'], help='Compression format, or none')
	parser.add_argument('--limit_fastq', type=int, default=0, help='Limit number of reads to scan from fastq file.')
	parser.add_argument('--limit_reads', type=int, default=0, help='Limit number of reads to generate pileups for.')
	parser.add_argument('--mapping', required=True, help='Path to the .paf file of read-to-read mappings.')
	parser.add_argument('--maxdepth', type=int, default=20, help='Maximum number of matched reads per pileup image.')
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


def make_pileup_rgb(pid, readname, readqual, readlen, matches, avgdepth, maxdepth, plot_opt):
	pileup, total, pixel = [], 0, [100.0, readqual, 100.0]
	seq = [pixel] * readlen
	pileup.append(seq)
	for i in range(maxdepth):
		pileup.append([[0.0,0.0,0.0]])  # fill in placeholder lines

	selections, depth_order, depth_index, num = range(len(matches.index)), range(1, maxdepth+1), 0, 0
	random.shuffle(selections); random.shuffle(depth_order)
	while total < avgdepth * readlen and num < len(selections):
		selection = matches.iloc[selections[num]]
		total += int(selection[3] - selection[2])
		num += 1
	selections = selections[:num]
	matches = matches.iloc[selections]

	for selection in matches.iterrows():
		selection = selection[1]
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
	if plot_opt:
		plt.imsave(readname+'.png', pileup, vmin=0, vmax=255, format='png', dpi=300)
	return 0


def main():
	args = parse_args()
	read_count, line_count, window_size = 0, 0, 200
	reads_df = process_reads(args.reads, args.compression, args.limit_fastq)
	reads_list = list(reads_df)
	pool = multiprocessing.Pool(processes=args.processes)

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
			cur_chunk.rename(columns={13: 'match_pct', 14: 'green', 15: 'blue'}, inplace=True)
			#cur_chunk = cur_chunk.sort_values('match_pct', ascending=False)
			res = pool.apply_async(make_pileup_rgb, (read_count, cur_read, reads_df[cur_read][2], len(reads_df[cur_read][0]), cur_chunk, args.avgdepth, args.maxdepth, args.saveplots,))
			read_count += 1
			if read_count % 100 == 0:
				print 'Finished pileups for ' + str(read_count) + ' lines'
			if read_count == args.limit_reads:
				break
			read_data, line_count = {}, 0

		splits += [splits[9] / splits[10] * 100.0, reads_df[splits[5]].iloc[2], ((splits[3]-splits[2])/(splits[8]-splits[7]))*100.0]
		read_data[line_count] = splits
		cur_read = splits[0]
		line_count += 1

	if read_data != {} and read_count < args.limit_reads:
		cur_chunk = pd.DataFrame.from_dict(read_data, orient='index')
		cur_chunk.rename(columns={13: 'match_pct', 14: 'green', 15: 'blue'}, inplace=True)
		#cur_chunk = cur_chunk.sort_values('match_pct', ascending=False)
		res = pool.apply_async(make_pileup_rgb, (read_count, cur_read, reads_df[cur_read][2], len(reads_df[cur_read][0]), cur_chunk, args.avgdepth, args.maxdepth, args.saveplots,))
		read_count += 1
		if read_count % 100 == 0:
			print 'Finished pileups for ' + str(read_count) + ' lines'	

	f.close()
	pool.close()
	pool.join()


if __name__ == "__main__":
	main()
#