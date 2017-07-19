import argparse, gzip, sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():  # handle user arguments
	parser = argparse.ArgumentParser(description='Create pileups from .paf read-to-read mapping and fastq reads.')
	parser.add_argument('--limit_fastq', type=int, default=0, help='Limit number of reads to scan from fastq file.')
	parser.add_argument('--limit_matches', type=int, default=0, help='Limit number of best matches kept per read.')
	parser.add_argument('--limit_reads', type=int, default=0, help='Limit number of reads to generate pileups for.')
	parser.add_argument('--mapping', required=True, help='Path to the .paf file of read-to-read mappings.')
	parser.add_argument('--reads', required=True, help='Path to the .fastq reads file.')
	parser.add_argument('--saveplots', action='store_true', help='If used, will plot the pileups and save them.')
	args = parser.parse_args()
	return args


def process_reads(reads, limit):  # using fastq file, create dict mapping read names to sequence and quality scores
	reads_file = gzip.open(reads, "r")
	reads_df, num, current = {}, -1, ''
	count = 0
	for line in reads_file:
		line = line.strip()
		num = (num + 1) % 4
		if num == 0:
			current = line[1:].split(' ')[0]
		elif num == 1:
			reads_df[current] = [line.upper()]
			count += 1
		elif num == 3:
			#scores = [int((ord(ch)-33)*2.75) for ch in line]
			scores = [(ord(ch)-33) for ch in line]
			reads_df[current].append(scores)
			reads_df[current].append(np.mean(scores))
			if limit > 0 and count > limit:
				break
	reads_file.close()
	reads_df = pd.DataFrame(reads_df)
	return reads_df


def make_pileup_rgb(readname, matches, reads_df, start, end, limit):
	pileup, pixel = [], [100.0, np.mean(reads_df[readname][1]), 100.0]
	seq = [pixel for i in range(min(end,len(reads_df[readname][0]))-start)]
	maxlen, querylen = len(seq), len(seq)
	pileup.append(seq)

	# select the reads that cover at least half of the window
	matches = matches[((matches[2]-start).clip(0,None) + (end-matches[3]).clip(0,None))  < (float(end-start)/2.0)]
	matches = matches.head(limit)

	for match in matches.iterrows():
		match = match[1]
		prefix = [[0.0,0.0,0.0] for i in range(int(match[2])-start)]
		r = match['match_pct'] * 100.0
		g = np.mean(reads_df[match[5]][1])
		b = ((match[3] - match[2]) / (match[8] - match[7])) * 100.0
		seq = prefix + [[r,g,b] for i in range(min(end,int(match[3]))-start-len(prefix))]
		pileup.append(seq)
		if len(seq) > maxlen:
			maxlen = len(seq)
		if len(pileup) > limit + 1:
			break

	while len(pileup) < limit:
		pileup.append([[0.0,0.0,0.0]])
	for line in range(len(pileup)):
		pileup[line].extend([[0.0,0.0,0.0] for i in range(maxlen - len(pileup[line]))])

	return np.array(pileup)


def saveplots_rgb(pileups):
	for p in range(len(pileups)):
		plt.imsave(pileups[p][1]+'.png', pileups[p][0], vmin=0, vmax=255, format='png', dpi=300)
	return


def main():
	count, window_size = 0, 200
	args = parse_args()
	reads_df = process_reads(args.reads, args.limit_fastq)
	reads_list = list(reads_df)

	cur_chunk, chunksize, pileups = pd.DataFrame({}), 1000, []
	for chunk in pd.read_table(args.mapping, sep='\t', compression='gzip', chunksize=chunksize, header=None):
		if args.limit_fastq > 0:
			chunk = chunk.loc[chunk[5].isin(reads_list)]
			if chunk.empty:
				continue

		cur_chunk = pd.concat([cur_chunk, chunk])
		cur_read = cur_chunk.iloc[0][0]
		while cur_chunk.iloc[0][0] != cur_chunk.iloc[-1][0]:
			read_data = cur_chunk[cur_chunk[0] == cur_read]  # split off the data for one read
			cur_chunk = cur_chunk[cur_chunk[0] != cur_read]
			read_data['match_pct'] = read_data[9] / read_data[10]
			read_data = read_data.sort_values('match_pct', ascending=False)
			start = window_size
			while start + (window_size*2) < len(reads_df[cur_read][0]):  # make pileup images for different parts of read
				pileups.append([make_pileup_rgb(cur_read, read_data, reads_df, start, start+window_size, args.limit_matches), cur_read+'_'+str(start)])
				start += window_size
			count += 1
			if count == args.limit_reads:
				break
			cur_read = cur_chunk.iloc[0][0]
		if count == args.limit_reads:
			break
		start = window_size

	# process last read now
	if count != args.limit_reads:
		read_data = cur_chunk
		read_data['match_pct'] = read_data[9] / read_data[10]
		read_data = read_data.sort_values('match_pct', ascending=False)
		start = window_size
		while start + (window_size*2) < len(reads_df[cur_read][0]):
			pileups.append([make_pileup_rgb(cur_read, read_data, reads_df, start, start+window_size, args.limit_matches), cur_read+'_'+str(start)])
			start += window_size

	if args.saveplots:
		saveplots_rgb(pileups)


if __name__ == "__main__":
	main()
#