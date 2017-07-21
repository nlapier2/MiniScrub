import argparse, gzip, sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


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
	reads_dict, num, current = {}, -1, ''
	count = 0
	for line in reads_file:
		line = line.strip()
		num = (num + 1) % 4
		if num == 0:
			current = line[1:].split(' ')[0]
		elif num == 1:
			reads_dict[current] = [line.upper()]
			count += 1
		elif num == 3:
			#scores = [int((ord(ch)-33)*2.75) for ch in line]
			scores = [(ord(ch)-33) for ch in line]
			reads_dict[current].append(scores)
			reads_dict[current].append(np.mean(scores))
			if limit > 0 and count > limit:
				break
	reads_file.close()
	return reads_dict


def select_top(matches):
	return matches


def make_pileup_bw(readname, matches, reads_dict, start, end, limit):
	encodings = {'A': 250.0, 'C': 200.0, 'G': 150.0, 'T': 100.0}
	pileup = []
	#read, seq = reads_dict[readname], []
	read, seq = reads_dict[readname][0], []
	readlen, maxlen = len(read), len(read)
	seq = [encodings[ch] for ch in read]
	pileup.append(seq)

	for match in matches:
		seq = []
		#segment = reads_dict[match[0]][match[4]:match[5]+1]
		segment = reads_dict[match[0]][0][match[4]:match[5]+1]
		start, encoded = [0.0 for i in range(match[2])], [encodings[ch] for ch in segment]
		seq = start + encoded + [0.0 for i in range(len(start) + len(encoded), readlen)]
		if len(seq) > maxlen:
			maxlen = len(seq)
		pileup.append(seq)

	for line in range(len(pileup)):
		pileup[line].extend([0 for i in range(maxlen - len(pileup[line]))])

	return np.array(pileup)


def make_pileup_rgb(readname, matches, reads_dict, start, end, limit):
	pileup, pixel = [], [100.0, np.mean(reads_dict[readname][1]), 100.0]
	#seq = [pixel for i in range(len(reads_dict[readname][0]))]
	seq = [pixel for i in range(min(end,len(reads_dict[readname][0]))-start)]
	maxlen = len(seq)
	pileup.append(seq)
	#pileup.append([pixel for i in range(200)])

	count = 0
	for match in matches:
		if max(0, match[2] - start) + max(0, end - match[3]) > ((end - start) / 2):
			continue
		prefix = [[0.0, 0.0, 0.0] for i in range(match[2])]
		r = match[1] * 100.0
		g = reads_dict[match[0]][2] #np.mean(reads_dict[match[0]][1])
		b = ((float(match[3]) - float(match[2])) / (float(match[5]) - float(match[4])) * 100.0)
		seq = prefix + [[r,g,b] for i in range(min(end,int(match[3]))-start-len(prefix))]
		pileup.append(seq)
		if len(seq) > maxlen:
			maxlen = len(seq)
		count += 1
		if count == limit:
			break

	while len(pileup) < limit:
		pileup.append([[0.0,0.0,0.0]])
	for line in range(len(pileup)):
		pileup[line].extend([[0.0,0.0,0.0] for i in range(maxlen - len(pileup[line]))])

	return np.array(pileup)


def saveplots_bw(reads_list, pileups):
	for p in range(len(pileups)):
		plt.imsave(reads_list[p]+'.png', pileups[p], cmap='gray', vmin=0, vmax=255)
	return


def saveplots_rgb(pileups):
	for p in range(len(pileups)):
		plt.imsave(pileups[p][1]+'.png', pileups[p][0], vmin=0, vmax=255, format='png', dpi=300)
	return


def main():
	count, window_size = 0, 200
	args = parse_args()
	mapfile = gzip.open(args.mapping, 'r')
	reads_dict = process_reads(args.reads, args.limit_fastq)

	prev_read, reads_list, matches, pileups = '', [], [], []
	for line in mapfile:
		splits = line.split('\t')
		cur_read = splits[0]

		if cur_read != prev_read:
			count += 1
			if args.limit_reads > 0 and count > args.limit_reads:
				break

			if prev_read == '':
				prev_read = cur_read
			else:
				matches.sort(key=lambda x: x[1], reverse=True)
				#if args.limit_matches > 0:
				#	matches = matches[:args.limit_matches]
				#matches = select_top(matches)
				#pileups.append(make_pileup_bw(prev_read, matches, reads_dict))
				start = window_size
				while start + (window_size*2) < len(reads_dict[prev_read][0]):
					pileups.append([make_pileup_rgb(prev_read, matches, reads_dict, start, start+window_size, args.limit_matches), prev_read+'_'+str(start)])
					start += window_size
				#reads_list.append(prev_read)
				prev_read = cur_read
				matches = []

		# store target name, match %, and start & end positions for query and target
		if splits[5] not in reads_dict:
			continue
		matches.append([splits[5], float(splits[9])/float(splits[10]), int(splits[2]), int(splits[3]), int(splits[7]), int(splits[8])])

	# Now process the last read
	# matches = select_top(matches)
	start = window_size
	while start + (window_size*2) < len(reads_dict[cur_read][0]):
		pileups.append([make_pileup_rgb(cur_read, matches, reads_dict, start, start+window_size, args.limit_matches), cur_read+'_'+str(start)])
		start += window_size
	#reads_list.append(cur_read)

	mapfile.close()
	if args.saveplots:
		#saveplots_bw(reads_list, pileups)
		saveplots_rgb(pileups)


if __name__ == "__main__":
	main()
#