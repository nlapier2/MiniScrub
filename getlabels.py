import argparse


def parse_args():  # handle user arguments
	parser = argparse.ArgumentParser(description='Create pileups from .paf read-to-read mapping and fastq reads.')
	parser.add_argument('--infile', required=True, help='Path to the sam file with ground truth labels.')
	parser.add_argument('--outfile', default='labels.txt', help='File to output label information to.')
	args = parser.parse_args()
	return args


def main():
	args = parse_args()
	infile, outfile = open(args.infile, 'r'), open(args.outfile, 'w')
	for line in infile:
		if line.startswith('@'):
			continue
		splits = line.split('\t')
		readname = splits[0]
		quality = splits[-1][6:]
		outfile.write(readname + ' ' + quality)

	infile.close(); outfile.close()


if __name__ == '__main__':
	main()
#