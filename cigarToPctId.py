import argparse, gzip, re, sys, time


start = time.time()


def echo(msg):
	global start
	seconds = time.time() - start
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	hms = "%02d:%02d:%02d" % (h, m, s)
	print '['+hms+'] ' + msg


def parse_args():  # handle user arguments
	parser = argparse.ArgumentParser(description='Use cigar strings in sam files to compute percent identities.')
	parser.add_argument('--amount', default=48, type=int, help='Number of bases/minimizers per label.')
	parser.add_argument('--compression', default='none', choices=['none', 'gzip'], help='Compression format, or none')
	parser.add_argument('--limit_length', default=0, type=int, help='Optionally do not label reads above a certain length.')
	parser.add_argument('--limit_paf', default=0, type=int, help='Optionally limit the number of reads from paf file.')
	parser.add_argument('--limit_sam', default=0, type=int, help='Optionally limit the number of reads from sam file.')
	parser.add_argument('--mode', default='minimizers', choices=['minimizers', 'bases'], help='Labels for minimizers or bases.')
	parser.add_argument('--outfile', default='labels.txt', help='File to output label information to.')
	parser.add_argument('--paf', default=None, help='Path to paf file with minimizers.')
	parser.add_argument('--sam', required=True, help='Path to the sam file with ground truth labels.')
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
			if limit_length > 0 and int(splits[1]) > limit_length:
				continue
			if splits[-1][5] == 'I':
				minimizers[splits[0]] = [int(i) for i in splits[-1][6:].split(',')]
			else:
				minimizers[splits[0]] = [int(i) for i in splits[-1][5:].split(',')]

			linecount += 1
			if linecount % 10000 == 0:
				echo('Done reading ' + str(linecount) + ' lines from paf')
			if limit_paf > 0 and linecount % limit_paf == 0:
				paf.close()
				return minimizers

	paf.close()
	return minimizers


def parse_cigar(cigar, mode, flag, minimizers, amount):
	if int(flag) >= 4 and bin(int(flag))[-3] == '1':  # unmapped
		return []
	reverse = int(flag) >= 16 and bin(int(flag))[-5] == '1'
	labels, start = [], 0
	if mode == 'minimizers' and amount < len(minimizers):
		window = minimizers[amount+start] - minimizers[start]
	elif mode == 'minimizers':
		window = len(cigar) - minimizers[0]
	else:
		window = amount

	count, place, num = 0, 0, 0
	matches, mismatches, insertions, deletions, soft = 0.0,0.0,0.0,0.0,0.0
	for ch in cigar:
		if re.match('[0-9]', ch) != None:  # if character is a digit
			num = (num * 10) + int(ch)
		else:
			for i in range(num):
				if ch == 'S':
					soft += 1
				elif ch == '=':
					matches += 1
				elif ch == 'X':
					mismatches += 1
				elif ch == 'I':
					insertions += 1
				elif ch == 'D':
					deletions += 1
				if ch != 'D':
					count += 1

				if count == window:
					labels.append(matches / (matches+mismatches+insertions+deletions+soft))
					matches, mismatches, insertions, deletions, soft = 0.0,0.0,0.0,0.0,0.0
					count = 0
					if mode == 'minimizers':
						start += amount
						if amount + start < len(minimizers):
							window = minimizers[amount+start] - minimizers[start]
						elif start < len(minimizers):
							window = len(cigar) - minimizers[start]
						else:
							break
			num = 0	

	if count > window / 3:  # heuristic
		labels.append(matches / (matches+mismatches+insertions+deletions+soft))
	if reverse:
		labels.reverse()
	return labels


def main():
	args = parse_args()
	if args.mode == 'minimizers' and args.paf == None:
		print 'If in minimizers mode, must specify paf file location.'
		sys.exit()

	minimizers = {}
	if args.mode == 'minimizers':
		echo('Reading paf file...')
		minimizers = read_paf(args.paf, args.compression, args.limit_paf, args.limit_length)

	if args.compression == 'none':
		infile, outfile = open(args.sam, 'r'), open(args.outfile, 'w')
	else:
		infile, outfile = gzip.open(args.sam, 'r'), open(args.outfile, 'w')

	linecount = 0
	res = []
	echo('Reading sam file...')
	for line in infile:
		if line.startswith('@'):
			continue
		splits = line.split('\t')
		if args.mode == 'minimizers' and splits[0] in minimizers:
			res = parse_cigar(splits[5], args.mode, splits[1], minimizers[splits[0]], args.amount)
		elif args.mode == 'bases':
			res = parse_cigar(splits[5], args.mode, splits[1], {}, args.amount)
		if res != []:
			outfile.write(splits[0] + ' ' + ','.join([str(i) for i in res]) + '\n')
			res = []

		linecount += 1
		if linecount % 10000 == 0:
			echo('Done processing ' + str(linecount) + ' lines from sam')
		if args.limit_sam > 0 and linecount % args.limit_sam == 0:
			break

	infile.close(); outfile.close()
	print''; echo('Done')


if __name__ == '__main__':
	main()
#