import random
f = open('seqs.fa', 'w')
for i in range(49):
	f.write('>seq'+str(i)+'\n')
	lenvar = int(random.random()*10000)
	for j in range(lenvar):
		r = random.random()
		if r < 0.75:
			f.write('A')
		elif r < 0.85:
			f.write('C')
		elif r < 0.95:
			f.write('G')
		else:
			f.write('T')
	f.write('\n')
f.close()
