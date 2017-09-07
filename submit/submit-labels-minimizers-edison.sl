#!/bin/bash 
#SBATCH -N 1  # use 1 node
#SBATCH -t 8:00:00  # set 24 hour time limit
#SBATCH -p regular  # submit to the regular 'partition'

module unload python
module load python

srun --ntasks=1 --cpus-per-task=4 time python cigarToPctId.py --amount 24 --compression gzip --mode minimizers --outfile x0124-labels-minimizers-w5-k15-24.txt --paf data/x0124-EcoliSkoreensis/x0124-mappings-w5-k15.paf.gz --sam data/x0124-EcoliSkoreensis/x0124-graphmapped-reads.sam.gz --limit_length 80000 

#srun --ntasks=1 --cpus-per-task=4 time python cigarToPctId.py --amount 12 --compression gzip --mode minimizers --outfile x0125-labels-minimizers-w5-k15-12.txt --paf data/x0125-Mock26/x0125-mappings-w5-k15.paf.gz --sam data/x0125-Mock26/x0125-graphmapped-reads.sam.gz --limit_length 80000
#
