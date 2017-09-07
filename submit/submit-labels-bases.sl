#!/bin/bash 
#SBATCH -N 1  # use 1 node
#SBATCH -t 8:00:00  # set 24 hour time limit
#SBATCH --mem 8G  # use 32 gb of memory
#SBATCH -p regular  # submit to the regular 'partition'
#SBATCH -L SCRATCH  # job requires $SCRATCH file system
#SBATCH -C haswell  # use haswell nodes
#SBATCH --qos jgi  # jgi 'quality of service' -- which is good
#SBATCH -A fungalp  # fungalp activity -- faster submission

module unload python
module load python

srun --ntasks=1 --cpus-per-task=4 time python cigarToPctId.py --amount 12 --compression gzip --mode bases --outfile x0125-labels-bases-12.txt --paf data/x0125-Mock26/x0125-mappings-w3-k13.paf.gz --sam data/x0125-Mock26/x0125-graphmapped-reads.sam.gz 

#
