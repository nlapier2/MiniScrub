#!/bin/bash 
#SBATCH -N 1  # use 1 node
#SBATCH -t 8:00:00  # set 48 hour time limit
#SBATCH -p regular  # submit to the regular 'partition'

module unload python
module load python/3.6-anaconda-4.4 

srun --ntasks=1 --cpus-per-task=17 time python pileup.py --color rgb --compression gzip -k 15 --limit_reads 25000 --mapping data/x0125-Mock26/x0125-mappings-w5-k15.paf.gz --mode minimizers --plotdir sample/x0125-Mock26/w5-k15-h24/rgb-minimizers/ --processes 16 --reads data/x0125-Mock26/x0125-reads.fastq.gz --saveplots --verbose --maxdepth 24

#time python pileup.py --color rgb --compression gzip -k 13 --limit_reads 5000 --mapping data/x0124-EcoliSkoreensis/x0124-mappings-w3-k13.paf.gz --mode whole --plotdir sample/x0124-EcoliSkoreensis/w3-k13/rgb-minimizers/ --processes 16 --reads data/x0124-EcoliSkoreensis/x0124-reads.fastq.gz --saveplots --verbose

#time python pileup.py --color rgb --compression gzip -k 11 --limit_reads 5000 --mapping data/x0125-Mock26/x0125-mappings-w1-k11.paf.gz --mode whole --plotdir sample/x0125-Mock26/w1-k11/rgb-minimizers/ --processes 8 --reads data/x0125-Mock26/x0125-reads.fastq.gz --saveplots --verbose 
#
