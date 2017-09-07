#!/bin/bash 
#SBATCH -N 2  # use 2 nodes
#SBATCH -t 00:30:00  # set 30 minute time limit
#SBATCH --mem 16M  # use 16 MB of memory
#SBATCH -p regular  # submit to the regular 'partition'
#SBATCH -L SCRATCH  # job requires $SCRATCH file system
#SBATCH -C haswell  # use haswell nodes
#SBATCH --qos jgi  # jgi 'quality of service' -- which is good
#SBATCH -A fungalp  # fungalp activity -- faster submission

srun --ntasks=2 --cpus-per-task=2 --qos jgi -A fungalp python test/test.py
#
