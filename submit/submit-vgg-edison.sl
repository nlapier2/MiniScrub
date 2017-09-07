#!/bin/bash 
#SBATCH -N 1  # use 1 node
#SBATCH -t 36:00:00  # set 36 hour time limit
#SBATCH -p regular  # submit to the regular 'partition'

echo -e "\n time python vgg.py --epochs 3 --extra 0 --input sample/combined/x0124-x0125/w5-k15-h24/rgb-minimizers/ --labels data/combined/x0124-x0125/x0124-x0125-labels-minimizers-48.txt --segment_size 48 --window_size 72 --baseline --output models/combined-x0124-x0125-w5-k15-h24-seg48-win72-rgb-minimizers-n50000i-e3-regr.hd5 --outputsvm models/svm-combined-x0124-x0125-w5-k15-h24-seg48-win72-rgb-minimizers-n50000i-regr.pkl \n"

module unload python
module load python/2.7-anaconda-4.4

srun --ntasks=1 --cpus-per-task=16 time python vgg.py --epochs 3 --extra 0 --input sample/combined/x0124-x0125/w5-k15-h24/rgb-minimizers/ --labels data/combined/x0124-x0125/x0124-x0125-labels-minimizers-48.txt --segment_size 48 --window_size 72 --baseline --output models/combined-x0124-x0125-w5-k15-h24-seg48-win72-rgb-minimizers-n50000i-e3-regr.hd5 --outputsvm models/svm-combined-x0124-x0125-w5-k15-h24-seg48-win72-rgb-minimizers-n50000i-regr.pkl 
#
#echo -e "\n time python vgg.py --epochs 5 --extra 0 --input sample/x0124-EcoliSkoreensis/w5-k15-h24/rgb-minimizers/ --labels data/x0124-EcoliSkoreensis/x0124-labels-minimizers-w5-k15-24.txt --segment_size 24 --window_size 48 --baseline --output models/x0124-w5-k15-h24-seg24-win48-rgb-minimizers-n25000i-e5-regr.hd5 --outputsvm models/svm-x0124-w5-k15-h24-seg24-win48-rgb-minimizers-n25000i-regr.pkl \n"
#
#srun --ntasks=1 --cpus-per-task=16 time python vgg.py --epochs 5 --extra 0 --input sample/x0124-EcoliSkoreensis/w5-k15-h24/rgb-minimizers/ --labels data/x0124-EcoliSkoreensis/x0124-labels-minimizers-w5-k15-24.txt --segment_size 24 --window_size 48 --baseline --output models/x0124-w5-k15-h24-seg24-win48-rgb-minimizers-n25000i-e5-regr.hd5 --outputsvm models/svm-x0124-w5-k15-h24-seg24-win48-rgb-minimizers-n25000i-regr.pkl
#
#echo -e "\n time python vgg.py --epochs 3 --extra 0 --input sample/combined/x0124-x0125/w5-k15-h24/rgb-minimizers/ --labels data/combined/x0124-x0125/x0124-x0125-labels-minimizers-48.txt --segment_size 48 --window_size 72 --baseline --output models/combined-x0124-x0125-w5-k15-h24-seg48-win72-rgb-minimizers-n50000i-e3-regr.hd5 --outputsvm models/svm-combined-x0124-x0125-w5-k15-h24-seg48-win72-rgb-minimizers-n50000i-regr.pkl \n"
#
#srun --ntasks=1 --cpus-per-task=16 time python vgg.py --epochs 3 --extra 0 --input sample/combined/x0124-x0125/w5-k15-h24/rgb-minimizers/ --labels data/combined/x0124-x0125/x0124-x0125-labels-minimizers-48.txt --segment_size 48 --window_size 72 --baseline --output models/combined-x0124-x0125-w5-k15-h24-seg48-win72-rgb-minimizers-n50000i-e3-regr.hd5 --outputsvm models/svm-combined-x0124-x0125-w5-k15-h24-seg48-win72-rgb-minimizers-n50000i-regr.pkl
#
