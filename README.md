# MiniScrub: *de novo* long read scrubbing using approximate alignment and deep learning

Current version: 0.2 | September 8, 2017

Note: this README is just a short introduction to MiniScrub. For documentation, please see: MiniScrub-Documentation.pdf

MiniScrub is a tool for *de novo* scrubbing, or trimming, low-quality parts of long reads. The rationale is that long reads are more informative than short reads, but most current technologies have high error rates, with low-quality segments within many reads.

MiniScrub uses minimap (https://github.com/lh3/minimap) for efficient *de novo* all-to-all read mapping, creates pileup images for each read showing the reads mapping to it, and uses deep learning to detect which portions of each read are low-quality. More specifically, we use a modified version of the Keras implementation (https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py) of VGG (https://arxiv.org/abs/1409.1556), a popular and major contest-winning Convolutional Neural Network architecture.

Scripts:
* cigarToPctId.py extracts percent identity labels for read segments from a sam file mapping those reads to a reference with graphmap
* miniscrub.py performs actual read scrubbing on fastq format reads based on a loaded Keras model
* pileup.py creates pileup images for the CNN given input fastq reads and minimap paf file
* vgg.py trains and tests the VGG implementation on the pileup images, and can compare results with an SVM baseline


### License Usage Information

The scripts in MiniScrub make use of a modified version of the Keras implementation of VGG.
* Keras is available under the MIT license (https://opensource.org/licenses/MIT).
* VGG is a CNN created by the Oxford Visual Geometry Group whose architecture and weights were released under the Creative Commons License (https://creativecommons.org/licenses/by/4.0/).

We use TensorFlow as our Keras backend, which is under the Apache License (https://www.apache.org/licenses/LICENSE-2.0). We use a modified version of the Minimap code to output the lists of colinear minimizers; Minimap is available under the MIT license as well.
