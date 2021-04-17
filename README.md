# PHYS 490 Project #
## A Convolutional Neural Network Approach to Gravitational Wave Data Analysis ##
Group 7: Michael Astwood, Madison Buitenhuis, Jensen Lawrence, Catie Terrey

## Introduction ##

This project follows the paper  "Improved deep learning techniques in gravitational-wave data analysis" by H. Xia et. Al. We will be recreating a convolutional neural network with batch normalization, dropout, and weight decay which classifies signals from spin aligned black hole mergers. In addition to this we will construct a bayesian convolutional neural network and compare the results from each network. 

## Packages Used ##

Please see [requiements](https://github.com/jensen-lawrence/Phys490-Project/blob/main/package_requirements.txt) for a list of packages used.

## How to Use ##

To utilize this project, first clone this repo onto a device of your choice. Please ensure you have all the packages used installed before attempting to run any code (see above for list).

Please note that the folders containing each section are linked accordingly to the subtitles in this section.

### [Data generation](https://github.com/jensen-lawrence/Phys490-Project/tree/main/data_generation) ###

To use our data generation tool...

### [CNN](https://github.com/jensen-lawrence/Phys490-Project/tree/main/cnn_files) ###

To use our CNN tool...
### [BayseianCNN](https://github.com/jensen-lawrence/Phys490-Project/tree/main/bnn_files) ###

To use our Bayesian CNN tool foloow these steps:

1. Follow the steps to geenerate data above and save the generated data.

2. Obtain the path of the generated data. For example purposes we will call this data_path

3. In the terminal run python3 imp_bnn.py --param param\bnn_params.json --data data_path.

4. Adjust the parameters in [bnn_params](https://github.com/jensen-lawrence/Phys490-Project/blob/main/param/bnn_params.json) to your liking.

## Resources / Credits ##

Main paper: https://arxiv.org/pdf/2011.04418.pdf

Data generation reference paper: https://journals.aps.org/prd/pdf/10.1103/PhysRevD.100.063015

BLiTZ - Bayesian Layers in Torch Zoo (a Bayesian Deep Learing library for Torch)}: https://github.com/piEsposito/blitz-bayesian-deep-learning 

Package provided by reference paper: https://github.com/timothygebhard/ggwd

LALSuite documentation: https://lscsoft.docs.ligo.org/lalsuite/

LALSimulation repo: https://git.ligo.org/lscsoft/lalsuite/-/tree/master/lalsimulation
