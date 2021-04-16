# PHYS 490 Project #
## A Convolutional Neural Network Approach to Gravitational Wave Data Analysis ##
Group 7: Michael Astwood, Madison Buitenhuis, Jensen Lawrence, Catie Terrey

## Introduction ##

This project follows the paper  "Improved deep learning techniques in gravitational-wave data analysis" by H. Xia et. Al. We will be recreating a convolutional neural network with batch normalization, dropout, and weight decay which classifies signals from spin allifned black hole mergers. In addition to this we will construct a bayesian convolutional neural network and compare the results from each netwrok. 

## Packages Used ##
Built with: 

* LALSuite

* LALSimulation

* Pytorch

* Blitz



## How to Use ##

### [Data generation](https://github.com/jensen-lawrence/Phys490-Project/tree/main/data_generation) ###

### [CNN](https://github.com/jensen-lawrence/Phys490-Project/blob/main/cnn_files/cnn.py) ###

### [BayseianCNN](https://github.com/jensen-lawrence/Phys490-Project/blob/main/bnn_files/bnn.py) ###

## Resources / Credits ##

Main paper: https://arxiv.org/pdf/2011.04418.pdf

Data generation reference paper: https://journals.aps.org/prd/pdf/10.1103/PhysRevD.100.063015

Package provided by reference paper: https://github.com/timothygebhard/ggwd

LALSuite documentation: https://lscsoft.docs.ligo.org/lalsuite/

LALSimulation repo: https://git.ligo.org/lscsoft/lalsuite/-/tree/master/lalsimulation
