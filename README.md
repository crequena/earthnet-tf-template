# TensorFlow template for EarthNet2021 dataset

This git repository contains the python code for the TensorFlow template of [EarthNet2021](https://github.com/earthnet2021/earthnet). 

This work is based on the [Tensorflow implementation of Video Prediction](https://github.com/alexlee-gk/video_prediction) by [Alex Lee et al. (ArXiv)](https://arxiv.org/abs/1804.01523)

## Content of the Repository

This repository contains two entry points for training and testing, namely 'train_earthnet.py' and 'generate_earthnet.py'.

This TensorFlow template used old TF 1.15. Are you developing TF 2.0 work? Please contribute your code to the [EarthNet2021 intercomparison toolkit!](https://github.com/earthnet2021/earthnet)

# Getting started
To get started clone the repository or use it as a github template.

# Requirements
We tested the code using Python 3.6, Tensorflow 1.9, CUDA Version 9.0.176 and CUDNN 7.4.2

Make sure the python dependencies in [Requirements](requirements.txt) are installed. Alternatively create a conda environment using conda.yml settings.

#### General workflow 
numpy
matplotlib
scipy
tensorflow-gpu==1.15