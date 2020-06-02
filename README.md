# GammaNet Code for Cardiac Images Segmentation

## Usage

* The whole architecture is config driven. Define the configuration for your experiment in `./config` directory. See `default.json` to see the structure and available options.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp`
* This will start the experiment. It will also take care of building the dataset, if it doesn't exist already in mentioned path.
* The logs, stats, plots and saved models would be stored in `../experiment_data/my_exp` dir.
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training.

## Config Options

TBA

## Supported Datasets
* ACDC

## Files
- main.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- gammanet.py: main gamma-net implementation
- fgru.py: implementation of fGRU
- dataset.py: simple Dataset class for PyTorch
- dataset_factory: Factory to build datasets based on config
- dataset_maker.py: conversion scripts to process different datasets into a format readable by the SimpleDataset class
- transforms.py: image preprocessing transforms
- transforms_factory.py: Factory to build image transforms based on config
- criteria.py: scoring functions
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 

---
## Original paper: 

Linsley, Drew, Junkyung Kim, and Thomas Serre. "Sample-efficient image segmentation through recurrence." arXiv preprint arXiv:1811.11356 (2018).
