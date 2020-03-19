# gammanet-pytorch

General-purpose PyTorch implementation of gamma-net. 

Files
- gammanet.py: main gamma-net implementation
- fgru.py: implementation of fGRU
- dataset.py: simple Dataset class for PyTorch
- dataset_maker.py: conversion scripts to process different datasets into a format readable by the SimpleDataset class
- transforms.py: image preprocessing transforms
- criteria.py: scoring functions
- experiments.ipynb: main gamma-net experiments notebook
- experiments-unet.ipynb: main u-net expetiments notebook
- experiments-comp.ipynb: notebook for comparing model artifacts and test performances
- exp-gn-template.ipynb: batch training template notebook for gamma-net
- datasets.ipynb: data processing notebook

Usage
- Use dataset_maker.py as a template for converting datasets for use with the SimpleDataset class. 
- Use the experiment notebooks to train and test gamma-net and u-net models. Some additional exploratory and diagnostic tests are included. 

---
Original paper: 

Linsley, Drew, Junkyung Kim, and Thomas Serre. "Sample-efficient image segmentation through recurrence." arXiv preprint arXiv:1811.11356 (2018).
