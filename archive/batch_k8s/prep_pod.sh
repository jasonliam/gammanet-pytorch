#!/bin/bash

# definitions
export CODE_ROOT="/ceph/gammanet"
export NB_SUBDIR="batch"
export DATA_SUBDIR="data"
export RESULTS_SUBDIR="results"
export LOG_SUBDIR="logs"
export DATA_FILE="sunnybrook_180.tar.bz2"

# init code repo
cd /root
git clone https://github.com/jasonliam/gammanet-pytorch
cd /root/gammanet-pytorch
git submodule update --init
mv pytorch-unet pytorch_unet

# check code root
if [ ! -d "${CODE_ROOT}" ]; then
  echo "Code root directory does not exist" >&2
  exit 1
fi

# check data file 
if [ ! -f "${CODE_ROOT}/${DATA_SUBDIR}/${DATA_FILE}" ]; then
  echo "Data file does not exist" >&2
  exit 1
fi

# copy and expand training data
cp "${CODE_ROOT}/${DATA_SUBDIR}/${DATA_FILE}" .
apt-get update && apt-get install pbzip2
tar -Ipbzip2 -xf "${DATA_FILE}"

