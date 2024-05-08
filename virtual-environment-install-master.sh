#!/bin/bash

# # For the Bouklas lab server
# VENV_PATH="/home/jpm445/research/projects/cuFJC-scission-lake-thomas-fracture-toughness"

# For the Macbook Air
VENV_PATH="/Users/jasonmulderrig/research/projects/cuFJC-scission-lake-thomas-fracture-toughness"

# Set up Python virtual environment and associated Python packages

if [ ! -d ${VENV_PATH} ]
then
  mkdir -p ${VENV_PATH}
  python3 -m venv ${VENV_PATH}
  cd ${VENV_PATH}
else
  cd ${VENV_PATH}
  if [ ! -f pyvenv.cfg ]
  then
    python3 -m venv ${VENV_PATH}
  else
    rm -rf bin include lib share && rm lib64 && rm pyvenv.cfg
    python3 -m venv ${VENV_PATH}
  fi
fi

source bin/activate

pip3 install wheel && pip3 install --upgrade setuptools && pip3 install --upgrade pip
pip3 install numpy scipy matplotlib
pip3 install cufjc-scission==1.6.0

deactivate
