#!/bin/bash

echo "Starting setup script..."
set -e  # Exit on any error

echo "Creating a '.venv' environment with python 3.12.0..."
pyenv install 3.12.0
pyenv virtualenv 3.12.0 .venv

pyenv activate .venv

echo "Installing torch and torch_geometric..."
pip3 install torch torchvision
pip3 install torch_geometric
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

pip3 install transformers accelerate
