#!/bin/bash

echo "Starting setup script..."
set -e  # Exit on any error

echo "Creating a '.venv' environment with python 3.9.16..."
pyenv install 3.9.16
pyenv virtualenv 3.9.16 .venv

pyenv activate .venv

echo "Installing torch and torch_geometric..."
pip3 install torch torchvision
pip3 install torch_geometric
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

echo "Installing other dependencies from requirements.txt..."
pip3 install -r requirements.txt

echo "Downloading amrlib sentence-to-graph model (stog)..."
AMRLIB_PATH=$(python -c "import amrlib, os; print(os.path.dirname(amrlib.__file__))")
mkdir -p $AMRLIB_PATH/data
wget -O $AMRLIB_PATH/data https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_large-v0_1_0/model_parse_xfm_bart_large-v0_1_0.tar.gz
tar -xvzf $AMRLIB_PATH/data/model_parse_xfm_bart_large-v0_1_0.tar.gz -C $AMRLIB_PATH/data
rm $AMRLIB_PATH/data/model_parse_xfm_bart_large-v0_1_0.tar.gz
mv $AMRLIB_PATH/data/model_parse_xfm_bart_large-v0_1_0 $AMRLIB_PATH/data/model_stog

if [ -d "$AMRLIB_PATH/data/model_stog" ]; then
    echo "stog model installed successfully."
else
    echo "stog model installation failed." >&2
fi

echo "Installing additional dependencies for GraphLanguageModel..."
pip3 install -r baseline/GraphLanguageModel/requirements.txt
