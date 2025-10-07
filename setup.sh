#!/bin/bash

echo "Starting setup script..."
set -e  # Exit on any error

PY_VERSION="3.13"
VENV_NAME=".venv"

# Ensure pyenv is available and initialize it for this shell
if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv not found. Please install pyenv and pyenv-virtualenv first." >&2
  exit 1
fi

# Load pyenv and pyenv-virtualenv into this non-interactive shell
# shellcheck disable=SC1090
eval "$(pyenv init -)"
# shellcheck disable=SC1090
if command -v pyenv-virtualenv-init >/dev/null 2>&1; then
  eval "$(pyenv virtualenv-init -)"
fi

echo "Checking existing python versions"

# Install Python version if missing
if pyenv versions --bare | grep -qx "${PY_VERSION}"; then
  echo "==>Python ${PY_VERSION} already installed in pyenv."
else
  echo "==>Installing Python ${PY_VERSION} with pyenv..."
  pyenv install "${PY_VERSION}"
fi
pyenv local ${PY_VERSION}

echo "==>Creating a '.venv' environment with python ${PY_VERSION}..."
# Create virtualenv if missing
if pyenv versions --bare | grep -qx "${VENV_NAME}"; then
  echo "pyenv virtualenv '${VENV_NAME}' already exists."
else
  echo "==>Creating pyenv virtualenv '${VENV_NAME}' from ${PY_VERSION}..."
  pyenv virtualenv "${PY_VERSION}" "${VENV_NAME}"
fi

# Activate the virtualenv
echo "Activating pyenv virtualenv '${VENV_NAME}'..."
pyenv activate "${VENV_NAME}"

echo "Python: $(python3 --version)"
echo "Pip   : $(pip3 --version)"

echo "==>Upgrading pip/setuptools/wheel..."
python3 -m pip install -q --upgrade pip setuptools wheel

echo "==>Installing torch and torch_geometric..."
pip3 install torch torchvision
pip3 install torch_geometric
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

echo "==>Installing other dependencies from requirements.txt..."
pip3 install -r requirements.txt

echo "==>Installing nltk and downloading propbank"
python3 -c "import nltk; nltk.download('propbank')"

echo "==>Downloading amrlib sentence-to-graph model (stog)..."

AMRLIB_PATH=$(python3 -c "import amrlib, os; print(os.path.dirname(amrlib.__file__))")
DATA_DIR="${AMRLIB_PATH}/data"
TARBALL="${DATA_DIR}/model_stog.tar.gz"
MODEL_URL="https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_large-v0_1_0/model_parse_xfm_bart_large-v0_1_0.tar.gz"

# Final directory name you want after extraction
MODEL_DIR="${DATA_DIR}/model_stog"

mkdir -p "$DATA_DIR"

# Helper: check if directory exists and is non-empty (POSIX-safe)
dir_nonempty=false
if [ -d "$MODEL_DIR" ] && [ -n "$(find "$MODEL_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  dir_nonempty=true
fi

# ---- OR condition: (MODEL_DIR non-empty) OR (tarball exists & non-zero size) ----
if $dir_nonempty; then
  echo "✅ Model already installed at $MODEL_DIR"
  exit 0
fi

# If we get here, the model_dir is missing/empty. If tarball is present, we'll use it; else download it.
if [ -s "$TARBALL" ]; then
  echo "📦 Found existing tarball: $TARBALL (will use it)"
else
  echo "⬇️  Downloading model (resumable)..."
  wget -c -O "$TARBALL" "$MODEL_URL"
fi

echo "🗜️  Inspecting tarball for top-level folder..."
TOPDIR=$(tar -tf "$TARBALL" | head -1 | cut -d/ -f1)

# Extract if that folder isn't already present
if [ -d "${DATA_DIR}/${TOPDIR}" ]; then
  echo "Found existing extracted folder: ${DATA_DIR}/${TOPDIR}"
else
  echo "🗜️  Extracting model..."
  tar -xzf "$TARBALL" -C "$DATA_DIR"
fi

# Rename extracted folder to the stable name (no symlink)
if [ "${TOPDIR}" != "model_stog" ]; then
  echo "🔁 Renaming ${TOPDIR} -> model_stog"
  rm -rf "$MODEL_DIR"
  mv "${DATA_DIR}/${TOPDIR}" "$MODEL_DIR"
fi

echo "🧹 Cleaning up tarball..."
rm -f "$TARBALL"

echo "✅ Model installed at $MODEL_DIR"

echo "==>Cloning GraphLanguageModels into baselines/ ..."
cd baselines
git clone https://github.com/Heidelberg-NLP/GraphLanguageModels.git
cd ..
git add submodule baselines/GraphLanguageModels
cd baselines/GraphLanguageModels
git remote set-url origin https://github.com/tenjust/graph-reasoning-project.git

#echo "Installing additional dependencies for GraphLanguageModel..."
# pip3 install -r requirements.txt
