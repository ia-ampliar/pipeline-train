#!/bin/bash

# Nome do ambiente Conda
ENV_NAME="tf-gpu-env"

# Criar e ativar o ambiente Conda com Python 3.9 (compatível com TensorFlow)
conda create -y -n $ENV_NAME python=3.9
source activate $ENV_NAME  # Para sistemas Unix/Linux
# conda activate $ENV_NAME  # Para Windows (caso necessário)

# Instalar CUDA e CuDNN compatíveis com TensorFlow
conda install -y -c conda-forge cudatoolkit=11.8 cudnn=8.9

# Configurar variáveis de ambiente
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

# Instalar bibliotecas via pip
pip install --upgrade pip
pip install -r requirements.txt

# Verificar instalação
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

echo "Configuração concluída com sucesso!"
