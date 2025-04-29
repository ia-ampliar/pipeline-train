#!/bin/bash

# Verifica se o arquivo YAML existe
if [ ! -f "$1" ]; then
    echo "Erro: Arquivo YAML '$1' não encontrado."
    exit 1
fi

# Nome do ambiente (extraído do nome do arquivo YAML, sem a extensão)
ENV_NAME=$(basename "$1" .yml)

# Cria o ambiente Conda a partir do YAML
echo "Criando ambiente Conda '$ENV_NAME' a partir de '$1'..."
conda env create -f "$1" -n "$ENV_NAME"

# Verifica se o ambiente foi criado com sucesso
if [ $? -ne 0 ]; then
    echo "Falha ao criar o ambiente '$ENV_NAME'."
    exit 1
fi

# Ativa o ambiente e instala o ipykernel
echo "Instalando ipykernel no ambiente '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"  # Inicializa o conda
conda activate "$ENV_NAME"
conda install -y ipykernel
python -m ipykernel install --user --name="$ENV_NAME" --display-name="Python ($ENV_NAME)"

echo "Ambiente '$ENV_NAME' criado e kernel registrado no Jupyter com sucesso!"
echo "Para usar no Jupyter, selecione o kernel: 'Python ($ENV_NAME)'."