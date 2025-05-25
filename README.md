Claro! Aqui está um exemplo de `README.md` com instruções claras para instalar, configurar e ativar o ambiente necessário para treinar os modelos apresentados no seu código:

---

# Sistema de Treinamento e Avaliação de Modelos

Este repositório contém um sistema interativo para treinamento, avaliação e continuação de treinamento de modelos de classificação de imagens. Ele permite utilizar arquiteturas populares como **AlexNet, DenseNet, EfficientNet, Inception, MobileNet, ResNet50, VGG16, ShuffleNet**.  A configuração do ambiente é feita com o arquivo `environment.yaml`.

---

## 📦 Requisitos

- Certifique-se de ter o Python 3.8+ instalado.

- [Anaconda ou Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- GPU compatível com CUDA (versão 12.8)
- Driver NVIDIA atualizado


### Dependências

## ⚙️ Configuração do Ambiente

Clone o repositório:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

Crie o ambiente Conda com suporte a CUDA e TensorFlow:

```bash
conda env create -f environment.yaml
```

Ative o ambiente:

```bash
conda activate train-pipeline-gpu
```

## 📁 Estrutura do Projeto

Certifique-se de manter a seguinte estrutura de diretórios (alguns serão criados automaticamente na primeira execução):

```
.
├── models/
│   ├── model.py              # Define as arquiteturas dos modelos
│   └── train.py              # Contém a lógica de treinamento
├── utils/
│   ├── config.py             # Funções auxiliares (memória da GPU, callbacks)
│   ├── evaluate.py           # Avaliação e métricas
│   └── visualize.py          # Gráficos de treino e métricas
├── main.py                   # Script principal com menu interativo
├── Dataset-splited/         # Base de dados com treino, validação e teste
├── weights/                 # Diretório para salvar pesos
├── models/                  # Diretório para salvar modelos
└── README.md
```

---

## 🔧 Configuração de Caminhos

Os caminhos utilizados no código apontam para um diretório de rede:

```python
BASE_PATH = "/run/user/1000/gvfs/smb-share:server=192.168.1.64,share=datasets/"
```

> ⚠️ Certifique-se de montar corretamente o compartilhamento de rede SMB antes de executar o código. Se preferir usar diretórios locais, altere o valor de `BASE_PATH` no script principal.

---

## ▶️ Como Executar

Para iniciar o sistema interativo, execute o script principal:

```bash
python main.py
```

### Menu Principal

Você verá o seguinte menu:

```
SISTEMA DE TREINAMENTO E AVALIAÇÃO DE MODELOS

MENU PRINCIPAL:
1. Treinar novo modelo
2. Avaliar modelo existente
3. Continuar treinamento do modelo
4. Sair
```

---

## 🧠 Modelos Disponíveis

- `alexnet`
- `densenet`
- `efficientnet`
- `inception`
- `mobilenet`
- `mobilenetv2`
- `resnet50`
- `vgg16`
- `shufflenet`

---

## 📊 Métricas e Visualizações

Durante e após o treinamento, o sistema gera:

- Gráficos de acurácia e perda
- Matriz de confusão
- Curva ROC

Essas ferramentas ajudam na avaliação do desempenho dos modelos treinados.

---

## 🧪 Continuação de Treinamento

Para continuar o treinamento de um modelo, certifique-se de que os checkpoints estejam salvos em:

```bash
weights/checkpoint_all/
```

Você será solicitado a informar o `epoch` de onde deseja continuar o treinamento.

---

## 💬 Suporte

Caso tenha dúvidas ou problemas, sinta-se à vontade para abrir uma *issue* ou entrar em contato com o mantenedor do projeto.

---