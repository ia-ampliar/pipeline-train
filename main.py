# config.py
import os
import datetime
from models.train import train_model
from models.model import get_model
from utils.evaluate import evaluate_model
from utils.visualize import plot_training


# Configurações
IMG_SIZE = (224, 224)  # Tamanho padrão para MobileNetV2
BATCH_SIZE = 128
NUM_CLASSES = 2  # Número de classes

# Configurações de treinamento
PACIENCE = 10  # Paciencia para early stopping
DELTA = 0.01  # Delta para early stopping
EPOCHS = 100  # Número de épocas para treinamento
LEARNING_RATE = 0.001  # Taxa de aprendizado


# Configurações de caminho
PATH_IMGS = '/home/ampliar/computerVision/train/dataset-224x224-10x-macenko-splited'
LOG_DIR = "logs/fit/mobileNetv2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

WEIGHT_PATH = "weights/"
if not os.path.exists(WEIGHT_PATH):
    os.makedirs(WEIGHT_PATH)


CHECKPOINT_PATH = "weights/checkpoint/"
if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)


MODEL_PATH = "models/"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Configurações de avaliação
EVAL_BATCH_SIZE = 32
EVAL_PATH = "dataset-224x224-10x-macenko_splited/test/"
if not os.path.exists(EVAL_PATH):
    os.makedirs(EVAL_PATH)

# Configurações de teste
TEST_BATCH_SIZE = 32
TEST_PATH = "dataset-224x224-10x-macenko_splited/test/"
if not os.path.exists(TEST_PATH):
    os.makedirs(TEST_PATH)

# Configurações de predição
PRED_BATCH_SIZE = 32
PRED_PATH = "dataset-224x224-10x-macenko_splited/test/"
if not os.path.exists(PRED_PATH):
    os.makedirs(PRED_PATH)

# Configurações de visualização
VISUALIZE_BATCH_SIZE = 32
VISUALIZE_PATH = "dataset-224x224-10x-macenko_splited/test/"
if not os.path.exists(VISUALIZE_PATH):
    os.makedirs(VISUALIZE_PATH)

def main():
    """
    Função principal para treinar o modelo.
    """

    # Treinamento do modelo
    train_model(
        path_images=PATH_IMGS,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        weights_path=WEIGHT_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        model_path=MODEL_PATH,
        log_dir=LOG_DIR,
        epochs=EPOCHS,
        num_classes=NUM_CLASSES,
        pacience=PACIENCE,
        delta=DELTA
    )

    # Avaliação do modelo
    # model, _ = get_model("MobileNetV2", num_classes=NUM_CLASSES, _mode=False)
    # model.load_weights(WEIGHT_PATH + 'mobileNetv2_weights_224x224.h5')
    # evaluate_model(model)

    # Visualização do modelo
    # plot_training(model)

if __name__ == "__main__":
    main()