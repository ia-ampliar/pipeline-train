import sys
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import datetime
import os

from utils.config import get_gpu_memory
from utils.config import get_callbacks
from models.train import train_model
from utils.visualize import plot_training
from utils.visualize import plot_confusion_matrix
from utils.visualize import plot_roc_curve
from utils.evaluate import evaluate_model
from utils.evaluate import calculate_metrics

from models.model import Models

def clear_screen():
    """Limpa a tela do console"""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_menu():
    """Exibe o menu principal"""
    clear_screen()
    print("="*50)
    print("SISTEMA DE TREINAMENTO E AVALIAÇÃO DE MODELOS")
    print("="*50)
    print("\nMENU PRINCIPAL:")
    print("1. Treinar novo modelo")
    print("2. Avaliar modelo existente")
    print("3. Continuar treinamento do modelo")
    print("4. Sair")
    print("\n" + "="*50)

def get_user_choice():
    """Obtém a escolha do usuário"""
    while True:
        try:
            choice = int(input("\nDigite sua opção (1-4): "))
            if 1 <= choice <= 4:
                return choice
            else:
                print("Opção inválida. Por favor, digite um número entre 1 e 4.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número.")

def train_new_model():
    """Executa o treinamento de um novo modelo"""
    print("\nIniciando treinamento de novo modelo...")
    
    # Verifica se há GPU disponível
    get_gpu_memory()

    # Configurações
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 128
    EPOCHS = 100
    NUM_CLASSES = 2
    PACIENTE = 7
    DELTA = 0.001

    BASE_PATH = "/mnt/efs-tcga/HEAL_Workspace/macenko_datas/"
    PATH_IMGS = BASE_PATH + 'splited'
    MODEL_NAME = "mnasNet"
    LOG_DIR = f"logs/fit/{MODEL_NAME}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    WEIGHT_PATH = BASE_PATH + "weights/"
    if not os.path.exists(WEIGHT_PATH):
        os.makedirs(WEIGHT_PATH)

    CHECKPOINT_PATH = BASE_PATH + "weights/checkpoint/"
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    CHECKPOINT_ALL_PATH = BASE_PATH + "weights/checkpoint_all/"
    if not os.path.exists(CHECKPOINT_ALL_PATH):
        os.makedirs(CHECKPOINT_ALL_PATH)

    MODEL_PATH = "/home/ec2-user/SageMaker/" + "models/"
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    md = Models()
    train_generator, validation_generator, test_generator = md.get_generators(PATH_IMGS, IMG_SIZE, BATCH_SIZE)

    # Criar modelo dentro da estratégia
    list_device = ["/gpu:0"]
    strategy = md.get_strategy(list_device)
    with strategy.scope():
        model = md.create_mnasnet_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Callbacks
    early_stopping, checkpoint, checkpoint_all, tensorboard_callback = get_callbacks(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH, 
        checkpoint_all_path=CHECKPOINT_ALL_PATH, 
        delta=DELTA, 
        pacience=PACIENTE, 
        log_dir=LOG_DIR,
        save_model_per_epoch=True)

    # Treinamento
    history = train_model(
        model=model, 
        model_name=MODEL_NAME,
        model_path=MODEL_PATH, 
        weights_path=WEIGHT_PATH, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        early_stopping=early_stopping, 
        checkpoint=checkpoint, 
        checkpoint_all=checkpoint_all,
        tensorboard_callback=tensorboard_callback, 
        train_generator=train_generator, 
        val_generator=validation_generator,
        initial_epoch=0,
        load_weight=None
    )

    # Plotar evolução do treinamento
    plot_training(history)

    print("\nTreinamento concluído com sucesso!")
    input("\nPressione Enter para voltar ao menu principal...")

def evaluate_model():
    """Executa a avaliação de um modelo existente"""
    print("\nIniciando avaliação do modelo...")
    
    BASE_PATH = "/mnt/efs-tcga/HEAL_Workspace/macenko_datas/"
    MODEL_NAME = "mnasNet"
    PATH_IMGS = BASE_PATH + 'splited'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 128
    
    # Carregar geradores
    md = Models()
    _, _, test_generator = md.get_generators(PATH_IMGS, IMG_SIZE, BATCH_SIZE)

    # Caminho do modelo salvo
    model_path = BASE_PATH + "/models" + f"{MODEL_NAME}_model_224x224.keras"

    # Carregar modelo
    model = load_model(model_path, compile=False)
    class_indices = test_generator.class_indices  # Índices das classes
    class_labels = list(class_indices.keys())  # Nomes das classes

    # Avaliação no conjunto de teste
    y_true = test_generator.classes  # Classes verdadeiras
    y_pred_probs = model.predict(test_generator)  # Probabilidades preditas
    y_pred = np.argmax(y_pred_probs, axis=1)  # Classes preditas

    # Calcular métricas
    calculate_metrics(y_true, y_pred, class_labels)

    # Plotar matriz de confusão
    plot_confusion_matrix(y_true, y_pred, class_labels)

    input("\nPressione Enter para voltar ao menu principal...")

def continue_training():
    """Continua o treinamento de um modelo existente"""
    print("\nIniciando continuação do treinamento...")
    
    # Verifica se há GPU disponível
    get_gpu_memory()

    # Configurações
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 128
    EPOCHS = 100
    NUM_CLASSES = 2
    PACIENTE = 7
    DELTA = 0.001
    INITIAL_EPOCH = 5  # Mude conforme necessário

    BASE_PATH = "/mnt/efs-tcga/HEAL_Workspace/macenko_datas/"
    PATH_IMGS = BASE_PATH + 'splited'
    MODEL_NAME = "mnasNet"
    LOG_DIR = f"logs/fit/{MODEL_NAME}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    WEIGHT_PATH = BASE_PATH + "weights/"
    CHECKPOINT_PATH = BASE_PATH + "weights/checkpoint/"
    CHECKPOINT_ALL_PATH = BASE_PATH + "weights/checkpoint_all/"
    MODEL_PATH = "/home/ec2-user/SageMaker/" + "models/"

    # Carregar modelo e pesos
    model = tf.keras.models.load_model(BASE_PATH + "weights/checkpoint_all/" + f'{MODEL_NAME}_epoch_05.keras')
    load_weight = BASE_PATH + "/weights/checkpoint_all/" + f"{MODEL_NAME}_epoch_05.keras"

    md = Models()
    train_generator, validation_generator, test_generator = md.get_generators(PATH_IMGS, IMG_SIZE, BATCH_SIZE)

    # Criar modelo dentro da estratégia
    list_device = ["/gpu:0"]
    strategy = md.get_strategy(list_device)
    with strategy.scope():
        model = md.create_mnasnet_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Callbacks
    early_stopping, checkpoint, checkpoint_all, tensorboard_callback = get_callbacks(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH, 
        checkpoint_all_path=CHECKPOINT_ALL_PATH, 
        delta=DELTA, 
        pacience=PACIENTE, 
        log_dir=LOG_DIR,
        save_model_per_epoch=True)

    # Treinamento
    history = train_model(
        model=model, 
        model_name=MODEL_NAME,
        model_path=MODEL_PATH, 
        weights_path=WEIGHT_PATH, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        early_stopping=early_stopping, 
        checkpoint=checkpoint, 
        checkpoint_all=checkpoint_all,
        tensorboard_callback=tensorboard_callback, 
        train_generator=train_generator, 
        val_generator=validation_generator,
        initial_epoch=INITIAL_EPOCH,
        load_weight=load_weight
    )

    # Plotar evolução do treinamento
    plot_training(history)

    print("\nTreinamento adicional concluído com sucesso!")
    input("\nPressione Enter para voltar ao menu principal...")

def main():
    """Função principal que controla o fluxo do programa"""
    while True:
        show_menu()
        choice = get_user_choice()

        if choice == 1:
            train_new_model()
        elif choice == 2:
            evaluate_model()
        elif choice == 3:
            continue_training()
        elif choice == 4:
            print("\nSaindo do sistema...")
            sys.exit()

if __name__ == "__main__":
    main()