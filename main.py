import sys
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import datetime
import os
import time

from utils.config import get_gpu_memory
from utils.config import get_callbacks
from models.train import train_model
from utils.visualize import plot_training
from utils.visualize import plot_confusion_matrix
from utils.visualize import plot_roc_curve
from utils.evaluate import evaluate_model
from utils.evaluate import calculate_metrics

from models.model import Models


# Configurações
IMG_SIZE = (224, 224)
BATCH_SIZE = 128
EPOCHS = 100
NUM_CLASSES = 2
PACIENTE = 7
DELTA = 0.001

BASE_PATH = "/run/user/1000/gvfs/smb-share:server=192.168.1.64,share=datasets/"
PATH_IMGS = BASE_PATH + 'Dataset-splited'

WEIGHT_PATH = BASE_PATH + "weights/"
if not os.path.exists(WEIGHT_PATH):
    os.makedirs(WEIGHT_PATH)

CHECKPOINT_PATH = BASE_PATH + "weights/checkpoint/"
if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

CHECKPOINT_ALL_PATH = BASE_PATH + "weights/checkpoint_all/"
if not os.path.exists(CHECKPOINT_ALL_PATH):
    os.makedirs(CHECKPOINT_ALL_PATH)

MODEL_PATH = BASE_PATH + "models/"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)



model_instance = Models()
# Carregar os geradores de dados
print(50*"=")
print("               Carregando o dataset...")
print(50*"=")
train_generator, validation_generator, test_generator = None, None, None
# train_generator, validation_generator, test_generator = model_instance.get_generators(PATH_IMGS, IMG_SIZE, BATCH_SIZE)
print(50*"=")
print("          Dataset carregado com sucesso!")
print(50*"=")


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

def call_model(model_name):
        """Chama o modelo baseado no nome"""
        if model_name == "alexnet":
            model = model_instance.create_alexnet_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            return model
        
        elif model_name == "densenet":
            model = model_instance.create_densenet_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            return model
        
        elif model_name == "efficientnet":
            model = model_instance.create_efficientnet_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            return model
        
        elif model_name == "inception":
            model = model_instance.create_inception_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            return model
        
        elif model_name == "mobilenet":
            model = model_instance.create_mobilenetv2_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            return model
        
        elif model_name == "mobilenetv2":
            model = model_instance.create_mobilenetv3_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
            model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
            return model
        
        elif model_name == "resnet50":
            model = model_instance.create_resnet50_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            return model
        
        elif model_name == "vgg16":
            model = model_instance.create_vgg16_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            return model
        
        elif model_name == "shufflenet":
            model = model_instance.create_shuffnet_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            return model
        
        else:
            raise ValueError("Modelo não reconhecido.")

def train_new_model(model_instance, MODEL_NAME, LOG_DIR):
    """Executa o treinamento de um novo modelo"""
    print("\nIniciando treinamento de novo modelo...")
    
    # Verifica se há GPU disponível
    get_gpu_memory()

    # Criar modelo dentro da estratégia
    list_device = ["/gpu:0"]
    strategy = model_instance.get_strategy(list_device)
    with strategy.scope():
        model = call_model(MODEL_NAME)
        
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

def evaluate_model(test_generator, MODEL_NAME):
    """Executa a avaliação de um modelo existente"""
    print("\nIniciando avaliação do modelo...")
    

    # Caminho do modelo salvo
    model_path = BASE_PATH + "/models" + f"{MODEL_NAME}_model_224x224.keras"
    if not os.path.exists(model_path):
        model_path = input("Por favor, insira o caminho completo do modelo que deseja avaliar: ")
        
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
    
    # Plotar curva ROC
    plot_roc_curve(y_true, y_pred_probs, class_labels)

    input("\nPressione Enter para voltar ao menu principal...")

def continue_training(model_instance, MODEL_NAME, LOG_DIR, train_generator, validation_generator):   
    """Continua o treinamento de um modelo existente"""
    print("\nIniciando continuação do treinamento...")
    
    # Verifica se há GPU disponível
    get_gpu_memory()

    initia_epoch = int(input("Digite o número do epoch inicial para continuar o treinamento: "))
    if initia_epoch < 0:
        print("Epoch inicial inválido. O treinamento não pode ser continuado.")
        return
    else:
        print(f"Continuando o treinamento a partir do epoch {initia_epoch}...")
        
    # Carregar modelo e pesos
    model = tf.keras.models.load_model(CHECKPOINT_ALL_PATH + f'{MODEL_NAME}_epoch_{initia_epoch}.keras')
    load_weight = CHECKPOINT_ALL_PATH + f"{MODEL_NAME}_epoch_{initia_epoch}.keras"

    # Criar modelo dentro da estratégia
    list_device = ["/gpu:0"]
    strategy = model_instance.get_strategy(list_device)
    with strategy.scope():
        model = call_model(MODEL_NAME)
        
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
        initial_epoch=initia_epoch,
        load_weight=load_weight
    )

    # Plotar evolução do treinamento
    plot_training(history)

    print("\nTreinamento adicional concluído com sucesso!")
    input("\nPressione Enter para voltar ao menu principal...")
    
    
def models_available():
    clear_screen()
    print("="*50)
    print("Modelos disponíveis:")
    print("1.   alexnet")
    print("2.   densenet")
    print("3.   efficientnet")
    print("4.   inception")
    print("5.   mobilenet")
    print("6.   mobilenetv2")
    print("7.   resnet50")
    print("8.   vgg16")
    print("9.   shufflenet")
    print("10.  retornar para o menu principal")
    print("="*50)
    model_choice = int(input("Escolha um dos modelos (1-10): "))
    
    model_options = {
        1: "alexnet",
        2: "densenet",
        3: "efficientnet",
        4: "inception",
        5: "mobilenet",
        6: "mobilenetv2",
        7: "resnet50",
        8: "vgg16",
        9: "shufflenet"
    }

    return model_options, model_choice

def get_model_name(model_choice):
    """Obtém o nome do modelo a partir da escolha do usuário"""
    
    if model_choice == 1:
        return "alexnet"
        
    elif model_choice == 2:
        return "densenet"
        
    elif model_choice == 3:
        return "efficientnet"
        
    elif model_choice == 4:
        return "inception"
        
    elif model_choice == 5:
        return "mobilenet"
        
    elif model_choice == 6:
        return "mobilenetv2"
        
    elif model_choice == 7:
        return "resnet50"
        
    elif model_choice == 8:
        return "vgg16"
        
    elif model_choice == 9:
        return "shufflenet"
    
    else:
        print("Opção inválida.")

    

def main():
    MODEL_NAME = None
    LOG_DIR = None
    
    """Função principal que controla o fluxo do programa"""    
    while True:
        show_menu()
        choice = get_user_choice()

        if choice == 1:
            model_options, model_choice = models_available()
            if model_choice == 10:
                continue
            
            if not model_choice in model_options:
                print("Opção inválida.")
                time.sleep(2)
                continue
            else:
                MODEL_NAME = get_model_name(model_choice)
                print(f"Modelo escolhido: {MODEL_NAME}")
                print(50*"=")
            time.sleep(2)
    
            LOG_DIR = f"logs/fit/{MODEL_NAME}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_new_model(model_instance, MODEL_NAME, LOG_DIR)
            
        elif choice == 2:
            model_options, model_choice = models_available()
            if model_choice == 10:
                continue
            
            if not model_choice in model_options:
                print("Opção inválida.")
                time.sleep(2)
                continue
            else:
                MODEL_NAME = get_model_name(model_choice)
                print(f"Modelo escolhido: {MODEL_NAME}")
                print(50*"=")
            time.sleep(2)
            # Avaliação do modelo
            evaluate_model(test_generator, MODEL_NAME)
        elif choice == 3:
            model_options, model_choice = models_available()
            if model_choice == 10:
                continue
            
            if not model_choice in model_options:
                print("Opção inválida.")
                time.sleep(2)
                continue
            else:
                MODEL_NAME = get_model_name(model_choice)
                print(f"Modelo escolhido: {MODEL_NAME}")
                print(50*"=")
            time.sleep(2)
    
            LOG_DIR = f"logs/fit/{MODEL_NAME}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")            
            continue_training(model_instance, MODEL_NAME, LOG_DIR, train_generator, validation_generator)
        elif choice == 4:
            print("\nSaindo do sistema...")
            sys.exit()

if __name__ == "__main__":
    main()