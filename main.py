import sys
import torch
import numpy as np
import os
import datetime
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.config import get_device
from utils.config import get_callbacks
from models.train import train_model
from utils.visualize import plot_training
from utils.evaluate import evaluate_model
from models.model import MobileNetV2, ResNet50, VGG16, AlexNet, EfficientNet, DenseNet121

# Configurações
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 100
NUM_CLASSES = 2
PACIENTE = 7
DELTA = 0.001

BASE_PATH = "/home/ampliar/computerVision/train/"
PATH_IMGS = BASE_PATH + "splited"

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

# Função para carregar o dataset com DataLoader
def get_dataloaders(base_path, img_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(base_path, 'train'), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(base_path, 'val'), transform=transform)
    test_ds = datasets.ImageFolder(os.path.join(base_path, 'test'), transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, train_ds.classes

# Carregar o dataset
print(50*"=")
print("               Carregando o dataset...")
print(50*"=")
train_loader, val_loader, test_loader, class_names = get_dataloaders(PATH_IMGS, IMG_SIZE, BATCH_SIZE)
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
        model = AlexNet(num_classes=NUM_CLASSES)
        model.to(get_device())
        return model

    elif model_name == "densenet":
        model = DenseNet121(num_classes=NUM_CLASSES)
        model.to(get_device())
        return model
    
    elif model_name == "efficientnet":
        model = EfficientNet(num_classes=NUM_CLASSES)
        model.to(get_device())
        return model

    elif model_name == "mobilenet":
        model = MobileNetV2(num_classes=NUM_CLASSES)
        model.to(get_device())
        return model

    elif model_name == "resnet50":
        model = ResNet50(num_classes=NUM_CLASSES)
        model.to(get_device())
        return model
    
    elif model_name == "vgg16":
        model = VGG16(num_classes=NUM_CLASSES)
        model.to(get_device())
        return model

    else:
        print("Modelo não reconhecido.")
        sys.exit()

def train_new_model(MODEL_NAME, LOG_DIR):
    """Executa o treinamento de um novo modelo"""
    print("\nIniciando treinamento de novo modelo...")

    model = call_model(MODEL_NAME)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    writer = get_callbacks(LOG_DIR)
    
    # Treinamento
    train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=get_device(), 
        epochs=EPOCHS, 
        optimizer=optimizer, 
        criterion=criterion, 
        scheduler=None, 
        save_path=CHECKPOINT_PATH + f'{MODEL_NAME}_best.pth'
    )

    print("\nTreinamento concluído com sucesso!")
    input("\nPressione Enter para voltar ao menu principal...")

def continue_training(MODEL_NAME, LOG_DIR):   
    """Continua o treinamento de um modelo existente"""
    print("\nIniciando continuação do treinamento...")

    model = call_model(MODEL_NAME)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    init_epoch = int(input("Digite o número do epoch inicial para continuar o treinamento: "))
    checkpoint_path = CHECKPOINT_ALL_PATH + f"{MODEL_NAME}_epoch_{init_epoch}.pth"
    model.load_state_dict(torch.load(checkpoint_path))

    writer = get_callbacks(LOG_DIR)

    # Continuação do treinamento
    train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=get_device(), 
        epochs=EPOCHS, 
        optimizer=optimizer, 
        criterion=criterion, 
        scheduler=None, 
        save_path=checkpoint_path
    )

    print("\nTreinamento adicional concluído com sucesso!")
    input("\nPressione Enter para voltar ao menu principal...")

def models_available():
    clear_screen()
    print("="*50)
    print("Modelos disponíveis:")
    print("1.   alexnet")
    print("2.   densenet")
    print("3.   efficientnet")
    print("4.   mobilenet")
    print("5.   resnet50")
    print("6.   vgg16")
    print("7.   retornar para o menu principal")
    print("="*50)
    model_choice = int(input("Escolha um dos modelos (1-6): "))
    
    model_options = {
        1: "alexnet",
        2: "densenet",
        3: "efficientnet",
        4: "mobilenet",
        5: "resnet50",
        6: "vgg16"
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
        return "mobilenet"
    elif model_choice == 5:
        return "resnet50"
    elif model_choice == 6:
        return "vgg16"
    else:
        print("Opção inválida.")

def main():
    MODEL_NAME = None
    LOG_DIR = None
    
    while True:
        show_menu()
        choice = get_user_choice()

        if choice == 1:
            model_options, model_choice = models_available()
            if model_choice == 7:
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
            train_new_model(MODEL_NAME, LOG_DIR)
            
        elif choice == 2:
            model_options, model_choice = models_available()
            if model_choice == 7:
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
            evaluate_model(model_instance, test_loader, get_device())
        elif choice == 3:
            model_options, model_choice = models_available()
            if model_choice == 7:
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
            continue_training(MODEL_NAME, LOG_DIR)
        elif choice == 4:
            print("\nSaindo do sistema...")
            sys.exit()

if __name__ == "__main__":
    main()
