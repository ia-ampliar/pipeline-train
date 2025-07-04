import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator



from models.kfold_pipeline import generate_folds
from models.combined_loss import CombinedBCESoftF1Loss
from utils.migrate_model import migrate_model
from models.kfold_pipeline import get_csv_generators
from utils.evaluate_kfold import evaluate_test_set
from models.train import train_model_kfold
from utils.config import get_gpu_memory
from utils.config import get_callbacks
from models.train import train_model
from utils.visualize import plot_training
from utils.evaluate import evaluate_model

from utils.ensamble import (
    load_keras_models, predict_dataset, majority_vote,
    plot_roc_curves, compute_metrics, plot_conf_matrix
)

from models.model import Models


# Configurações
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 100
NUM_CLASSES = 4
PACIENTE = 7
DELTA = 0.001

BASE_PATH = "/mnt/efs-tcga/HEAL_Workspace/macenko_datas/"
PATH_IMGS = BASE_PATH + 'splited'

# CSV contendo os caminhos das imagens e labels
FOLDS_DIR = '/root/pipeline-train/outputs/folds/'

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

TEST_DIR = '/mnt/efs-tcga/HEAL_Workspace/macenko_datas/splited/test'
PROJECT_ROOT = '/root/pipeline-train'

METRICS_PATH = os.path.join(PROJECT_ROOT, 'metrics')
os.makedirs(METRICS_PATH, exist_ok=True)




model_instance = Models()


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
    print("2. Treina novo modelo com k-fold")
    print("3. Avaliar modelo existente")
    print("4. Avaliar modelo K-Fold")
    print("5. Continuar treinamento do modelo")
    print("6. Criar modelo ensemble")
    print("7. Migrar modelos antigos para o novo formato")
    print("8. Sair")
    print("\n" + "="*50)


def get_user_choice():
    """Obtém a escolha do usuário"""
    while True:
        try:
            choice = int(input("\nDigite sua opção (1-8): "))
            if 1 <= choice <= 7:
                return choice
            else:
                print("Opção inválida. Por favor, digite um número entre 1 e 8.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número.")


def call_model(model_name):
        """Chama o modelo baseado no nome"""
        loss_fn = CombinedBCESoftF1Loss(alpha=0.7)  # 70% BCE + 30% F1

        if model_name == "alexnet":
            model, _ = model_instance.create_alexnet(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=loss_fn,
            metrics=['accuracy',
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision')]
                )       
            # model.summary()
            return model
        
        elif model_name == "densenet":
            model, _ = model_instance.create_densenet_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss=loss_fn,
                    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                              tf.keras.metrics.Precision(name='precision')]
                )
            # model.summary()
            return model
        
        elif model_name == "efficientnet":
            model, _ = model_instance.create_efficientnet_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss= loss_fn,
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )
            # model.summary()
            return model

        elif model_name == "efficientnetv2":
            model, _ = model_instance.create_efficientnetb4_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss= loss_fn,
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                          tf.keras.metrics.Precision(name='precision')]
            )
            # model.summary()
            return model
        
        elif model_name == "inception":
            model, _ = model_instance.create_inception_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy",
                                                                    tf.keras.metrics.AUC(name='auc'),
                                                                    tf.keras.metrics.Precision(name='precision')]
                           )
            # model.summary()
            return model
        
        elif model_name == "mobilenet":
            model, _ = model_instance.create_mobilenet_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy",
                                                                    tf.keras.metrics.AUC(name='auc'),
                                                                    tf.keras.metrics.Precision(name='precision')]
                           )
            # model.summary()
            return model
        
        elif model_name == "mobilenetv2":
            model, _ = model_instance.create_mobilenetv2_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy",
                                                                    tf.keras.metrics.AUC(name='auc'),
                                                                    tf.keras.metrics.Precision(name='precision')]
                           )
            # model.summary()
            return model
        
        elif model_name == "resnet50":
            model, _ = model_instance.create_resnet50_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy",
                                                                    tf.keras.metrics.AUC(name='auc'),
                                                                    tf.keras.metrics.Precision(name='precision')]
                           )
            # model.summary()
            return model
        
        elif model_name == "vgg16":
            model, _ = model_instance.create_vgg16_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy",
                                                                    tf.keras.metrics.AUC(name='auc'),
                                                                    tf.keras.metrics.Precision(name='precision')]
                           )
            # model.summary()
            return model
        
        elif model_name == "resnet152":
            model, _ = model_instance.create_resnet152_model(pretrained=True, num_classes=NUM_CLASSES, img_size=IMG_SIZE)
            model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy",
                                                                    tf.keras.metrics.AUC(name='auc'),
                                                                    tf.keras.metrics.Precision(name='precision')]
                           )
            # model.summary()
            return model
        
        else:
            raise ValueError("Modelo não reconhecido.")


def train_new_model(model_instance, MODEL_NAME, LOG_DIR, train_generator, validation_generator):
    """Executa o treinamento de um novo modelo"""
    print("\nIniciando treinamento de novo modelo...")
    
    # Verifica se há GPU disponível
    get_gpu_memory()

    # Criar modelo dentro da estratégia
    list_device = None
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
    plot_training(history, MODEL_NAME)

    print("\nTreinamento concluído com sucesso!")
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
    list_device = None
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
    plot_training(history, MODEL_NAME)

    print("\nTreinamento adicional concluído com sucesso!")
    input("\nPressione Enter para voltar ao menu principal...")


def k_fold_model(model_instance, MODEL_NAME, LOG_DIR):
    """Executa o treinamento de um modelo com k-fold"""
    print("\nIniciando treinamento com k-fold...")
    
    # Verifica se há GPU disponível
    get_gpu_memory()

    # Criar modelo dentro da estratégia
    list_device = None
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

    # Treinamento com k-fold
    history = train_model_kfold(
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
        folds_dir=FOLDS_DIR,
        k=10,
        initial_epoch=0,
        load_weight=None
    )
    # Plotar evolução do treinamento
    plot_training(history, MODEL_NAME)
    print("\nTreinamento com k-fold concluído com sucesso!")
    input("\nPressione Enter para voltar ao menu principal...")
    
    
def models_available():
    clear_screen()
    print("="*50)
    print("Modelos disponíveis:")
    print("1.   alexnet")
    print("2.   densenet")
    print("3.   efficientnet")
    print("4.   efficientnetv2")
    print("5.   inception")
    print("6.   mobilenet")
    print("7.   mobilenetv2")
    print("8.   resnet50")
    print("9.   vgg16")
    print("10.  shufflenet")
    print("11.  retornar para o menu principal")
    print("="*50)
    model_choice = int(input("Escolha um dos modelos (1-10): "))
    
    model_options = {
        1: "alexnet",
        2: "densenet",
        3: "efficientnet",
        4: "efficientnetv2",
        5: "inception",
        6: "mobilenet",
        7: "mobilenetv2",
        8: "resnet50",
        9: "vgg16",
        10: "shufflenet"
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
        return "efficientnetv2"
        
    elif model_choice == 5:
        return "inception"
        
    elif model_choice == 6:
        return "mobilenet"
        
    elif model_choice == 7:
        return "mobilenetv2"
        
    elif model_choice == 8:
        return "resnet50"
        
    elif model_choice == 9:
        return "vgg16"
        
    elif model_choice == 10:
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
            
            # Carregar os geradores de dados
            print(50*"=")
            print("               Carregando o dataset...")
            print(50*"=")
            train_generator, validation_generator = model_instance.get_generators(PATH_IMGS, IMG_SIZE, BATCH_SIZE)
            print(50*"=")
            print("          Dataset carregado com sucesso!")
            print(50*"=")


            model_options, model_choice = models_available()
            if model_choice == 11:
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
            train_new_model(model_instance, MODEL_NAME, LOG_DIR, train_generator, validation_generator)
            
        elif choice == 2:
            model_options, model_choice = models_available()
            if model_choice == 11:
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
            print("➡️     Gerando os folds...")
            time.sleep(2)

            path_csv = input("Digite o caminho do CSV com os caminhos das imagens e labels: ")
            if not path_csv.endswith(".csv"):
                    print("Caminho inválido. O CSV deve terminar com '.csv'.")
                    continue
            generate_folds(path_csv, k=10, split_ratios=(0.7, 0.2, 0.1))
          
            print("➡️     Iniciando treinamento com k-fold...")
            time.sleep(2)
            k_fold_model(model_instance, MODEL_NAME, LOG_DIR)
       

        elif choice == 3:
            model_options, model_choice = models_available()
            if model_choice == 11:
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
            # Carregar os geradores de dados
            print(50*"=")
            print("               Carregando o dataset...")
            print(50*"=")
            test_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
                                    directory=PATH_IMGS + "/test/",
                                    target_size=IMG_SIZE,
                                    batch_size=BATCH_SIZE,
                                    class_mode="categorical",
                                    shuffle=False,
                                )
            
            # Avaliação do modelo
            evaluate_model(test_generator, MODEL_NAME, BASE_PATH, multiclass=True)


        elif choice == 4:
            print("\n==== Avaliação de modelos K-Fold ====\n")

            # Solicita os parâmetros ao usuário
            model_name = input("Digite o nome do modelo (ex: resnet, mobilenet): ").strip()
            model_dir = input("Digite o caminho da pasta onde estão os modelos (.keras): ").strip()
            folds_dir = input("Digite o caminho da pasta onde estão os arquivos dos folds (fold CSVs + test.csv): ").strip()
            output_dir = input("Digite o diretório de saída para salvar os resultados (padrão: train history/): ").strip()
            output_dir = output_dir if output_dir else "train history/"

            try:
                k = int(input("Digite o número de folds (ex: 10): ").strip())
            except ValueError:
                print("[ERRO] Número de folds inválido. Abortando.")
                return

            for fold in range(k):
                print(f"\n[INFO] Avaliando Fold {fold} - Modelo: {model_name}")

                model_path = os.path.join(model_dir, f'{model_name}_fold{fold}_model_224x224.keras')

                if not os.path.exists(model_path):
                    print(f"[AVISO] Modelo para Fold {fold} não encontrado: {model_path}. Pulando.")
                    continue

                evaluate_test_set(
                    model_path=model_path,
                    fold=fold,
                    model_name=model_name,
                    folds_dir=folds_dir,
                    output_base_dir=output_dir
                )

            print("\n[FINALIZADO] Avaliação de todos os folds concluída.")
            input("\nPressione Enter para voltar ao menu principal...")


        elif choice == 5:
             
            # Carregar os geradores de dados
            print(50*"=")
            print("               Carregando o dataset...")
            print(50*"=")
            train_generator, validation_generator = model_instance.get_generators(PATH_IMGS, IMG_SIZE, BATCH_SIZE)
            print(50*"=")
            print("          Dataset carregado com sucesso!")
            print(50*"=")

            model_options, model_choice = models_available()
            if model_choice == 11:
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


        elif choice == 6:

            BASE_PATH_MODEL = input("Digite o caminho base dos modelos: ")
            if not BASE_PATH_MODEL.endswith("/"):
                BASE_PATH_MODEL += "/"

            model_paths = []

            while True:
                path = input("Digite o caminho do modelo (ou 'sair' para terminar): ")
                
                if path.lower() == 'sair':
                    break
                
                # Você pode adicionar validação do caminho aqui se necessário
                model_paths.append(BASE_PATH_MODEL + path)
                
                continuar = input("Deseja adicionar outro modelo? (s/n): ")
                if continuar.lower() != 's':
                    break

            print("\nCaminhos dos modelos carregados:")
            for path in model_paths:
                print(path)

            model_names = [os.path.basename(p).replace(".keras", "") for p in model_paths]

            print("➡️     Carregando modelos...")
            models = load_keras_models(model_paths)

            print("➡️     Carregando imagens de teste...")
            test_ds = tf.keras.utils.image_dataset_from_directory(
                TEST_DIR,
                labels='inferred',
                label_mode='int',
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                shuffle=False
            )
            class_names = test_ds.class_names

            normalization_layer = tf.keras.layers.Rescaling(1./255)
            test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

            print("➡️     Realizando predições...")
            y_true, preds_list = predict_dataset(models, test_ds)

            print("➡️     Votando entre modelos...")
            ensemble_preds = majority_vote(preds_list)

            print("➡️     Plotando curvas ROC e calculando métricas...")
            individual_metrics = plot_roc_curves(y_true, preds_list, model_names, METRICS_PATH, multiclass=True)

            print("➡️     Plotando matriz de confusão...")
            plot_conf_matrix(y_true, ensemble_preds, class_names, METRICS_PATH)

            print("➡️     Salvando métricas no CSV...")
            ensemble_metrics = compute_metrics(y_true, ensemble_preds)
            ensemble_metrics['model'] = 'Majority_Vote'
            ensemble_metrics['auc'] = np.nan  # não aplicável diretamente

            full_metrics = pd.DataFrame(individual_metrics + [ensemble_metrics])
            full_metrics.to_csv(os.path.join(METRICS_PATH, 'model_metrics_log.csv'), index=False)

            print("✅     Concluído! Arquivos salvos em:", METRICS_PATH)
            input("\nPressione Enter para voltar ao menu principal...")


        elif choice == 7:
            model_dir = input("Digite o caminho da pasta onde estão os modelos antigos (.keras): ").strip()
            output_dir = input("Digite o caminho da pasta onde os modelos migrados serão salvos (pode ser a mesma): ").strip()

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for filename in os.listdir(model_dir):
                if filename.endswith(".keras"):
                    old_path = os.path.join(model_dir, filename)
                    new_path = os.path.join(output_dir, filename)

                    print(f"\n[MIGRANDO] {filename}")
                    migrate_model(old_path, new_path)

            print("\n[FINALIZADO] Migração de todos os modelos concluída.")
            input("\nPressione Enter para voltar ao menu principal...")

        elif choice == 8:
            print("\nSaindo do sistema...")
            sys.exit()

if __name__ == "__main__":
    main()