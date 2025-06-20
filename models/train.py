# train.py
import os
import pandas as pd
import tensorflow as tf
from datetime import datetime
import pickle
from models.kfold_pipeline import get_csv_generators, EpochTimer
from models.model import Models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger




def train_model(model, model_name, model_path, weights_path, batch_size, epochs, 
                early_stopping, checkpoint, checkpoint_all, 
                tensorboard_callback, train_generator, val_generator, 
                initial_epoch=0, load_weight=None):
    
    BASE_DIR = "train history/"
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    # Verifica se há pesos salvos e carrega-os se initial_epoch > 0
    if initial_epoch > 0:
        try:
            if load_weight is not None:
                model = tf.keras.models.load_model(load_weight)
                print(f"Carregando pesos salvos para retomar treinamento a partir da época {initial_epoch}.")
        except:
            print(f"Arquivo de pesos não encontrado em {load_weight}. Iniciando treinamento do zero.")
            initial_epoch = 0

    # Lista de callbacks
    callbacks_list = [early_stopping, checkpoint, tensorboard_callback,  EpochTimer()]
    if checkpoint_all is not None:
        callbacks_list.append(checkpoint_all)
    
    # Adiciona callback para salvar histórico em CSV
    csv_logger = tf.keras.callbacks.CSVLogger(
        f'{model_name}_training_history.csv',
        separator=',',
        append=False
    )
    callbacks_list.append(csv_logger)

    # Treinamento
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_steps=val_generator.samples // batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Salva o histórico completo em pickle (opcional)
    with open(BASE_DIR + f'{model_name}_historico.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Processa o histórico para criar um DataFrame mais completo
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, 'epoch', range(initial_epoch + 1, initial_epoch + 1 + len(history_df)))
    history_df.insert(1, 'model_name', model_name)

    # Adiciona coluna 'best_epoch' com marcação booleana
    if hasattr(early_stopping, 'best_epoch'):
        best_epoch_absolute = early_stopping.stopped_epoch - early_stopping.patience
        history_df['is_best_epoch'] = history_df['epoch'] == (best_epoch_absolute + 1)  
    else:
        history_df['is_best_epoch'] = False

    # Salva em CSV com informações adicionais
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = BASE_DIR + f'{model_name}_training_history_{timestamp}.csv'
    history_df.to_csv(csv_filename, index=False)
    print(f"Histórico completo salvo em: {csv_filename}")

    # Salvar os pesos finais
    model.save_weights(weights_path + f'{model_name}_weights_224x224.weights.h5')
    model.save(model_path + f'{model_name}_model_224x224.keras')
    print(f"Pesos treinados salvos em: {weights_path}")

    return history


def train_model_kfold(model, model_name, model_path, weights_path, batch_size, epochs, 
                early_stopping, checkpoint, checkpoint_all, 
                tensorboard_callback, folds_dir, k=10,
                initial_epoch=0, load_weight=None):
    
    BASE_DIR = "train history/"
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)


    for fold in range(k):
        print(f"\n[INFO] Treinando Fold {fold + 1}/{k}")

        train_csv = os.path.join(folds_dir, f"fold_{fold}_train.csv")
        val_csv = os.path.join(folds_dir, f"fold_{fold}_val.csv")
        test_csv = os.path.join(folds_dir, "test.csv")

        train_gen, val_gen, test_gen = get_csv_generators(
            train_csv, val_csv, test_csv,
            image_size=(224, 224), batch_size=batch_size
        )


        # Define o caminho para salvar os pesos e o modelo
        csv_logger = CSVLogger(f'{model_name}_training_history.csv', separator=',', append=False)
        epoch_timer = EpochTimer()

        callbacks_list = [early_stopping, checkpoint, tensorboard_callback, epoch_timer, csv_logger]
        if checkpoint_all is not None:
            callbacks_list.append(checkpoint_all)

        # Verifica se há pesos salvos para continuar treinamento
        if initial_epoch > 0 and load_weight is not None:
            try:
                model = tf.keras.models.load_model(load_weight)
                print(f"Carregando pesos de {load_weight} para continuar da época {initial_epoch}.")
            except Exception as e:
                print(f"[ERRO] Falha ao carregar pesos: {e}. Iniciando do zero.")
                initial_epoch = 0

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=train_gen.__len__(),
            validation_steps=val_gen.__len__(),
            callbacks=callbacks_list,
            verbose=1
        )

        # Salva histórico em pickle
        with open(BASE_DIR + f'{model_name}_historico.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        # DataFrame para CSV final
        history_df = pd.DataFrame(history.history)
        history_df.insert(0, 'epoch', range(initial_epoch + 1, initial_epoch + 1 + len(history_df)))
        history_df.insert(1, 'model_name', model_name)

        if hasattr(early_stopping, 'stopped_epoch'):
            best_epoch = early_stopping.stopped_epoch - early_stopping.patience + 1
            history_df['is_best_epoch'] = history_df['epoch'] == best_epoch
        else:
            history_df['is_best_epoch'] = False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = BASE_DIR + f'{model_name}_training_history_{timestamp}.csv'
        history_df.to_csv(csv_filename, index=False)
        print(f"Histórico salvo em: {csv_filename}")

        model.save_weights(weights_path + f'{model_name}_weights_224x224.weights.h5')
        model.save(model_path + f'{model_name}_model_224x224.keras')
        print(f"Pesos salvos para Fold {fold}: {weights_path}")

        # Avalia no teste
        if test_gen:
            test_loss, test_acc = model.evaluate(test_gen)
            print(f"[TESTE] Fold {fold} - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

        return history
