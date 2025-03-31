# train.py
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def train_model(model, model_name, model_path, weights_path, batch_size, epochs, 
                early_stopping, checkpoint, checkpoint_all, 
                tensorboard_callback, train_generator, val_generator, 
                initial_epoch=0, load_weight=None):
    
    # Verifica se há pesos salvos e carrega-os se initial_epoch > 0
    if initial_epoch > 0:
        try:
            if load_weight is None:
                model.load_load_s(load_weight)
                print(f"Carregando pesos salvos para retomar treinamento a partir da época {initial_epoch}.")
        except:
            print(f"Arquivo de pesos não encontrado em {load_weight}. Iniciando treinamento do zero.")
            initial_epoch = 0  # Se não houver pesos, começa do zero

    if checkpoint_all is not None:
        # Treinamento
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            initial_epoch=initial_epoch,  # Define a época inicial
            steps_per_epoch=train_generator.samples // batch_size,
            validation_steps=val_generator.samples // batch_size,
            callbacks=[early_stopping, checkpoint, checkpoint_all, tensorboard_callback],
            verbose=1
        )
    else:
        # Treinamento
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            initial_epoch=initial_epoch,  # Define a época inicial
            steps_per_epoch=train_generator.samples // batch_size,
            validation_steps=val_generator.samples // batch_size,
            callbacks=[early_stopping, checkpoint, tensorboard_callback],
            verbose=1
        )

    # Salvar os pesos finais
    model.save_weights(weights_path + f'{model_name}_weights_224x224.h5')
    model.save(model_path + f'{model_name}_model_224x224.keras')
    print(f"Pesos treinados salvos em: {weights_path}")

    return history