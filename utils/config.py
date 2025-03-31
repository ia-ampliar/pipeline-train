import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def get_gpu_memory():
    """
    Get the GPU memory usage.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU {gpu.name} memory growth set to True")
    else:
        print("No GPU found, using CPU.")


def get_callbacks(model_name, checkpoint_path, checkpoint_all_path, delta, pacience, log_dir, save_model_per_epoch=False):
    """
    Get the callbacks for training.
    """
    early_stopping = None
    checkpoint = None
    checkpoint_all = None
    tensorboard_callback = None

    if save_model_per_epoch:
       # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', 
                                       patience=7, min_delta=0.001, 
                                       restore_best_weights=True, 
                                       verbose=1)

        # Salva apenas o melhor modelo escolhido pelo early stopping
        checkpoint = ModelCheckpoint(checkpoint_path + f"{model_name}_checkpoint.keras", 
                                          save_weights_only=False,
                                          save_best_only=True, 
                                          monitor='val_loss', 
                                          verbose=1)

        # Salva o modelo em toda época
        checkpoint_all = ModelCheckpoint(checkpoint_all_path + f"{model_name}_checkpoint.keras", 
                                         save_weights_only=False, 
                                         save_best_only=False, 
                                         verbose=1)
    
        tensorboard_callback = TensorBoard(log_dir=log_dir, 
                                           histogram_freq=1, 
                                           write_graph=True)

        return early_stopping, checkpoint, checkpoint_all, tensorboard_callback


    else:
        early_stopping = EarlyStopping(monitor='val_loss', 
                                    patience=pacience, 
                                    min_delta=delta, 
                                    restore_best_weights=True,
                                    verbose=1)
        
        checkpoint = ModelCheckpoint(checkpoint_path + f"{model_name}_checkpoint.keras",
                                    save_weights_only=False, 
                                    save_best_only=True, 
                                    monitor='val_loss', 
                                    verbose=1)
        
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

        return early_stopping, checkpoint, checkpoint_all, tensorboard_callback


