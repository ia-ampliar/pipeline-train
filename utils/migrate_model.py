import os
import tensorflow as tf
from models.combined_loss import CombinedBCESoftF1Loss 

def migrate_model(old_model_path, new_model_path):
    try:
        # Carrega o modelo antigo, usando custom_objects (pois o antigo foi salvo sem o serializador correto)
        model = tf.keras.models.load_model(
            old_model_path,
            custom_objects={'CombinedBCESoftF1Loss': CombinedBCESoftF1Loss}
        )
        print(f"[INFO] Modelo carregado de: {old_model_path}")

        # Re-salva o modelo no novo caminho (com a loss agora serializável)
        model.save(new_model_path)
        print(f"[INFO] Modelo re-salvo com sucesso em: {new_model_path}")

    except Exception as e:
        print(f"[ERRO] Falha ao migrar o modelo {old_model_path}: {e}")

