import os
import tensorflow as tf
from models.combined_loss import CombinedBCESoftF1Loss  # Importa a loss já corrigida!

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

def main():
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

if __name__ == "__main__":
    main()
