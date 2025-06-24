import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import tensorflow as tf
from models.kfold_pipeline import get_csv_generators
from models.combined_loss import CombinedBCESoftF1Loss 

def evaluate_test_set(model_path, fold, model_name, folds_dir, output_base_dir="train history/"):
    try:
        # ===== 1. Tentativa de carregar o modelo =====
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"[INFO] Modelo carregado com sucesso: {model_path}")
        except Exception as e1:
            print(f"[AVISO] Não foi possível carregar o modelo normalmente: {e1}")
            print("[INFO] Tentando carregar com custom_objects...")
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'CombinedBCESoftF1Loss': CombinedBCESoftF1Loss}
                )
                print(f"[INFO] Modelo carregado com custom_objects: {model_path}")
            except Exception as e2:
                print(f"[ERRO] Falha ao carregar o modelo com custom_objects também: {e2}")
                return

        # ===== 2. Carregar test generator =====
        test_csv = os.path.join(folds_dir, "test.csv")
        if not os.path.exists(test_csv):
            print(f"[ERRO] test.csv não encontrado em: {test_csv}")
            return

        _, _, test_gen = get_csv_generators(
            train_csv=None,
            val_csv=None,
            test_csv=test_csv,
            image_size=(224, 224),
            batch_size=64
        )

        # ===== 3. Criar pasta do fold =====
        fold_dir = os.path.join(output_base_dir, f"{model_name}_fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # ===== 4. Avaliação no test set =====
        results = model.evaluate(test_gen, verbose=1)
        metric_names = model.metrics_names
        results_dict = {name: value for name, value in zip(metric_names, results)}

        metrics_df = pd.DataFrame([results_dict])
        metrics_df.insert(0, 'fold', fold)
        metrics_df.to_csv(os.path.join(fold_dir, "test_metrics.csv"), index=False)

        # ===== 5. Previsões =====
        y_true = []
        y_pred = []

        for batch_x, batch_y in test_gen:
            preds = model.predict(batch_x)
            y_true.extend(np.argmax(batch_y, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
            if len(y_true) >= test_gen.samples:
                break

        y_true = np.array(y_true[:test_gen.samples])
        y_pred = np.array(y_pred[:test_gen.samples])

        # ===== 6. Matriz de Confusão =====
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusão - {model_name} - Fold {fold}")
        plt.xlabel('Previsto')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, "confusion_matrix.png"))
        plt.close()

        # ===== 7. Distribuição de Classes =====
        plt.figure(figsize=(8, 6))
        sns.histplot(y_true, color='blue', label='Real', kde=False, bins=len(np.unique(y_true)), alpha=0.5)
        sns.histplot(y_pred, color='orange', label='Previsto', kde=False, bins=len(np.unique(y_true)), alpha=0.5)
        plt.title(f"Distribuição de Classes - {model_name} - Fold {fold}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, "class_distribution.png"))
        plt.close()

        # ===== 8. Classification Report =====
        report_text = classification_report(y_true, y_pred, digits=4)
        with open(os.path.join(fold_dir, "classification_report.txt"), 'w') as f:
            f.write(report_text)

        # ===== 9. Métricas Adicionais =====
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        additional_metrics = pd.DataFrame([{
            'fold': fold,
            'accuracy': acc,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1
        }])
        additional_metrics.to_csv(os.path.join(fold_dir, "additional_metrics.csv"), index=False)

        # ===== 10. Curva de Loss/Accuracy =====
        try:
            history_path = os.path.join(output_base_dir, f'{model_name}_training_history.csv')
            if os.path.exists(history_path):
                history_df = pd.read_csv(history_path)
                fold_history = history_df[history_df['model_name'] == model_name]

                plt.figure(figsize=(10, 6))
                plt.plot(fold_history['epoch'], fold_history['loss'], label='Train Loss')
                plt.plot(fold_history['epoch'], fold_history['val_loss'], label='Val Loss')
                if 'accuracy' in fold_history.columns and 'val_accuracy' in fold_history.columns:
                    plt.plot(fold_history['epoch'], fold_history['accuracy'], label='Train Acc')
                    plt.plot(fold_history['epoch'], fold_history['val_accuracy'], label='Val Acc')
                plt.xlabel('Epoch')
                plt.ylabel('Loss/Accuracy')
                plt.title(f"Treino vs Validação - {model_name} - Fold {fold}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(fold_dir, "loss_accuracy_curve.png"))
                plt.close()
        except Exception as e:
            print(f"[AVISO] Erro ao gerar curva de loss/acc para Fold {fold}: {e}")

        print(f"[INFO] Avaliação completa do Fold {fold} - {model_name} salva em: {fold_dir}")

    except Exception as e:
        print(f"[ERRO] Falha na avaliação do Fold {fold} - {model_name}: {e}")
