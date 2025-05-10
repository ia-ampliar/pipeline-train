import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score,
    recall_score, f1_score,
      confusion_matrix, ConfusionMatrixDisplay
)
from keras.saving import register_keras_serializable
from models.model import SpatialAttentionLayer



# === FUNÇÕES ===
def load_keras_models(paths):
    loaded_models = []
    for p in paths:
        try:
            # Primeira tentativa: carrega sem custom_objects
            model = tf.keras.models.load_model(p)
            loaded_models.append(model)
        except TypeError as e:
            if "SpatialAttentionLayer" in str(e):
                # Segunda tentativa: carrega com custom_objects
                try:
                    model = tf.keras.models.load_model(
                        p,
                        custom_objects={'SpatialAttentionLayer': SpatialAttentionLayer}
                    )
                    loaded_models.append(model)
                except Exception as e:
                    print(f"❌ Falha ao carregar {p} mesmo com custom_objects: {str(e)}")
            else:
                print(f"❌ Erro ao carregar {p}: {str(e)}")
        except Exception as e:
            print(f"❌ Erro inesperado ao carregar {p}: {str(e)}")
    return loaded_models



def predict_dataset(models, dataset):
    y_true = []
    preds_list = [[] for _ in models]

    for x_batch, y_batch in dataset:
        y_true.extend(y_batch.numpy())
        for i, model in enumerate(models):
            preds = model.predict(x_batch, verbose=0)
            preds_list[i].extend(preds)

    y_true = np.array(y_true)
    preds_list = [np.array(p) for p in preds_list]
    return y_true, preds_list

def majority_vote(preds_list):
    class_preds = [np.argmax(p, axis=1) for p in preds_list]
    stacked_preds = np.stack(class_preds)
    voted = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=stacked_preds)
    return voted

def plot_roc_curves(y_true, preds_list, names, save_path):
    plt.figure(figsize=(10, 8))
    metrics = []

    for i, preds in enumerate(preds_list):
        fpr, tpr, _ = roc_curve(y_true, preds[:, 1])
        roc_auc = auc(fpr, tpr)

        # Nome-base simplificado para salvar arquivos
        model_base = names[i].split('_epoch')[0] if '_epoch' in names[i] else names[i].split('_')[0]


        # Salvar arquivos .npy
        path_npy = os.path.join(save_path, 'npy')
        os.makedirs(path_npy, exist_ok=True)

        np.save(os.path.join(path_npy, f'{model_base}_fpr.npy'), fpr)
        np.save(os.path.join(path_npy, f'{model_base}_tpr.npy'), tpr)
        np.save(os.path.join(path_npy, f'{model_base}_auc.npy'), np.array([roc_auc]))

        plt.plot(fpr, tpr, label=f"{model_base} (AUC = {roc_auc:.2f})")

        y_pred = np.argmax(preds, axis=1)
        metrics.append({
            'model': model_base,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc': roc_auc
        })

    # Ensemble ROC
    avg_preds = np.mean(np.stack(preds_list), axis=0)
    fpr_ens, tpr_ens, _ = roc_curve(y_true, avg_preds[:, 1])
    auc_ens = auc(fpr_ens, tpr_ens)
    os.makedirs(path_npy, exist_ok=True)
    np.save(os.path.join(path_npy, 'ensemble_fpr.npy'), fpr_ens)
    np.save(os.path.join(path_npy, 'ensemble_tpr.npy'), tpr_ens)
    np.save(os.path.join(path_npy, 'ensemble_auc.npy'), np.array([auc_ens]))

    plt.plot(fpr_ens, tpr_ens, lw=3, linestyle="--", color="black", label=f"ensemble (AUC = {auc_ens:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC - Todos os Modelos")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'roc_curves.png'))
    plt.close()

    return metrics


def plot_conf_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão - Modelo Ensemble (Voto Majoritário)")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
