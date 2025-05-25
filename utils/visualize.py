import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_training(history, network_name, pickle_path=None):
    BASE_PATH = "metrics"
    os.makedirs(BASE_PATH, exist_ok=True)

    plt.figure(figsize=(12, 5))

    if pickle_path is not None:
        with open(pickle_path, 'rb') as f:
            history = pickle.load(f)

        if not isinstance(history, dict):
            print("Erro: O conteúdo do arquivo não é um dicionário válido.")
            return

        plt.subplot(1, 2, 1)
        plt.plot(history.get('loss', []), label='Loss Treinamento')
        plt.plot(history.get('val_loss', []), label='Loss Validação')
        plt.title(f'Evolução da Loss: {network_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.get('accuracy', []), label='Acurácia Treinamento')
        plt.plot(history.get('val_accuracy', []), label='Acurácia Validação')
        plt.title(f'Evolução da Acurácia: {network_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()

    else:
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Loss Treinamento')
        plt.plot(history['val_loss'], label='Loss Validação')
        plt.title(f'Evolução da Loss: {network_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Acurácia Treinamento')
        plt.plot(history['val_accuracy'], label='Acurácia Validação')
        plt.title(f'Evolução da Acurácia: {network_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()

    plt.tight_layout()
    output_path = os.path.join(BASE_PATH, f"training_history_(loss_acc)_{network_name}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Figura salva em: {output_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, network_name):
    BASE_PATH = "metrics/"
    os.makedirs(BASE_PATH, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão: " + network_name)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Classe Verdadeira")
    plt.xlabel("Classe Predita")
    plt.savefig(os.path.join(BASE_PATH, f'confuse_matrix_{network_name}.png'))
    plt.tight_layout()
    plt.close()


def plot_roc_curve(y_true, y_pred_probs, network_name, multiclass=False):
    BASE_PATH = "metrics/"
    os.makedirs(BASE_PATH, exist_ok=True)

    n_classes = y_pred_probs.shape[1]
    plt.figure(figsize=(10, 8))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"Classe 1 (AUC = {roc_auc:.2f})")
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        colors = cycle(["blue", "green", "red", "purple", "orange", "brown"])
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f"Classe {i} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title(f"Curva ROC - {network_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(BASE_PATH, f'roc_curve_{network_name}.png'))
    plt.close()
