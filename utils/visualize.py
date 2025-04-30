from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)
import os
import matplotlib.pyplot as plt
import pickle

def plot_training(history, network_name, pickle_path=None):
    """
    Plota a evolução da loss e acurácia durante o treinamento.
    """
    BASE_PATH = "metrics"
    os.makedirs(BASE_PATH, exist_ok=True)  # Cria o diretório se não existir (sem erros)

    plt.figure(figsize=(12, 5))

    if (pickle_path is not None) or (history is None):
        # Carrega o histórico de treinamento a partir do arquivo pickle
        with open(pickle_path, 'rb') as f:
            history = pickle.load(f)
        
        if not isinstance(history, dict):
            print("Erro: O conteúdo do arquivo não é um dicionário válido.")
            return
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.get('loss', []), label='Loss Treinamento')
        plt.plot(history.get('val_loss', []), label='Loss Validação')
        plt.title(f'Evolução da Loss: {network_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.get('accuracy', []), label='Acurácia Treinamento')
        plt.plot(history.get('val_accuracy', []), label='Acurácia Validação')
        plt.title(f'Evolução da Acurácia: {network_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()

    else:
        # Plot Loss (direto do history do Keras)
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Loss Treinamento')
        plt.plot(history.history['val_loss'], label='Loss Validação')
        plt.title(f'Evolução da Loss: {network_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Accuracy (direto do history do Keras)
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Acurácia Treinamento')
        plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
        plt.title(f'Evolução da Acurácia: {network_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()

    plt.tight_layout()
    
    # Salva a figura (usando os.path.join para compatibilidade entre sistemas)
    output_path = os.path.join(BASE_PATH, f"training_history_(loss_acc)_{network_name}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Figura salva em: {output_path}")


# Função para plotar a matriz de confusão
def plot_confusion_matrix(y_true, y_pred, class_names, network_name):
    """
    Plota a matriz de confusão.
    
    Args:
        y_true (array): Classes verdadeiras.
        y_pred (array): Classes preditas.
        class_names (list): Nomes das classes.
    """
    BASE_PATH = "metrics/"
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão: " + network_name)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Anotações
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Classe Verdadeira")
    plt.xlabel("Classe Predita")
    plt.savefig(BASE_PATH + f'confuse_matrix_{network_name}.png')
    plt.tight_layout()
    

def plot_roc_curve(y_true, y_pred_probs, class_labels, network_name, multiclass=False):
    """
    Plota a curva ROC para problemas binários ou multiclasse.
    """

    BASE_PATH = "metrics/"
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)


    if multiclass:
        y_pred_probs = 1 - y_pred_probs  # usar a probabilidade da classe positiva (CIN)
    else:
        pass

    n_classes = len(class_labels)
    plt.figure(figsize=(10, 8))

    # Se for binário
    if n_classes == 2:
        if y_pred_probs.ndim > 1:
            y_pred_probs = y_pred_probs[:, 0]  # Só pegar a primeira coluna
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            color="blue",
            lw=2,
            label=f"Classe {class_labels[1]} (AUC = {roc_auc:.2f})",
        )

    else:
        # Multiclasse (como estava)
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        if y_pred_probs.shape[1] != n_classes:
            raise ValueError(
                f"Dimensão de y_pred_probs ({y_pred_probs.shape[1]}) não corresponde ao número de classes ({n_classes})."
            )

        colors = cycle(["blue", "green", "red", "purple", "orange", "brown"])
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i, color in zip(range(n_classes), colors):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f"Classe {class_labels[i]} (AUC = {roc_auc[i]:.2f})",
            )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos", fontsize=14)
    plt.ylabel("Taxa de Verdadeiros Positivos", fontsize=14)
    plt.title(f"Curva ROC - {network_name}", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(BASE_PATH + f'roc_curve_{network_name}.png')
    plt.show()
    plt.tight_layout()
