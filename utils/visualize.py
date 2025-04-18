import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)

# Função para plotar a evolução do treinamento
def plot_training(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss Treinamento')
    plt.plot(history.history['val_loss'], label='Loss Validação')
    plt.xlabel('Epocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Evolução da Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
    plt.xlabel('Epocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.title('Evolução da Acurácia')

    plt.show()
    plt.savefig('training_history.png')



# Função para plotar a matriz de confusão
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plota a matriz de confusão.
    
    Args:
        y_true (array): Classes verdadeiras.
        y_pred (array): Classes preditas.
        class_names (list): Nomes das classes.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
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
    plt.tight_layout()
    


def plot_roc_curve(y_true, y_pred_probs, class_labels):
    """
    Plota a curva ROC para cada classe com base nas predições.

    Args:
        y_true: Rótulos reais das amostras (array).
        y_pred_probs: Probabilidades preditas para cada classe (array).
        class_labels: Lista com os nomes das classes.
    """
    # Verificar se há pelo menos duas classes
    n_classes = len(class_labels)
    if n_classes < 2:
        raise ValueError("Curva ROC requer pelo menos duas classes.")

    # Binarizar os rótulos reais
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Validar dimensões de y_pred_probs
    if y_pred_probs.shape[1] != n_classes:
        raise ValueError(
            f"Dimensão de y_pred_probs ({y_pred_probs.shape[1]}) não corresponde ao número de classes ({n_classes})."
        )

    # Configurações para plotar a curva ROC
    plt.figure(figsize=(10, 8))
    colors = cycle(["blue", "green", "red", "purple", "orange", "brown"])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Plotar a curva ROC para cada classe
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"Classe {class_labels[i]} (AUC = {roc_auc[i]:.2f})",
        )

    # Plotar a linha de referência
    plt.plot([0, 1], [0, 1], "k--", lw=2)

    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("Curva ROC - Multiclasse", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()


