import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import os

def evaluate_model(model, test_loader, device, multiclass=False):
    model.eval()
    model.to(device)

    y_true = []
    y_pred_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            y_pred_probs.extend(probs)
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    calculate_metrics(y_true, y_pred, model.__class__.__name__)
    plot_confusion_matrix(y_true, y_pred, model.__class__.__name__)
    plot_roc_curve(y_true, y_pred_probs, model.__class__.__name__, multiclass=multiclass)


def calculate_metrics(y_true, y_pred, model_name):
    print(f"Classification Report: {model_name}")
    class_names = np.unique(y_true).astype(str)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão: {model_name}")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Classe Verdadeira")
    plt.xlabel("Classe Predita")
    plt.tight_layout()
    plt.savefig(f"metrics/confusion_matrix_{model_name}.png")
    plt.close()


def plot_roc_curve(y_true, y_pred_probs, model_name, multiclass=False):
    from sklearn.preprocessing import label_binarize
    from itertools import cycle

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
    plt.title(f"Curva ROC - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(f"metrics/roc_curve_{model_name}.png")
    plt.close()
