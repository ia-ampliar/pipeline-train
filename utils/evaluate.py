import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils.visualize import plot_confusion_matrix
from utils.visualize import plot_roc_curve
from sklearn.metrics import ConfusionMatrixDisplay

import os

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from itertools import cycle

def evaluate_model(test_generator, MODEL_NAME, BASE_PATH, multiclass=False):
    """Executa a avaliação de um modelo existente"""
    print("\nIniciando avaliação do modelo...")
    

    # Caminho do modelo salvo
    model_path = BASE_PATH + "/models" + f"{MODEL_NAME}_model_224x224.keras"
    if not os.path.exists(model_path):
        model_path = input("Por favor, insira o caminho completo do modelo que deseja avaliar: ")
        
    # Carregar modelo
    model = load_model(model_path, compile=False)
    class_indices = test_generator.class_indices  # Índices das classes
    class_labels = list(class_indices.keys())  # Nomes das classes

    # Avaliação no conjunto de teste
    y_true = test_generator.classes  # Classes verdadeiras
    y_pred_probs = model.predict(test_generator)  # Probabilidades preditas
    y_pred = np.argmax(y_pred_probs, axis=1)  # Classes preditas

    # Calcular métricas
    calculate_metrics(y_true, y_pred, class_labels, MODEL_NAME)

    # Plotar matriz de confusão
    plot_confusion_matrix(y_true, y_pred, class_labels, MODEL_NAME)
    
    # Plotar curva ROC
    plot_roc_curve(y_true, y_pred_probs, class_labels, MODEL_NAME, multiclass=False)

    input("\nPressione Enter para voltar ao menu principal...")





# Função para calcular e exibir métricas
def calculate_metrics(y_true, y_pred, class_names, network_name):
    """
    Calcula e exibe métricas de avaliação.
    
    Args:
        y_true (array): Classes verdadeiras.
        y_pred (array): Classes preditas.
        class_names (list): Nomes das classes.
    """
    BASE_PATH = "metrics/"
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    print(f"Classification Report: {network_name}")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


    # Salvar o classification report em uma figura
    plt.figure(figsize=(12, 5))
    plt.text(0.1, 0.1, report, fontfamily='monospace', fontsize=10)
    plt.axis('off')
    plt.savefig(BASE_PATH + f"classification_report{network_name}.png", bbox_inches='tight', dpi=300)
    plt.close()



    

