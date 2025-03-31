import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

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

from itertools import cycle


def evaluate_model(model_path, test_generator):
    """
    Avalia o modelo carregado no conjunto de teste.
    
    Args:
        model_path (str): Caminho para o modelo salvo.
        test_generator (ImageDataGenerator): Gerador de dados de teste.

    Returns:
        y_true, y_pred: Classes verdadeiras e preditas.
    """
    # Carregar modelo salvo
    model = load_model(model_path)

    # Obter as classes verdadeiras e as predições
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    return y_true, y_pred




# Função para calcular e exibir métricas
def calculate_metrics(y_true, y_pred, class_names):
    """
    Calcula e exibe métricas de avaliação.
    
    Args:
        y_true (array): Classes verdadeiras.
        y_pred (array): Classes preditas.
        class_names (list): Nomes das classes.
    """
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")



    

