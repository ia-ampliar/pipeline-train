import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import os

from models.model import Models




class ModelEnsemble:
    def __init__(self, model_paths):
        """
        Inicializa o ensemble com os caminhos dos modelos treinados
        :param model_paths: Lista de dicionários com informações dos modelos
        Exemplo: [{'name': 'EfficientNetB4', 'path': 'path/to/model.h5', 'type': 'keras'},
                 {'name': 'ResNet50', 'path': 'path/to/model.keras', 'type': 'keras'}]
        """
        self.models = []
        self.model_names = []
        self.model_instance = Models()

        
        for model_info in model_paths:
            print(f"Carregando modelo {model_info['name']}...")
            if model_info['type'] == 'keras':
                model = load_model(model_info['path'])
            elif model_info['type'] == 'weights':
                model = self.build_model_architecture(model_info['name'])
                model.load_weights(model_info['path'])
            else:
                raise ValueError("Tipo de modelo desconhecido. Use 'keras' ou 'weights'")
                
            self.models.append(model)
            self.model_names.append(model_info['name'])
        
        print("Ensemble criado com sucesso com os seguintes modelos:")
        print("\n".join(self.model_names))
    
    def build_model_architecture(self, model_name):
        """Constroi a arquitetura do modelo se carregando apenas os pesos"""
        if model_name == 'efficientnetv2':
            return self.model_instance.create_efficientnetb4_model()
        elif model_name == 'resnet50':
            return self.model_instance.create_resnet50_model()
        elif model_name == 'densenet':
            return self.model_instance.create_densenet_model()
        elif model_name == 'vgg16':
            return self.model_instance.create_vgg16_model()
        elif model_name == 'mobilenetv2':
            return self.model_instance.create_mobilenetv2_model()
        elif model_name == 'alexnet':
            return self.model_instance.create_alexnet_model()
        elif model_name == 'inception':
            return self.model_instance.create_inception_model()
        elif model_name == 'shufflenet':
            return self.model_instance.create_shufflenet_model()
        elif model_name == 'mnasnet':
            return self.model_instance.create_mnasnet_model()
        else:
            raise ValueError(f"Arquitetura desconhecida: {model_name}")
    
    def predict(self, x):
        """
        Faz predições usando votação por maioria
        :param x: Dados de entrada (batch de imagens)
        :return: Predições finais e matriz de votação
        """
        all_predictions = []
        
        # Coleta predições de todos os modelos
        for model in self.models:
            preds = model.predict(x, verbose=0)
            classes = np.argmax(preds, axis=1)  # Converte probabilidades em classes
            all_predictions.append(classes)
        
        # Transpõe para ter predições por amostra
        sample_predictions = np.array(all_predictions).T
        
        # Aplica votação por maioria
        final_predictions = []
        voting_matrix = []
        
        for sample in sample_predictions:
            counts = Counter(sample)
            majority_vote = counts.most_common(1)[0][0]
            final_predictions.append(majority_vote)
            
            # Registra como cada modelo votou
            voting_matrix.append([(model_name, int(pred)) 
                                for model_name, pred in zip(self.model_names, sample)])
        
        return np.array(final_predictions), voting_matrix
    
    def evaluate_ensemble(self, test_data, test_labels):
        """
        Avalia o desempenho do ensemble no conjunto de teste
        :return: Métricas de desempenho
        """
        final_predictions, _ = self.predict(test_data)
        
        accuracy = np.mean(final_predictions == test_labels)
        report = classification_report(test_labels, final_predictions, output_dict=True)
        cm = confusion_matrix(test_labels, final_predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }

# Exemplo de uso:
if __name__ == "__main__":
    BASE_PATH  = "/mnt/efs-tcga/HEAL_Workspace/macenko_datas/weights/checkpoint/"
    # 1. Definir caminhos dos modelos treinados
    MODEL_PATHS = [
        {'name': 'EfficientNetV2',  'path': BASE_PATH + 'efficientnetv2_epoch_11.keras',    'type': 'keras'},
        {'name': 'ResNet50',        'path': BASE_PATH + 'resnet50_epoch_07.keras',          'type': 'keras'},
        {'name': 'DenseNet',        'path': BASE_PATH + 'densenet_epoch_04.keras',          'type': 'keras'},
        {'name': 'MobileNetV2',     'path': BASE_PATH + 'mobileNet_checkpoint.keras',       'type': 'keras'},
        {'name': 'AlexNet',         'path': BASE_PATH + 'alexnet_epoch_05.keras',           'type': 'keras'},
        {'name': 'InceptionNet',    'path': BASE_PATH + 'inception_checkpoint.keras',       'type': 'keras'},
        {'name': 'MnasNet',         'path': BASE_PATH + 'mnasNet_checkpoint.keras',         'type': 'keras'},
        {'name': 'ShuffleNet',      'path': BASE_PATH + 'shufflenet_epoch_07.keras',        'type': 'keras'},
        {'name': 'VggNet',          'path': BASE_PATH + 'vgg16_epoch_02.keras',             'type': 'keras'},
    ]
    
    # 2. Carregar o ensemble
    ensemble = ModelEnsemble(MODEL_PATHS)
    
    # 3. Exemplo de predição (substitua com seus dados reais)
    test_images = input("Insira o caminho para suas imagens de teste: ")
    test_images = np.load(test_images)  # Carregue suas imagens de teste aqui

    predictions, voting_matrix = ensemble.predict(test_images)
    
    print("\nPredições do ensemble:")
    for i, (pred, votes) in enumerate(zip(predictions, voting_matrix)):
        print(f"\nAmostra {i+1}:")
        print(f"Predição final: {pred}")
        print("Votos individuais:")
        for model, vote in votes:
            print(f"- {model}: {vote}")
    
    # 4. Avaliação completa (se tiver labels)
    # metrics = ensemble.evaluate_ensemble(test_images, test_labels)
    # print(f"\nAcurácia do ensemble: {metrics['accuracy']:.2%}")