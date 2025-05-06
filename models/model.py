# model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Resizing
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.layers import Multiply, Reshape, GlobalMaxPooling2D
from tensorflow.keras.regularizers import l2


class SpatialAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')
        super(SpatialAttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        # Average pooling along the channel axis
        avg_pool = tf.keras.backend.mean(inputs, axis=-1, keepdims=True)
        # Max pooling along the channel axis
        max_pool = tf.keras.backend.max(inputs, axis=-1, keepdims=True)
        # Concatenate
        concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        # Apply conv layer
        attention = self.conv(concat)
        # Multiply with original input
        return tf.keras.layers.Multiply()([inputs, attention])
    


class Models:
    def __init__(self):
        pass

    def get_strategy(self, list_devices=None):
        """
        Get the distribution strategy for TensorFlow.
        """
        # Verifica se há GPU disponível
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            strategy = tf.distribute.MirroredStrategy(devices=list_devices)  # Usa múltiplas GPUs se disponíveis
            print(f"Treinando com {len(gpus)} GPU(s)")
        else:
            strategy = tf.distribute.get_strategy()  # Treina na CPU
            print("Treinando com CPU")
        return strategy
    
        
    def get_generators(self, path_images, image_size, batch_size, augmentation=False):
        """
        Get the data generators for training and validation.
        """
        train_generator = None
        val_generator   = None
        test_generator  = None

        if augmentation:

            datagen_train = ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode="nearest",
            )
            
            train_generator = datagen_train.flow_from_directory(
                directory=path_images + "/train/",
                target_size=image_size,
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=True,  
            )
            val_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
                directory=path_images + "/val/",
                target_size=image_size,
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False,
            )
            test_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
                directory=path_images + "/test/",
                target_size=image_size,
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False,
            )
        # If augmentation is not used, just rescale the images
        else:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

            train_generator = datagen.flow_from_directory(
                directory=path_images + "/train/",
                target_size=image_size,
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=True,
            )
            # Validation generator
            val_generator = datagen.flow_from_directory(
                directory=path_images + "/val/",
                target_size=image_size,
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False,
            )
            test_generator = datagen.flow_from_directory(
                directory=path_images + "/test/",
                target_size=image_size,
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False,
            )

        return train_generator, val_generator, test_generator
    
    
    def spatial_attention(self, input_tensor):
        """Spatial attention mechanism for VGG's larger feature maps."""
        avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)
        attention = tf.keras.activations.sigmoid(tf.concat([avg_pool, max_pool], axis=-1))
        return Multiply()([input_tensor, attention])


    def channel_attention(self, input_tensor):
        """Módulo de Attention para focar em regiões relevantes da imagem."""
        channels = input_tensor.shape[-1]
        shared_layer = Dense(channels // 8, activation='relu')
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        max_pool = GlobalMaxPooling2D()(input_tensor)
        avg_out = shared_layer(avg_pool)
        max_out = shared_layer(max_pool)
        attention = Dense(channels, activation='sigmoid')(avg_out + max_out)
        return Multiply()([input_tensor, Reshape((1, 1, channels))(attention)])
    

    def depthwise_attention(self, input_tensor):
        """Lightweight attention mechanism optimized for MobileNet's depthwise convolutions."""
        # Depthwise squeeze-excitation
        channels = input_tensor.shape[-1]
        squeeze = GlobalAveragePooling2D()(input_tensor)
        excitation = Dense(channels//4, activation='relu')(squeeze)
        excitation = Dense(channels, activation='sigmoid')(excitation)
        return Multiply()([input_tensor, Reshape((1, 1, channels))(excitation)])


    def create_mobilenetv2_model(self, pretrained=True, num_classes=2, img_size=(224, 224)):
        """
        Create an improved MobileNetV2 model with:
        - ImageNet pretrained weights (optional)
        - Lightweight depthwise attention
        - Advanced regularization (Dropout + L2 + BatchNorm)
        - Targeted fine-tuning (last 15 inverted residual blocks)
        - Optimized classifier head
        """
        # Base model
        base_model = MobileNetV2(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=(*img_size, 3),
            alpha=1.0  # Using standard width multiplier
        )
        
        # Fine-tuning: unfreeze last 15 blocks (about 30 layers)
        base_model.trainable = False
        for layer in base_model.layers[-30:]:  # MobileNetV2 has about 155 layers total
            layer.trainable = True
        
        # Add attention + custom layers
        x = self.depthwise_attention(base_model.output)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)  # Smaller dense layer
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)  # Moderate dropout
        output_layer = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output_layer)
        return model


    def create_mnasnet_model(self, pretrained=True, num_classes=2, img_size=(224, 224)):
        """
        Cria um modelo NASNetMobile aprimorado com:
        - Pré-treinamento em ImageNet (opcional)
        - Mecanismo de Attention
        - Regularização (Dropout + L2 + BatchNorm)
        - Fine-tuning direcionado (últimas 10 camadas)
        """
        # Base do modelo
        base_model = NASNetMobile(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=(*img_size, 3)
        )
        
        # Fine-tuning: descongelar últimas 10 camadas
        base_model.trainable = False
        for layer in base_model.layers[-10:]:
            layer.trainable = True
        
        # Adicionar Attention + Camadas Personalizadas
        x = self.channel_attention(base_model.output)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)
        output_layer = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output_layer)
        return model
    

    def create_alexnet_model(self, num_classes=2, img_size=(224, 224)):
        """
        Create an improved AlexNet model with:
        - Modern regularization techniques (BatchNorm, L2)
        - Enhanced feature extraction
        - Optimized dropout rates
        - Spatial attention mechanism
        - Adaptive input sizing
        """
        model = Sequential([
            # Input preprocessing
            Resizing(227, 227),  # Original AlexNet size
            layers.Rescaling(1./255),
            
            # Enhanced Conv Block 1 with spatial attention
            Conv2D(96, (11, 11), strides=4, activation='relu', padding='valid',
                kernel_regularizer=l2(1e-4), input_shape=(227, 227, 3)),
            BatchNormalization(),
            MaxPooling2D((3, 3), strides=2),
            Dropout(0.2),  # Lower dropout in early layers
            
            # Conv Block 2 with improved capacity
            Conv2D(256, (5, 5), padding='same', activation='relu',
                kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            MaxPooling2D((3, 3), strides=2),
            Dropout(0.3),
            
            # Conv Blocks 3-5 with feature refinement
            Conv2D(384, (3, 3), padding='same', activation='relu',
                kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            
            Conv2D(384, (3, 3), padding='same', activation='relu',
                kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            
            Conv2D(256, (3, 3), padding='same', activation='relu',
                kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            MaxPooling2D((3, 3), strides=2),
            Dropout(0.4),
            
            # Classifier with modern improvements
            Flatten(),
            Dense(4096, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(0.6),  # Higher dropout in dense layers
            
            Dense(2048, activation='relu', kernel_regularizer=l2(1e-4)),  # Additional layer
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(num_classes, activation='softmax')
        ])
        
        return model
    

    def create_shufflenet_model(self, pretrained=True, num_classes=2, img_size=(224, 224)):
        """
        Create an improved ShuffleNetV2 model with:
        - ImageNet pretrained weights (optional)
        - Channel attention mechanism
        - Advanced regularization (Dropout + L2 + BatchNorm)
        - Targeted fine-tuning (last 15 layers unfrozen)
        """
        # Base model (MobileNetV2 is more efficient than original ShuffleNet)
        base_model = MobileNetV2(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=(*img_size, 3)
        )
        
        # Fine-tuning: unfreeze last 15 layers (ShuffleNet has fewer layers than Inception)
        base_model.trainable = False
        for layer in base_model.layers[-15:]:
            layer.trainable = True
        
        # Add attention + custom layers
        x = self.channel_attention(base_model.output)
        x = GlobalAveragePooling2D()(x)
        x = Dense(384, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)  # Reduced from 512 to match ShuffleNet's smaller capacity
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)  # Slightly less dropout than larger models
        output_layer = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output_layer)
        return model

    
    def create_resnet50_model(self, pretrained=True, num_classes=2, img_size=(224, 224)):
        """
        Create an improved ResNet50 model with:
        - ImageNet pretrained weights (optional)
        - Channel attention mechanism
        - Advanced regularization (Dropout + L2 + BatchNorm)
        - Targeted fine-tuning (last 20 layers unfrozen)
        - Intermediate dense layer with 512 units
        """
        # Base model
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=(*img_size, 3)
        )
        
        # Fine-tuning: unfreeze last 20 layers (ResNet50 has more layers than ShuffleNet)
        base_model.trainable = False
        for layer in base_model.layers[-20:]:
            layer.trainable = True
        
        # Add attention + custom layers
        x = self.channel_attention(base_model.output)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)  # Higher dropout for ResNet's capacity
        output_layer = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output_layer)
        return model
    

    def create_densenet_model(self, pretrained=True, num_classes=2, img_size=(224, 224)):
        """
        Create an improved DenseNet121 model with:
        - ImageNet pretrained weights (optional)
        - Channel attention mechanism
        - Advanced regularization (Dropout + L2 + BatchNorm)
        - Targeted fine-tuning (last 30 dense blocks)
        - Intermediate dense layer with 1024 units
        """
        # Base model
        base_model = tf.keras.applications.DenseNet121(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=(*img_size, 3)
        )
        
        # Fine-tuning: unfreeze last 30 layers (DenseNet has many interconnected layers)
        base_model.trainable = False
        for layer in base_model.layers[-30:]:
            layer.trainable = True
        
        # Add attention + custom layers
        x = self.channel_attention(base_model.output)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.6)(x)  # Slightly lower dropout due to DenseNet's natural regularization
        output_layer = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output_layer)
        return model
    

    def create_vgg16_model(self, pretrained=True, num_classes=2, img_size=(224, 224)):
        """
        Create an improved VGG16 model with:
        - ImageNet pretrained weights (optional)
        - Spatial attention mechanism (better for VGG than channel attention)
        - Advanced regularization (Dropout + L2 + BatchNorm)
        - Targeted fine-tuning (last 4 conv blocks)
        - Expanded classifier head
        """
        # Base model
        base_model = tf.keras.applications.VGG16(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=(*img_size, 3)
        )        
        # Fine-tuning: unfreeze last 4 conv blocks (blocks 3-5)
        base_model.trainable = False
        for layer in base_model.layers:
            if layer.name.startswith(('block3', 'block4', 'block5')):
                layer.trainable = True
        
        # Add attention + custom layers
        x = SpatialAttentionLayer()(base_model.output)  # Using the custom layer here
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)  # Higher dropout for VGG's large dense layers
        x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output_layer)
        return model
        

    def create_inception_model(self, pretrained=True, num_classes=2, img_size=(224, 224)):
        """
        Cria um modelo InceptionV3 aprimorado com:
        - Pré-treinamento em ImageNet (opcional)
        - Mecanismo de Attention
        - Regularização (Dropout + L2)
        - Fine-tuning direcionado
        - Batch Normalization
        """
        # Base do modelo
        base_model = tf.keras.applications.InceptionV3(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=(*img_size, 3)
        )
        
        # Fine-tuning: descongelar últimos 15 layers (Inception tem mais camadas que EfficientNet)
        base_model.trainable = False
        for layer in base_model.layers[-15:]:
            layer.trainable = True
        
        # Adicionar Attention + Camadas Personalizadas
        x = self.channel_attention(base_model.output)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)
        output_layer = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output_layer)
        return model


    def create_efficientnet_model(self, pretrained=True, num_classes=2, img_size=(224, 224)):
        """
        Create an EfficientNet model.
        Args:
            pretrained (bool): If True, use pre-trained weights.
            num_classes (int): Number of output classes.
            img_size (tuple): Input image size.
        Returns:
            model (tf.keras.Model): Keras model instance.
        """
        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=(*img_size, 3),
        )
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        output_layer = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=output_layer)
        return model
    

    def create_efficientnetb4_model(self, pretrained=True, num_classes=2, img_size=(224, 224)):
        """
        Cria um modelo EfficientNetB4 com:
        - Pré-treinamento em ImageNet.
        - Camadas de Attention.
        - Regularização (Dropout + L2).
        - Fine-tuning direcionado.
        """
        # Base do modelo (EfficientNetB4)
        base_model = tf.keras.applications.EfficientNetB4(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=(*img_size, 3)
        )
        
        # Congelar camadas iniciais e descongelar as últimas 10
        base_model.trainable = False
        for layer in base_model.layers[-10:]:
            layer.trainable = True
        
        # Adicionar Attention + Camadas Personalizadas
        x = self.channel_attention(base_model.output)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)
        output_layer = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output_layer)
        return model
