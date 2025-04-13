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
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Resizing
from tensorflow.keras.applications import MobileNet


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
        

    def create_mobilenetv2_model(self, pretrained=True, num_classes=2, img_size=(224, 224)):
        base_model = MobileNetV2(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=(*img_size, 3),
        )
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        output_layer = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=output_layer)
        return model
    
    def create_mobilenetv3_model(self, pretrained=True, num_classes=2, img_size=(224, 224)):
        base_model = MobileNetV2(
            weights="imagenet" if pretrained else None, 
            include_top=False, 
            input_shape=(*img_size, 3))
        base_model.trainable = False  # Congela inicialmente
        
        x = GlobalAveragePooling2D()(base_model.output)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(num_classes, activation="softmax", 
                           kernel_regularizer=regularizers.l2(0.001))(x)
        model = Model(inputs=base_model.input, outputs=output_layer)
        return model


    def create_mnasnet_model(self, pretrained=True, num_classes=2, img_size=(224, 224)):
        """
        Create a MobileNetV2 model with optional pre-trained weights.
        Args:
            pretrained (bool): If True, use pre-trained weights.
            num_classes (int): Number of output classes.
            img_size (tuple): Input image size.
        Returns:
            model (tf.keras.Model): Keras model instance.
        """
        base_model = NASNetMobile(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=(*img_size, 3),
        )
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        output_layer = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=output_layer)
        return model
    
    
    def create_alexnet(self, num_classes=2, img_size=(224, 224)):
        """
        Create an AlexNet model.
        Args:
            num_classes (int): Number of output classes.
            img_size (tuple): Input image size.
        Returns:
            model (tf.keras.Model): Keras model instance.
        """
        model = Sequential([
            Resizing(227, 227),  # Redimensiona a entrada para 227x227
            Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)),
            MaxPooling2D((3, 3), strides=2),
            Conv2D(256, (5, 5), padding='same', activation='relu'),
            MaxPooling2D((3, 3), strides=2),
            Conv2D(384, (3, 3), padding='same', activation='relu'),
            Conv2D(384, (3, 3), padding='same', activation='relu'),
            Conv2D(256, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((3, 3), strides=2),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        return model
    
    def create_shuffnet(self, pretrained=True, num_classes=2, img_size=(224, 224)):
        """
        Create a ShuffleNet model.
        Args:
            num_classes (int): Number of output classes.
            img_size (tuple): Input image size.
        Returns:
            model (tf.keras.Model): Keras model instance.
        """
        base_model = MobileNet(
            weights="imagenet" if pretrained else None,
            include_top=False,
            input_shape=(*img_size, 3),
        )
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        output_layer = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=output_layer)
        return model