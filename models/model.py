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
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.layers import Multiply, Reshape, GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, VGG16, VGG19
from tensorflow.keras.applications import DenseNet201, InceptionV3, MobileNet, NASNetMobile
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, EfficientNetB4
from keras.saving import register_keras_serializable
from tensorflow.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


@register_keras_serializable()
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


    def channel_attention(self, input_tensor):
        channels = input_tensor.shape[-1]
        shared_layer = Dense(channels // 8, activation='relu')
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        max_pool = GlobalMaxPooling2D()(input_tensor)
        avg_out = shared_layer(avg_pool)
        max_out = shared_layer(max_pool)
        attention = Dense(channels, activation='sigmoid')(avg_out + max_out)
        return Multiply()([input_tensor, Reshape((1, 1, channels))(attention)])


    def build_model(self, base_model, num_classes, mode):
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        activation = 'sigmoid' if mode else 'softmax'
        loss_fn = tf.keras.losses.BinaryCrossentropy() if mode else tf.keras.losses.CategoricalCrossentropy()
        output_layer = Dense(num_classes, activation=activation)(x)
        return Model(inputs=base_model.input, outputs=output_layer), loss_fn


    def create_resnet50_model(self, pretrained=True, num_classes=2, img_size=(224, 224), mode=False):
        base_model = ResNet50(weights='imagenet' if pretrained else None, include_top=False, input_shape=(*img_size, 3))
        return self.build_model(base_model, num_classes, mode)


    def create_resnet101_model(self, pretrained=True, num_classes=2, img_size=(224, 224), mode=False):
        base_model = ResNet101(weights='imagenet' if pretrained else None, include_top=False, input_shape=(*img_size, 3))
        return self.build_model(base_model, num_classes, mode)


    def create_resnet152_model(self, pretrained=True, num_classes=2, img_size=(224, 224), mode=False):
        base_model = ResNet152(weights='imagenet' if pretrained else None, include_top=False, input_shape=(*img_size, 3))
        return self.build_model(base_model, num_classes, mode)


    def create_vgg16_model(self, pretrained=True, num_classes=2, img_size=(224, 224), mode=False):
        base_model = VGG16(weights='imagenet' if pretrained else None, include_top=False, input_shape=(*img_size, 3))
        return self.build_model(base_model, num_classes, mode)


    def create_vgg19_model(self, pretrained=True, num_classes=2, img_size=(224, 224), mode=False):
        base_model = VGG19(weights='imagenet' if pretrained else None, include_top=False, input_shape=(*img_size, 3))
        return self.build_model(base_model, num_classes, mode)


    def create_densenet_model(self, pretrained=True, num_classes=2, img_size=(224, 224), mode=False):
        base_model = DenseNet201(weights='imagenet' if pretrained else None, include_top=False, input_shape=(*img_size, 3))
        return self.build_model(base_model, num_classes, mode)


    def create_inception_model(self, pretrained=True, num_classes=2, img_size=(224, 224), mode=False):
        base_model = InceptionV3(weights='imagenet' if pretrained else None, include_top=False, input_shape=(*img_size, 3))
        x = self.channel_attention(base_model.output)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)
        activation = 'sigmoid' if mode else 'softmax'
        loss_fn = tf.keras.losses.BinaryCrossentropy() if mode else tf.keras.losses.CategoricalCrossentropy()
        output_layer = Dense(num_classes, activation=activation)(x)
        return Model(inputs=base_model.input, outputs=output_layer), loss_fn
    

    def create_mobilenet_model(self, pretrained=True, num_classes=2, img_size=(224, 224), mode=False):
        base_model = MobileNet(weights='imagenet' if pretrained else None, include_top=False, input_shape=(*img_size, 3))
        return self.build_model(base_model, num_classes, mode)
    

    def create_mobilenetv2_model(self, pretrained=True, num_classes=2, img_size=(224, 224), mode=False):
        base_model = MobileNetV2(weights='imagenet' if pretrained else None, include_top=False, input_shape=(*img_size, 3))
        return self.build_model(base_model, num_classes, mode)


    def create_mnasnet_model(self, pretrained=True, num_classes=2, img_size=(224, 224), mode=False):
        base_model = NASNetMobile(weights='imagenet' if pretrained else None, include_top=False, input_shape=(*img_size, 3))
        return self.build_model(base_model, num_classes, mode)

    def create_alexnet(self, num_classes=2, img_size=(224, 224), mode=False):
        model = Sequential([
            tf.keras.layers.Resizing(227, 227),
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
            Dense(num_classes, activation='sigmoid' if mode else 'softmax')
        ])
        loss_fn = tf.keras.losses.BinaryCrossentropy() if mode else tf.keras.losses.CategoricalCrossentropy()
        return model, loss_fn

    def create_efficientnet_model(self, pretrained=True, num_classes=2, img_size=(224, 224), mode=False):
        base_model = EfficientNetB0(weights='imagenet' if pretrained else None, include_top=False, input_shape=(*img_size, 3))
        return self.build_model(base_model, num_classes, mode)

    def create_efficientnetb4_model(self, pretrained=True, num_classes=2, img_size=(224, 224), mode=False):
        base_model = EfficientNetB4(weights='imagenet' if pretrained else None, include_top=False, input_shape=(*img_size, 3))
        x = self.channel_attention(base_model.output)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        activation = 'sigmoid' if mode else 'softmax'
        loss_fn = tf.keras.losses.BinaryCrossentropy() if mode else tf.keras.losses.CategoricalCrossentropy()
        output_layer = Dense(num_classes, activation=activation)(x)
        return Model(inputs=base_model.input, outputs=output_layer), loss_fn
