import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
class CombinedBCESoftF1Loss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, consider_true_negative=True, sigmoid_is_applied_to_input=True, name="combined_bce_softf1_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.consider_true_negative = consider_true_negative
        self.sigmoid_is_applied_to_input = sigmoid_is_applied_to_input

        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.f1 = MacroSoftF1Loss(
            consider_true_negative=self.consider_true_negative,
            sigmoid_is_applied_to_input=self.sigmoid_is_applied_to_input
        )

    def call(self, y_true, y_pred):
        bce_loss = self.bce(y_true, y_pred)
        f1_loss = self.f1(y_true, y_pred)
        return self.alpha * bce_loss + (1 - self.alpha) * f1_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "consider_true_negative": self.consider_true_negative,
            "sigmoid_is_applied_to_input": self.sigmoid_is_applied_to_input
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class MacroSoftF1Loss(tf.keras.losses.Loss):
    def __init__(self, consider_true_negative=True, sigmoid_is_applied_to_input=False, name="macro_soft_f1_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.consider_true_negative = consider_true_negative
        self.sigmoid_is_applied_to_input = sigmoid_is_applied_to_input

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        if not self.sigmoid_is_applied_to_input:
            y_pred = tf.sigmoid(y_pred)

        TP = tf.reduce_sum(y_pred * y_true, axis=0)
        FP = tf.reduce_sum((1 - y_pred) * y_true, axis=0)
        FN = tf.reduce_sum(y_pred * (1 - y_true), axis=0)

        f1_class1 = 2 * TP / (2 * TP + FP + FN + 1e-8)
        loss_class1 = 1 - f1_class1

        if self.consider_true_negative:
            TN = tf.reduce_sum((1 - y_pred) * (1 - y_true), axis=0)
            f1_class0 = 2 * TN / (2 * TN + FP + FN + 1e-8)
            loss_class0 = 1 - f1_class0
            loss = (loss_class0 + loss_class1) / 2
        else:
            loss = loss_class1

        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "consider_true_negative": self.consider_true_negative,
            "sigmoid_is_applied_to_input": self.sigmoid_is_applied_to_input
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
