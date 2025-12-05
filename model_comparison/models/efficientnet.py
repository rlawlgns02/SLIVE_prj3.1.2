"""
EfficientNet 모델 - Efficient Neural Network 구조
손 랜드마크 데이터를 위한 1D EfficientNet 구현
MBConv (Mobile Inverted Bottleneck Convolution) 블록 사용
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class EfficientNetModel:
    """EfficientNet 모델 (1D MBConv blocks for hand landmarks)"""

    def __init__(self, num_classes=74, input_shape=(21, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.name = "EfficientNet"

    def se_block(self, x, filters, reduction=4):
        """Squeeze-and-Excitation Block"""
        se = layers.GlobalAveragePooling1D()(x)
        se = layers.Reshape((1, filters))(se)
        se = layers.Dense(filters // reduction, activation='relu',
                         kernel_initializer='he_normal')(se)
        se = layers.Dense(filters, activation='sigmoid',
                         kernel_initializer='he_normal')(se)
        return layers.Multiply()([x, se])

    def mbconv_block(self, x, filters, kernel_size=3, stride=1, expand_ratio=4):
        """Mobile Inverted Bottleneck Convolution Block"""
        input_filters = x.shape[-1]
        expanded_filters = input_filters * expand_ratio

        # Expansion phase
        if expand_ratio != 1:
            x_expanded = layers.Conv1D(expanded_filters, 1, padding='same',
                                      kernel_initializer='he_normal')(x)
            x_expanded = layers.BatchNormalization()(x_expanded)
            x_expanded = layers.Lambda(lambda x: x * tf.nn.sigmoid(x))(x_expanded)
        else:
            x_expanded = x

        # Depthwise conv
        x_dw = layers.DepthwiseConv1D(kernel_size, strides=stride, padding='same',
                                      depthwise_initializer='he_normal')(x_expanded)
        x_dw = layers.BatchNormalization()(x_dw)
        x_dw = layers.Lambda(lambda x: x * tf.nn.sigmoid(x))(x_dw)

        # Squeeze-and-Excitation
        x_se = self.se_block(x_dw, expanded_filters)

        # Output phase
        x_out = layers.Conv1D(filters, 1, padding='same',
                             kernel_initializer='he_normal')(x_se)
        x_out = layers.BatchNormalization()(x_out)

        # Skip connection
        if stride == 1 and input_filters == filters:
            x_out = layers.Add()([x, x_out])

        return x_out

    def build_model(self):
        """EfficientNet 모델 구조 생성"""
        inputs = layers.Input(shape=self.input_shape)

        # Stem
        x = layers.Conv1D(32, 3, strides=2, padding='same',
                         kernel_initializer='he_normal')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Lambda(lambda x: x * tf.nn.sigmoid(x))(x)

        # MBConv blocks
        # Stage 1
        x = self.mbconv_block(x, filters=16, kernel_size=3, stride=1, expand_ratio=1)

        # Stage 2
        x = self.mbconv_block(x, filters=24, kernel_size=3, stride=2, expand_ratio=6)
        x = self.mbconv_block(x, filters=24, kernel_size=3, stride=1, expand_ratio=6)

        # Stage 3
        x = self.mbconv_block(x, filters=40, kernel_size=5, stride=2, expand_ratio=6)
        x = self.mbconv_block(x, filters=40, kernel_size=5, stride=1, expand_ratio=6)

        # Stage 4
        x = self.mbconv_block(x, filters=80, kernel_size=3, stride=1, expand_ratio=6)
        x = self.mbconv_block(x, filters=80, kernel_size=3, stride=1, expand_ratio=6)

        # Stage 5
        x = self.mbconv_block(x, filters=112, kernel_size=5, stride=1, expand_ratio=6)
        x = self.mbconv_block(x, filters=112, kernel_size=5, stride=1, expand_ratio=6)

        # Head
        x = layers.Conv1D(320, 1, padding='same',
                         kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Lambda(lambda x: x * tf.nn.sigmoid(x))(x)

        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Dense layers
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Lambda(lambda x: x * tf.nn.sigmoid(x))(x)
        x = layers.Dropout(0.3)(x)

        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model

    def compile_model(self, learning_rate=0.001):
        """모델 컴파일"""
        if self.model is None:
            self.build_model()

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def get_model(self):
        """모델 반환"""
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model

    def summary(self):
        """모델 요약"""
        if self.model is None:
            self.build_model()
        return self.model.summary()

    def count_parameters(self):
        """파라미터 수 계산"""
        if self.model is None:
            self.build_model()
        return self.model.count_params()
