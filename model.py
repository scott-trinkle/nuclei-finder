import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate)


def conv_block(x, num_filters, do=0.1, activation='elu'):
    x = Conv2D(num_filters, (3, 3), activation=activation,
               kernel_initializer='he_normal', padding='same')(x)
    x = Dropout(do)(x)
    x = Conv2D(num_filters, (3, 3), activation=activation,
               kernel_initializer='he_normal', padding='same')(x)
    return x


def build_model(input_size=(128, 128, 3), num_filters=None, dropouts=None):
    # Inspired by:
    # https://idiotdeveloper.com/polyp-segmentation-using-unet-in-tensorflow-2/
    if num_filters is None:
        num_filters = [16, 32, 64, 128, 256]
    if dropouts is None:
        dropouts = [0.1, 0.1, 0.2, 0.2, 0.3]

    # Input layer: scale to 0-1
    inputs = Input(input_size)
    x = Lambda(lambda x: x/255)(inputs)

    skip_x = []

    # Encoder
    for f, do in zip(num_filters[:-1], dropouts[:-1]):
        x = conv_block(x, f, do)
        skip_x.append(x)
        x = MaxPooling2D((2, 2))(x)

    # Bridge
    x = conv_block(x, num_filters[-1], dropouts[-1])

    num_filters.reverse()
    dropouts.reverse()
    skip_x.reverse()

    # Decoder
    for i, (f, do) in enumerate(zip(num_filters[1:], dropouts[1:])):
        x = Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(x)
        xs = skip_x[i]
        x = concatenate([x, xs])
        x = conv_block(x, f, do)

    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile('adam', loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.MeanIoU(2)])
    return model


def get_model():
    model = build_model()
    model.load_weights('nuclei-segment.h5')
    return model
