from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

K = tf.keras


def autoencoding(num_inputs ):
    input_cv = K.Input(shape=(num_inputs,))
    encoding_input = K.Input(shape=(1024,))
    encoded = K.layers.Dense(1024, activation='relu')(input_cv)
    decoded = K.layers.Dense(num_inputs, activation='sigmoid')(encoded)

    encoder = K.Model(input_cv, encoded)
    autoencoder=K.Model(input_cv,decoded)
    decoded_layers=autoencoder.layers[-1]
    decoder=K.Model(encoding_input, decoded_layers(encoding_input))

    return [encoder,decoder, autoencoder]