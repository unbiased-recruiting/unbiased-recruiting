import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

from operator import itemgetter

K = tf.keras


def gender_clf_model(representation):

    hidden_layer=K.layers.Dense(512, activation='relu')(representation)
    output_gender=K.layers.Dense(shape=(2,), activation='sigmoid')(hidden_layer)

    gender_clf = K.Model(input_encoder, output_gender)

    
    return gender_clf
