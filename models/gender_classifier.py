import os
import numpy as np
import tensorflow as tf
import pandas as pd

from operator import itemgetter

K = tf.keras


def gender_clf_model(encoding_dim):
    representation = K.Input(shape=(encoding_dim,))
    hidden_layer=K.layers.Dense(512, activation='relu')(representation)
    output_gender=K.layers.Dense(units=2)(hidden_layer)

    gender_clf = K.Model(representation, output_gender)
    
    return gender_clf
