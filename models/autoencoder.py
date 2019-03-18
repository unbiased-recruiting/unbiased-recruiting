from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ast import literal_eval

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

from operator import itemgetter

K = tf.keras

# np.random.seed(34) # to make the results reproductible
# tf.random.set_random_seed(35) # to make the results reproductible 
# tf.logging.set_verbosity(tf.logging.INFO)

# #Flags definition
# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_integer('batch_size', 50, '')
# flags.DEFINE_float('learning_rate', 0.001, '')
# flags.DEFINE_integer('info_freq', 10, '')
# flags.DEFINE_integer('info_valid_freq', 5, '')
# flags.DEFINE_string('data_dir', '../data', '')

# #Importing train and validation datasets
# data_dir = '../data/'
# df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
# df_val = pd.read_csv(os.path.join(data_dir, "val.csv"))

# data = [df_train, df_val]

# #Converting TXT column from str to array
# for df in data:
#     df.loc[:, 'TXT'] = df.loc[:, 'TXT'].apply(lambda x: literal_eval(x))
#     df.loc[:, 'TXT'] = df.loc[:, 'TXT'].apply(lambda x: [float(w)/10000 for w in x]) #Normalizing tokens (10 000 is the maximum and 0 min)

# X_train = df_train['TXT'].tolist()
# X_val = df_val['TXT'].tolist()

# y_train = df_train[['GENRE_1.0', 'GENRE_2.0']].values
# y_val = df_val[['GENRE_1.0', 'GENRE_2.0']].values
 
# # Create three `tf.data.Iterator` objects

# def make_iterator(CVs, labels, batch_size, shuffle_and_repeat=False):
#     """function that creates a `tf.data.Iterator` object"""
#     dataset = tf.data.Dataset.from_tensor_slices((CVs, labels))
#     if shuffle_and_repeat:
#         dataset = dataset.apply(
#             tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
    
#     def parse(CV, label):
#         """function that returns cv and associated gender in a queriable format"""
#         return {'CV': CV, 'label': label}

#     dataset = dataset.apply(tf.data.experimental.map_and_batch(
#         map_func=parse, batch_size=batch_size, num_parallel_batches=8))

#     if shuffle_and_repeat:
#         return dataset.make_one_shot_iterator()
#     else:
#         return dataset.make_initializable_iterator()

# train_iterator = make_iterator(X_train, y_train,
#     batch_size=FLAGS.batch_size, shuffle_and_repeat=True)
# valid_iterator = make_iterator(X_val, y_val,
#     batch_size=FLAGS.batch_size)


# #Network constant initialisation
# num_inputs= df_train.apply(lambda x: len(x['TXT']), axis = 1).max()

#Layer initialisation
def autoencoding(num_inputs):
    input_cv = K.layers.Input(shape=(num_inputs,))

    encoded = K.layers.Dense(1024, activation='relu')(input_cv)
    decoded = K.layers.Dense(num_inputs, activation='sigmoid')(encoded)

    encoder = K.Model(input_cv, encoded)
    autoencoder=K.Model(input_cv,decoded)
    decoded_layers=autoencoder.layers[-1]
    decoder=K.Model(encoded, decoded_layers)

    
    return [encoder,decoder, autoencoder]

# # Create training operations
# features = train_iterator.get_next()
# CVs, labels = itemgetter('CV', 'label')(features)

# logits = autoencoder(CVs)
# loss= tf.losses.mean_squared_error(CVs, logits)
# optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate)
# train=optimizer.minimize(loss)

# #Training model
# init=tf.global_variables_initializer()
# num_epoch=10000

# with tf.Session() as sess:
#     sess.run(init)
#     sess.run(init)
#     for epoch in range(num_epoch):
#         sess.run(train)
#         loss_value = sess.run([loss])
#         if epoch % 100 ==0:
#             print(loss_value)
        
        