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
# ====================== Preliminary settings =======================


np.random.seed(34) # to make the results reproductible
tf.random.set_random_seed(35) # to make the results reproductible 
tf.logging.set_verbosity(tf.logging.INFO)

#Flags definition
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 50, '')
flags.DEFINE_float('learning_rate', 0.001, '')
flags.DEFINE_integer('info_freq', 10, '')
flags.DEFINE_integer('info_valid_freq', 5, '')
flags.DEFINE_string('data_dir', '../data', '')
flags.DEFINE_float('classifier_learning_rate', 0.001, '')

# ====================== Loading settings =======================


#Importing train and validation datasets
data_dir = '../data/'
df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
df_val = pd.read_csv(os.path.join(data_dir, "val.csv"))

data = [df_train, df_val]

#Converting TXT column from str to array
for df in data:
    df.loc[:, 'TXT'] = df.loc[:, 'TXT'].apply(lambda x: literal_eval(x))
    df.loc[:, 'TXT'] = df.loc[:, 'TXT'].apply(lambda x: [float(w)/10000 for w in x]) #Normalizing tokens (10 000 is the maximum and 0 min)

X_train = np.array(df_train['TXT'].tolist(), dtype = np.float32)
X_val = np.array(df_val['TXT'].tolist(), dtype = np.float32)

print(X_train.shape)


y_train = df_train[['GENRE_1.0', 'GENRE_2.0']].values
y_val = df_val[['GENRE_1.0', 'GENRE_2.0']].values

#y_train = df_train[['GENRE_1.0', 'GENRE_2.0']].apply(lambda x : 0 if x['GENRE_1.0'] == 0 else 1, axis =1 ).values
#y_val = df_val[['GENRE_1.0', 'GENRE_2.0']].apply(lambda x : 0 if x['GENRE_1.0'] == 0 else 1, axis =1 ).values

print(y_train)
print(y_train.shape)
# Create three `tf.data.Iterator` objects

def make_iterator(CVs, labels, batch_size, shuffle_and_repeat=False):
    """function that creates a `tf.data.Iterator` object"""
    dataset = tf.data.Dataset.from_tensor_slices((CVs, labels))
    if shuffle_and_repeat:
        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
    
    def parse(CV, label):
        """function that returns cv and associated gender in a queriable format"""
        return {'CV': CV, 'label': label}

    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        map_func=parse, batch_size=batch_size, num_parallel_batches=8))

    if shuffle_and_repeat:
        return dataset.make_one_shot_iterator()
    else:
        return dataset.make_initializable_iterator()

train_iterator = make_iterator(X_train, y_train,
    batch_size=FLAGS.batch_size, shuffle_and_repeat=True)
valid_iterator = make_iterator(X_val, y_val,
    batch_size=FLAGS.batch_size)


def deduce_class(predictions, threshold=0.5):
    for i in range(len(predictions)):
        if predictions[i]<threshold:
            predictions[i]=0
        else:
            predictions[i]=1
    return predictions

# ====================== Network architecture =======================

#Network constant initialisation
num_inputs= len(df_train.loc[0, 'TXT'])

#Layer initialisation

def model_builder(input_cv,  compression_size, num_inputs):
    #autoencoder
    encoded = K.layers.Dense(compression_size, activation = 'relu')(input_cv)
    decoded = K.layers.Dense(num_inputs, activation = 'sigmoid')(encoded)
    autoencoder = K.Model(input_cv, decoded, name = "autoencoder")
    encoder = K.Model(input_cv, encoded, name = "encoder")
    encoded_input = K.layers.Input(shape = (compression_size,), name = "encoder_input")
    decoded_layer = autoencoder.layers[-1]
    decoder = K.Model(encoded_input, decoded_layer(encoded_input))

    #gender_clf
    clf = K.layers.Dense(compression_size, activation = 'relu')(encoded_input)
    outputs = K.layers.Dense(units=2)(clf)
    gender_clf = K.Model(encoded_input, outputs, name =  'clf')
    return autoencoder, encoder, decoder, gender_clf

#Autoencoder layers
input_cv = K.layers.Input(shape=(num_inputs,), name = "input_cv")

autoencoder, encoder, decoder, gender_clf = model_builder(input_cv, 1024, num_inputs)

print("encoder")
encoder.summary()
print("decoder")
decoder.summary()
print("autoencoder")
autoencoder.summary()
# ====================== Defining training operations =======================

#Defining training operations
def autoencoder_step(input_cv, clf_loss, Beta):
    logits = autoencoder(input_cv)
    loss = tf.losses.mean_squared_error(input_cv, logits) + Beta*clf_loss
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train = optimizer.minimize(loss) 
    return train, loss

def clf_step(encoded_input, label):
    logits = gender_clf(encoded_input)
    prediction = deduce_class(logits)
    clf_optimizer = tf.train.AdamOptimizer(FLAGS.classifier_learning_rate)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=logits)
    train = clf_optimizer.minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(label, prediction), tf.float32))
    return train, loss, accuracy

# Create training operations
features = train_iterator.get_next()
CVs, labels = itemgetter('CV', 'label')(features)

clf_train, clf_loss, clf_accuracy = clf_step(encoder(CVs), labels)
print("hello1")
autoencoder_train, autoencoder_loss = autoencoder_step(CVs,clf_loss, 0.0001)
print("hello2")
# ====================== Defining training operations =======================

#Training model
init=tf.global_variables_initializer()
num_epoch=10000
print("hello3")
with tf.Session() as sess:
    sess.run(init)
    print("hello4")
    for epoch in range(num_epoch):
        if epoch % 2 == 0:
            sess.run(autoencoder_train)
            print("hello5")
            if epoch %100 ==0:
                loss_value = sess.run(autoencoder_loss)
                print("autoencoder loss", loss_value)
        else:
            sess.run(clf_train)
            print("hello6")
            if (epoch - 1)%100 == 0:
                accuracy = sess.run(clf_accuracy)
                print("accuracy", accuracy)
        
        