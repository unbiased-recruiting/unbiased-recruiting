import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

from gender_classifier import *
from autoencoder import *
from utils import *

from operator import itemgetter

K = tf.keras
Beta=10

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

#Importing train and validation datasets
data_dir = '../data/'
df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
df_val = pd.read_csv(os.path.join(data_dir, "val.csv"))

data = [df_train, df_val]

#Converting TXT column from str to array
for df in data:
    df.loc[:, 'TXT'] = df.loc[:, 'TXT'].apply(lambda x: literal_eval(x))
    df.loc[:, 'TXT'] = df.loc[:, 'TXT'].apply(lambda x: [float(w)/10000 for w in x]) #Normalizing tokens (10 000 is the maximum and 0 min)

X_train = df_train['TXT'].tolist()
X_val = df_val['TXT'].tolist()

y_train = df_train[['GENRE_1.0', 'GENRE_2.0']].values
y_val = df_val[['GENRE_1.0', 'GENRE_2.0']].values


train_iterator = make_iterator(X_train, y_train,
    batch_size=FLAGS.batch_size, shuffle_and_repeat=True)
valid_iterator = make_iterator(X_val, y_val,
    batch_size=FLAGS.batch_size)

#Network constant initialisation
num_inputs= df_train.apply(lambda x: len(x['TXT']), axis = 1).max()

# Create training operations
features = train_iterator.get_next()
CVs, labels = itemgetter('CV', 'label')(features)

optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate)
def encoder_decoder_step(CV,label):
    with tf.GradientTape() as tape:
        encoder=autoencoding[0]
        decoder=autoencoding[1]
        autoencoder=autoencoding[2]

        representation = encoder(CV)
        estimated_gender=gender_clf_model(representation)
        decoded_cv=decoder(representation)

        loss_classifier=tf.losses.softmax_cross_entropy(
            labels=label, logits=estimated_gender)

        loss_autoencoder = tf.losses.mean_squared_error(CV,autoencoder(CV))

        loss = loss_autoencoder-Beta*loss_classifier

        gradients=tape.gradient(loss,autoencoder.trainable+decoder.trainable)

        optimizer.apply_gradients(gradients)

def classifier_step(representation,label):
    

