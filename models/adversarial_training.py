import os
import numpy as np
import tensorflow as tf
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
flags.DEFINE_float('adversarial_learning_rate', 0.001, '')
flags.DEFINE_float('classifier_learning_rate', 0.001, '')
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

#Place Holders
CV_ph = K.Input(shape=(num_inputs,))
representation_ph = K.Input(shape=(1024,)) #Format of the representation

encoder=autoencoding(num_inputs)[0]
decoder=autoencoding(num_inputs)[1]
auto_encoder=autoencoding(num_inputs)[2]
gender_clf = gender_clf_model(1024)


def encoder_decoder_step(CV,label):
    with tf.GradientTape() as tape:
        representation = encoder(CV)
        estimated_gender=gender_clf(representation)
        decoded_cv=decoder(representation)

        loss_classifier=tf.losses.softmax_cross_entropy(
            onehot_labels=label, logits=estimated_gender)

        loss_autoencoder = tf.losses.mean_squared_error(CV, decoded_cv)
        adversarial_loss = loss_autoencoder-Beta*loss_classifier
        optimizer=tf.train.AdamOptimizer(FLAGS.adversarial_learning_rate)
        gradients=tape.gradient(adversarial_loss,tf.constant([encoder.trainable])+tf.constant([decoder.trainable]))
        train_op = optimizer.apply_gradients(gradients)
        return train_op, adversarial_loss, loss_autoencoder


def classifier_step(representation,label):
    estimated_gender = gender_clf(representation)
    clf_optimizer = tf.train.AdamOptimizer(FLAGS.classifier_learning_rate)
    loss_classifier = tf.losses.softmax_cross_entropy(
        onehot_labels=label, logits=estimated_gender)
    train_optimizer = clf_optimizer.minimize(loss_classifier)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(estimated_gender, label), tf.int32))
    return train_optimizer, accuracy

#Training variables
features = train_iterator.get_next()
CVs, labels = itemgetter('CV', 'label')(features)

train_encoder, adversarial_loss, autoencoder_loss = encoder_decoder_step(CVs, labels)
representation = encoder(CVs)
train_clf, clf_accuracy = classifier_step(representation,labels)


# Adversarial Training
num_epoch=10000

print("Adversarial learning ...")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(num_epoch):
        if epoch%2==0:
            sess.run(train_encoder)
        else:
            sess.run(train_clf)
        metrics = sess.run([adversarial_loss, autoencoder_loss, clf_accuracy])
        if epoch%100==0:
            print('adversarial loss: {}, \nautoencoder loss: {}, \nclassifier accuracy : {}'.format(metrics[0], metrics[1], metrics[2]))


