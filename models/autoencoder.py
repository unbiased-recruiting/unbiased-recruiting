from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ast import literal_eval

import os
import numpy as np
import tensorflow as tf
import pandas as pd

from operator import itemgetter

K = tf.keras
# ====================== Preliminary settings =======================

# fix random seed for reproducibility
np.random.seed(34)
tf.random.set_random_seed(35)
tf.logging.set_verbosity(tf.logging.INFO)

#Flags definition
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 50, '')
flags.DEFINE_float('learning_rate', 0.001, '')
flags.DEFINE_integer('info_freq', 100, '')
flags.DEFINE_integer('info_valid_freq', 50, '')
flags.DEFINE_string('data_dir', '../data', '')
flags.DEFINE_float('classifier_learning_rate', 0.001, '')
flags.DEFINE_float('beta', 10, '')

# ====================== Loading settings =======================

print('Loading Data...')
#Importing train and validation datasets
data_dir = '../data/'
df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
df_val = pd.read_csv(os.path.join(data_dir, "val.csv"))
df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))

data = [df_train, df_val, df_test]
print('Preprocessing Data...')
#Converting TXT column from str to array
for df in data:
    df.loc[:, 'TXT'] = df.loc[:, 'TXT'].apply(lambda x: literal_eval(x))
    df.loc[:, 'TXT'] = df.loc[:, 'TXT'].apply(lambda x: [float(w) for w in x]) #Normalizing tokens (10 000 is the maximum and 0 min)

X_train = np.array(df_train['TXT'].tolist(), dtype = np.float32)
X_val = np.array(df_val['TXT'].tolist(), dtype = np.float32)
X_test = np.array(df_test['TXT'].tolist(), dtype = np.float32)


y_train = df_train[['GENRE_1.0', 'GENRE_2.0']].values
y_val = df_val[['GENRE_1.0', 'GENRE_2.0']].values
y_test = df_test[['GENRE_1.0', 'GENRE_2.0']].values


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

print('Building Iterators...')
train_iterator = make_iterator(X_train, y_train,
    batch_size=FLAGS.batch_size, shuffle_and_repeat=True)
valid_iterator = make_iterator(X_val, y_val,
    batch_size=FLAGS.batch_size)
test_iterator = make_iterator(X_test, y_test,
    batch_size=FLAGS.batch_size)

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

print('Building Models...')
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
    loss = tf.losses.mean_squared_error(input_cv, logits) - Beta*clf_loss
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train = optimizer.minimize(loss) 
    return train, loss

def clf_step(encoded_input, label, dataset = 'train'):
    logits = gender_clf(encoded_input)
    prediction = tf.argmax(logits,1)
    truth_label = tf.argmax(label,1)
    clf_optimizer = tf.train.AdamOptimizer(FLAGS.classifier_learning_rate)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=logits)
    train = clf_optimizer.minimize(loss)
    if dataset != 'train' :
        name=str(dataset+'_accuracy')
        accuracy, accuracy_op = tf.metrics.accuracy(truth_label, prediction, name=name)
        return train, loss, accuracy, accuracy_op
    else:
        accuracy = tf.reduce_mean(tf.cast(tf.equal(truth_label, prediction), tf.float32))
        return train, loss, accuracy

# Create training operations
train_features = train_iterator.get_next()
CVs, labels = itemgetter('CV', 'label')(train_features)

clf_train, clf_loss, clf_accuracy = clf_step(encoder(CVs), labels)
autoencoder_train, autoencoder_loss = autoencoder_step(CVs,clf_loss, FLAGS.beta)

# ====================== Training Model =======================

# Create validation operations
val_features = valid_iterator.get_next()
val_CVs, val_labels = itemgetter('CV', 'label')(val_features)

val_clf, val_clf_loss, clf_valid_accuracy, clf_valid_accuracy_op = clf_step(encoder(val_CVs), val_labels, 'valid')
val_autoencoder_train, val_autoencoder_loss = autoencoder_step(val_CVs,val_clf_loss, FLAGS.beta)

valid_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='valid_accuracy') # to measure validation accuracy during training
valid_running_vars_initializer = tf.variables_initializer(var_list=valid_running_vars)

# Create Test Operations
test_features = test_iterator.get_next()
test_CVs, test_labels = itemgetter('CV', 'label')(test_features)

test_clf, test_clf_loss, test_clf_accuracy, test_clf_accuracy_op = clf_step(encoder(test_CVs), test_labels, 'test')
test_autoencoder_train, test_autoencoder_loss = autoencoder_step(test_CVs,test_clf_loss, FLAGS.beta)

test_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='test_accuracy') # to measure validation accuracy during training
test_running_vars_initializer = tf.variables_initializer(var_list=test_running_vars)


# ====================== Defining training operations =======================
print('Training Models...')
#Training model
init=tf.global_variables_initializer()
num_epoch= 1000

adversarial_losses = []
clf_accuracies = []
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoch):
            if epoch % FLAGS.info_freq == 0:
                loss_value = sess.run(autoencoder_loss)
                adversarial_losses.append(loss_value)
                print("epoch: {} , autoencoder loss during training : {}".format(epoch, loss_value))

            else:
                sess.run(clf_train)
                if (epoch - 1)%FLAGS.info_freq == 0:
                    accuracy = sess.run(clf_accuracy)
                    clf_accuracies.append(accuracy)
                    print("epoch: {} , classifier accuracy during training".format(epoch, accuracy))

                # validation
                if (epoch-1) % FLAGS.info_valid_freq == 0:
                    sess.run(valid_running_vars_initializer)  # reinitialize accuracy
                    sess.run(valid_iterator.initializer)
                    while True:
                        try:
                            valid_loss = sess.run(val_autoencoder_loss)
                            sess.run(clf_valid_accuracy_op)
                        except tf.errors.OutOfRangeError:
                            break
                    valid_accuracy_value = sess.run(clf_valid_accuracy)
                    print('epoch: {} , validation_accuracy of classifier: {}'.format(epoch, valid_accuracy_value))
                    print('validation loss of autoencoder : ', valid_loss)
    # test
    sess.run(test_running_vars_initializer)  # reinitialize accuracy
    sess.run(test_iterator.initializer)
    while True:
        try:
            test_loss = sess.run(test_autoencoder_loss)
            sess.run(test_clf_accuracy_op)
        except tf.errors.OutOfRangeError:
            break

    test_accuracy_value = sess.run(test_clf_accuracy)
    print('test loss of autoencoder : ', test_loss)
    print('test_accuracy for classifier : {}'.format(test_accuracy_value))

print('Session successfully closed !')

# ====================== Exporting model =======================

# serialize weights to HDF5
saving_path = './saved_models'
encoder.save_weights(os.path.join(saving_path, "encoder.h5"))
print("Encoder saved")
encoder.save_weights(os.path.join(saving_path, "decoder.h5"))
print("Decoder saved")

