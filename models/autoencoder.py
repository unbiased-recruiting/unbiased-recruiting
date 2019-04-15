from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ast import literal_eval

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys

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
#flags.DEFINE_integer('batch_size', 50, '')
flags.DEFINE_integer('info_freq', 100, '')
flags.DEFINE_integer('info_valid_freq', 500, '')
flags.DEFINE_string('data_dir', '../data', '')
#flags.DEFINE_float('learning_rate', 0.001, '')
#flags.DEFINE_float('classifier_learning_rate', 0.001, '')
#flags.DEFINE_float('beta', 10, '')

# ====================== Loading settings =======================

print('Loading Data...')
#Importing train and validation datasets
data_dir = '../data/'
df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
df_val = pd.read_csv(os.path.join(data_dir, "val.csv"))
df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))

data = [df_train, df_val, df_test]
vocab_len = 10000 # Determined during preprocessing
print('Preprocessing Data...')
#Converting TXT column from str to array
for df in data:
    df.loc[:, 'TXT'] = df.loc[:, 'TXT'].apply(lambda x: literal_eval(x))
    df.loc[:, 'TXT'] = df.loc[:, 'TXT'].apply(lambda x: [float(w)/10000 for w in x]) #Normalizing tokens (10 000 is the maximum and 0 min)

saving_path = './saved_models/normalized/'
graphs_path = './graphs'
if not os.path.exists(saving_path):
    os.makedirs(saving_path)
if not os.path.exists(graphs_path):
    os.mkdirs(graphs_path)

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


# ====================== Network architecture =======================

#Network constant initialisation
num_inputs= len(df_train.loc[0, 'TXT'])

#Layer initialisation

def model_builder(input_cv,  compression_size, num_inputs, type = 'MLP'):
    #autoencoder
    if type == 'MLP':
        print('MLP architecture')
        encoded = K.layers.Dense(compression_size, activation = 'relu')(input_cv)
        decoded = K.layers.Dense(num_inputs, activation = 'sigmoid')(encoded)
        autoencoder = K.Model(input_cv, decoded, name = "autoencoder")
        encoder = K.Model(input_cv, encoded, name = "encoder")
        encoded_input = K.layers.Input(shape = (compression_size,), name = "encoder_input")
        decoded_layer = autoencoder.layers[-1]
        decoder = K.Model(encoded_input, decoded_layer(encoded_input))

    if type == 'LSTM':
        print('LSTM architecture')

    else:
        print('Problem occured when specifying AutoEncoder Type of architecture')
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

num_epoch= 6 #must be an even number
batch_sizes = [32]#10,, 50, 100 ]
autoencoder_learning_rates = [0.01]#,0.1,0.001,0.0001,0.00001]
clf_learning_rates = [ 0.1]#,0.0001,0.00001,0.001,0.01]
beta_values = [1]#[0.01,1,0.1,10,100,1000]


# Manual grid search
model_params = []

for batch_size in batch_sizes:
    for AE_lr in autoencoder_learning_rates:
        print('Building Iterators...')
        train_iterator = make_iterator(X_train, y_train,
                                       batch_size=batch_size, shuffle_and_repeat=True)
        valid_iterator = make_iterator(X_val, y_val,
                                       batch_size=batch_size)
        test_iterator = make_iterator(X_test, y_test,
                                      batch_size=batch_size)
        for clf_lr in clf_learning_rates:
            for beta in beta_values:
                
                hyparam_name = str('batchsize_'+str(batch_size)+'AElr_'+ str(AE_lr)+'_CLFlr_'+str(clf_lr)+'_beta_'+str(beta))
                model_params.append(hyparam_name)
                print('(batch_size, AE_learning_rate, clf_learning_rate, beta) = ({} ; {} ; {} ;{})'.format(batch_size,AE_lr, clf_lr, beta))
                
                #Defining training operations
                def autoencoder_step(input_cv, clf_loss, Beta):
                    logits = autoencoder(input_cv)
                    autoencoder_loss=tf.losses.mean_squared_error(input_cv, logits)
                    adversarial_loss = autoencoder_loss - Beta*clf_loss
                    optimizer = tf.train.AdamOptimizer(AE_lr)
                    train = optimizer.minimize(adversarial_loss)
                    return train, adversarial_loss, autoencoder_loss

                def clf_step(encoded_input, label, dataset = 'train'):
                    logits = gender_clf(encoded_input)
                    prediction = tf.argmax(logits,1)
                    truth_label = tf.argmax(label,1)
                    clf_optimizer = tf.train.AdamOptimizer(clf_lr)
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
                autoencoder_train, adversarial_loss, autoencoder_loss = autoencoder_step(CVs,clf_loss, beta)

                # ====================== Training Model =======================

                # Create validation operations
                val_features = valid_iterator.get_next()
                val_CVs, val_labels = itemgetter('CV', 'label')(val_features)

                val_clf, val_clf_loss, clf_valid_accuracy, clf_valid_accuracy_op = clf_step(encoder(val_CVs), val_labels, 'valid')
                val_autoencoder_train, val_adversarial_loss, val_autoencoder_loss = autoencoder_step(val_CVs,val_clf_loss, beta)

                valid_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='valid_accuracy') # to measure validation accuracy during training
                valid_running_vars_initializer = tf.variables_initializer(var_list=valid_running_vars)

                # Create Test Operations
                test_features = test_iterator.get_next()
                test_CVs, test_labels = itemgetter('CV', 'label')(test_features)

                test_clf, test_clf_loss, test_clf_accuracy, test_clf_accuracy_op = clf_step(encoder(test_CVs), test_labels, 'test')
                test_autoencoder_train, test_adversarial_loss, test_autoencoder_loss = autoencoder_step(test_CVs,test_clf_loss, beta)

                test_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='test_accuracy') # to measure validation accuracy during training
                test_running_vars_initializer = tf.variables_initializer(var_list=test_running_vars)

                # ====================== Defining training operations =======================
                print('Training Models...')
                #Training model
                init=tf.global_variables_initializer()

                adversarial_losses = []
                clf_accuracies = []
                autoencoder_losses = []

                even_epoch=[]
                odd_epoch=[]

                test_accuracies = []
                AE_test_losses = []
                ADV_test_losses = []
                with tf.Session() as sess:
                    sess.run(init)
                    for epoch in range(num_epoch):
                            if epoch % 2 == 0:
                                even_epoch.append(epoch)
                                autoencoder_loss_value = sess.run(autoencoder_loss)
                                autoencoder_losses.append(autoencoder_loss_value)
                                adversarial_loss_value = sess.run(adversarial_loss)
                                adversarial_losses.append(adversarial_loss_value)
                                if epoch % FLAGS.info_freq == 0:
                                    print("TRAIN epoch: {} , AE loss : {}, ADV loss : {}".format(epoch, autoencoder_loss_value, adversarial_loss_value))
   
                            else:
                                sess.run(clf_train)
                                accuracy = sess.run(clf_accuracy)
                                clf_accuracies.append(accuracy)
                                odd_epoch.append(epoch)
                                if (epoch - 1)%FLAGS.info_freq == 0:
                                    print("TRAIN epoch: {} , clf accuracy : {}".format(epoch, accuracy))

                                # validation
                            if (epoch-1) % FLAGS.info_valid_freq == 0:
                                sess.run(valid_running_vars_initializer)  # reinitialize accuracy
                                sess.run(valid_iterator.initializer)
                                while True:
                                    try:
                                        AE_valid_loss = sess.run(val_autoencoder_loss)
                                        ADV_valid_loss = sess.run(val_adversarial_loss)
                                        sess.run(clf_valid_accuracy_op)
                                    except tf.errors.OutOfRangeError:
                                        break
                                valid_accuracy_value = sess.run(clf_valid_accuracy)
                                print('VAL epoch: {} , clf accuracy : {}, AE loss : {}, ADV loss : {}'.format(epoch, valid_accuracy_value, AE_valid_loss, ADV_valid_loss))            
       
                    # test
                    sess.run(test_running_vars_initializer)  # reinitialize accuracy
                    sess.run(test_iterator.initializer)
                    while True:
                        try:
                            AE_test_loss = sess.run(test_autoencoder_loss)
                            ADV_test_loss = sess.run(test_adversarial_loss)
                            sess.run(test_clf_accuracy_op)
                        except tf.errors.OutOfRangeError:
                            break
                    test_accuracy_value = sess.run(test_clf_accuracy)
                    print('TEST - AE loss : {}, ADV loss : {} '.format(AE_test_loss, ADV_test_loss))
                    print('TEST clf accuracy : {}'.format(test_accuracy_value))
                    test_accuracies.append(test_accuracy_value)
                    AE_test_losses.append(AE_test_loss)
                    ADV_test_losses.append(ADV_test_loss)
                    test_results = pd.DataFrame(
                        {'Parameters': model_params, "AE test loss": AE_test_losses, "test_accuracy": test_accuracies, "ADV test loss": ADV_test_losses}) #Overwrite test results
                    test_results.to_csv(os.path.join(saving_path,'test_results.csv'))

                print('Test results saved')
                                           
                 # Train results into csv
                print("DEBUG - len of adv, ae loss and acc: ({},{},{})".format(len(adversarial_losses), len(autoencoder_losses), len(clf_accuracies)))
                results = pd.DataFrame({"train Adversarial loss": adversarial_losses, "train AE loss": autoencoder_losses, "train clf accuracy": clf_accuracies})
                results.to_csv(os.path.join(saving_path+hyparam_name+'_results.csv'))
                
                print('Train results saved')
                                           
                # ====================== Exporting model =======================

                # serialize weights to HDF5
                '''
                encoder.save_weights(os.path.join(saving_path, hyparam_name+"encoder.h5"))
                print("Encoder saved")
                encoder.save_weights(os.path.join(saving_path, hyparam_name+"decoder.h5"))
                print("Decoder saved")
                '''
                                           
                # ====================== Plotting Train results =======================
                                           
                if len(sys.argv)>1 and str(sys.argv[1])=='plot':
                  print("plotting TRAIN graphs...")
                  plt.figure(1, figsize=(10,15))
                  plt.subplot(3,1,1)
                  plt.plot(even_epoch, adversarial_losses, color="red")
                  plt.title("Adversarial Loss vs Epochs")
                  plt.xlabel("Number of epoch")
                  plt.ylabel("Adversarial Loss")
                  plt.subplot(3,1,2)
                  plt.plot(even_epoch, autoencoder_losses, color="green")
                  plt.title("Autoencoder Loss vs Epochs")
                  plt.xlabel("Number of epoch")
                  plt.ylabel("Autoencoder Loss")
                  plt.subplot(3,1,3)
                  plt.plot(odd_epoch, clf_accuracies, color="orange")
                  plt.title("Classifier Accuracy vs Epochs")
                  plt.xlabel("Number of epoch")
                  plt.ylabel("Classifier accuracy")
                  # TODO : beware if values of beta etc are FLAGS values afterwards :)
                  name="losses and accuracy vs epochs for Beta={Beta}, batch_size={batch_size}, AE_learning_rate={learning_rate}, classifier_learning_rate={clf_learning_rate}.png".format(Beta=beta, batch_size=batch_size, learning_rate=AE_lr,clf_learning_rate=clf_lr)

                  plt.savefig("graphs/"+name)


                  plt.figure(2)
                  plt.plot(clf_accuracies,autoencoder_losses)
                  plt.title("Autoencoder Loss vs Classifier Accuracy")
                  plt.xlabel("Classifier accuracy")
                  plt.ylabel("Autoencoder Loss")
                  plt.savefig("graphs/autoencoder loss vs classifier accuracy for Beta={Beta}, batch_size={batch_size}, AE_learning_rate={learning_rate}, classifier_learning_rate={clf_learning_rate}.png".format(Beta=beta, batch_size=batch_size, learning_rate=AE_lr,clf_learning_rate=clf_lr))

               
                  print('Session successfully closed !')

