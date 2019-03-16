#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import pandas as pd
import os
import numpy as np

#Load data
data_dir = os.getcwd()+"/data/"
print(data_dir)
train = pd.read_csv(data_dir+"train.csv")
test = pd.read_csv(data_dir+"test.csv")
val = pd.read_csv(data_dir+"val.csv")

#Keep only tokekization representation of each doc and the gender label
X_train, y_train = train.iloc[:,1], train.iloc[:,2]
X_test, y_test = test.iloc[:,1], test.iloc[:,2]
X_val, y_val = val.iloc[:,1], val.iloc[:,2]

#Hyperparameters
DATA_SIZE = 6840 #TODO mettre taille de sequence
NUM_HIDDEN_1 = 512
NUM_HIDDEN_2 = 512
NUM_HIDDEN_3 = 512
NUM_CLASSES = 2 # Male & Female


# Functions

def init_weights(shape):
    """
    Randomly initializes the weights of a variable of
    shape = 'shape' with Gaussian repartition. This function returns a tensor of
    the specified shape filled with random values.
    """
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

#Initiating weights -> TODO we can see whether there are existing weights and do fine-tuning
w_h1 = init_weights([DATA_SIZE, NUM_HIDDEN_1])
w_h2 = init_weights([NUM_HIDDEN_1, NUM_HIDDEN_2])
w_h3 = init_weights([NUM_HIDDEN_2, NUM_HIDDEN_3])
w_o = init_weights([NUM_HIDDEN_3, NUM_CLASSES])


def model(X, w_h1, w_h2, w_h3, w_o):
    '''
    Model with 3 hidden layers
    :param X:
    :param w_h1:
    :param w_h2:
    :param w_h3:
    :param w_o:
    :return:
    '''
    h1 = tf.nn.relu(tf.matmul(X, w_h1))
    h2 = tf.nn.relu(tf.matmul(h1, w_h2))
    h3 = tf.nn.relu(tf.matmul(h2, w_h3))
    return tf.matmul(h3, w_o)


X = tf.placeholder(tf.float32, [None, DATA_SIZE])
Y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

Y_p = model(X, w_h1, w_h2, w_h3, w_o)

cost_function = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_p, labels=Y))

optimization_algorithm = tf.train.AdamOptimizer(0.5) \
    .minimize(cost_function)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(Y_p,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
BATCH_SIZE = 1000

for epoch in range(10000):
    # melange des donnees a chaque epoch (=iteration d'apprentissage)
    p = np.random.permutation(range(len(X_train)))
    X_train, y_train = X_train[p], y_train[p]# apprentissage avec des minibatches de taille 1000
    p2 = np.random.permutation(range(len(X_val)))
    X_val , y_val = X_val[p2], y_val[p2]
    for start in range(0, len(X_train), BATCH_SIZE):
        end = start + BATCH_SIZE
        optimization_algorithm.run(feed_dict={X: X_train[start:end], Y: y_train[start:end]})

    if epoch%500 ==0:
        train_accuracy = accuracy.eval(feed_dict = {X:X_train\
                                                    , Y:y_train})
        val_accuracy = accuracy.eval(feed_dict = {X:X_val\
                                                    , Y:y_val})
        print("epoch: %d, training accuracy: %g"%(epoch, train_accuracy))
        print("epoch: %d, validation accuracy: %g"%(epoch, val_accuracy))
    #if epoch%1000 == 0:
       # print("epoch: %d, \n\nTest accuracy: %g" % (epoch, accuracy.eval(feed_dict={X: X_test, Y: y_test})))

# Testing the network on the test data
    print("\n\nTest accuracy: %g" % accuracy.eval(
        feed_dict={X: X_test, Y: y_test,}))

#Exporting model
choice = None
while choice == None:
    print(': = = = = = = = = = = = = = = = = = = = = = = = = = = :')
    choice = input('Voulez-vous exporter ce modèle ? :\ny - oui\nn - non\n')
    if choice == 'y':
        model_name = input('saisissez le nom de ce modèle \n')
        export_path = os.getcwd()+"/models/"+model_name
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.save()