import io
import pandas as pd
import os
import numpy as np
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

"""# Data processing

## Loading data
"""

df = pd.read_csv("data_with_gender_rec.csv")

"""##Stop words removal"""

import string
filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

#Import stop words
language = 'english'
stop_words = set(stopwords.words(language)) 
punctuation = string.punctuation + '-' + '['+ '–'+ '\uf0a7'+ ']' + '•' + filters #remember to remove utf words

#Row by row tokenization
def tokenization_and_stop_words_out(text):
  x = word_tokenize(text)
  y = [w for w in x if not w in stop_words and not w in punctuation]
  return y
  
df.loc[:,'cv'] = df['cv'].apply(tokenization_and_stop_words_out)

"""## Encoding labels

Here we use to one hot encoding for encoding genders
"""

y = pd.get_dummies(df['label'])
y = y.values

"""## Splitting into train and test df"""

X =df.drop(['id', 'label'], axis =1)

from sklearn.model_selection import train_test_split

df_train, df_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(len(df_train))

"""## Tokenization"""
#Tokenizer training 
filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' #punctuation removal
num_words = 10000
len_max_seq = 840
train_values = df_train.loc[:,'cv'].tolist()
test_values = df_test.loc[:,'cv'].tolist()

tokenizer = Tokenizer(num_words = num_words, filters= filters,lower =True)
tokenizer.fit_on_texts(df_train['cv'].tolist())

#Text to sequences
X_train = tokenizer.texts_to_sequences(train_values)
X_test = tokenizer.texts_to_sequences(test_values)

#Padding sequences
X_train = pad_sequences(X_train, len_max_seq)
X_test = pad_sequences(X_test, len_max_seq)

