import pandas as pd
import os
import numpy as np
import numpy as np
import string
import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

"""# Data processing

## Loading data
"""

#Importing
data_dir = '../data'
df = pd.read_csv(os.path.join(data_dir,"data_with_gender_rec.csv"))

#leaning data
df = df[['MATRICULEINT', 'TXT', 'GENRE']]
df.loc[:,'GENRE'] = pd.to_numeric(df.loc[:,'GENRE'], errors = 'coerce', downcast = 'integer')
df.dropna(inplace = True)
df = df[df['GENRE'].isin([1, 2])]

"""##Stop words removal"""
filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

#Import stop words
language = 'french'
stop_words = set(stopwords.words(language)) 
punctuation = string.punctuation + filters #remember to remove utf words

#Row by row tokenization
def tokenization_and_stop_words_out(text):
  x = word_tokenize(text)
  y = [w for w in x if not w in stop_words and not w in punctuation]
  return y
  
df.loc[:,'TXT'] = df['TXT'].apply(tokenization_and_stop_words_out)

"""## Encoding labels

Here we use to one hot encoding for encoding genders
"""

df = pd.get_dummies(df, columns = ['GENRE'])

"""## Splitting into train and test df"""

#Initial split into train and test dataframes
df_train_init, df_test = train_test_split(df, test_size = 0.25)

#Second split of train dataframe into train and val dataframes
df_train, df_val = train_test_split(df_train_init, test_size = 0.25)

datasets  = [df_train, df_val, df_test]

"""## Tokenization"""
#Tokenizer training 
num_words = 10000
len_max_seq = df_train.apply(lambda x: len(x['TXT']), axis = 1).max()

train_values = df_train.loc[:,'TXT'].tolist()

tokenizer = Tokenizer(num_words = num_words, filters= filters,lower =True)
tokenizer.fit_on_texts(train_values)

#Text to padded sequences
for df in datasets:
  df['TXT'] = tokenizer.texts_to_sequences(df.loc[:,'TXT'])
  df['TXT'] = pad_sequences(df.loc[:,'TXT'], maxlen = len_max_seq).tolist()

"""## Export"""
df_train.to_csv(os.path.join(data_dir, "train.csv"),index = False)
df_val.to_csv(os.path.join(data_dir, "val.csv"),index = False)
df_test.to_csv(os.path.join(data_dir, "test.csv"),index = False)






