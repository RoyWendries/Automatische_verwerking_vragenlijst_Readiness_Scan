# LSTM for sequence classification in the IMDB dataset
# Importing the required libraries
import unicodedata
import string
import re
from sklearn.metrics import f1_score, confusion_matrix
from keras import backend
from keras.layers import *
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.pooling import GlobalMaxPooling1D
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from sklearn.model_selection import train_test_split


from keras.layers import BatchNormalization
from AttentionLayer import AttentionWithContext
from sklearn.model_selection import train_test_split
from keras.layers import Bidirectional
from keras.optimizers import Adam


# Ignoring the warnings
import warnings
warnings.filterwarnings('ignore')


df = pd.read_json('MovieReactionDS_medium.json')
df['output'] = df['output'].map({'negative': 0, 'positive': 1})
X = df['input']
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33)

# Function for Text Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# cleaned text looks like shit potential problem here
def clean_text(X):
    processed = []
    for text in X:
        text = text[0]
        text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
        text = re.sub('<.*?>', '', text)
        text = text.lower()
        text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
        text = [lemmatizer.lemmatize(token, "v") for token in text]
        text = [word for word in text if not word in stop_words]
        text = " ".join(text)
        processed.append(text)
    return processed


X_train_final = clean_text(X_train)
X_test_final = clean_text(X_test)


print(f'Number of rows & columns in X : {np.shape(X_train_final)}')
print(f'Number of rows & columns in y : {np.shape(y)}')
print(f'Number of categories in y : {len(np.unique(y))}')
print(f'Categories in y : {np.unique(y)}')
print(X_train_final)

'''
#Tokenization and Padding
vocab_size = 60000
maxlen = 500
encode_dim = 20
batch_size = 32
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_final)
tokenized_word_list = tokenizer.texts_to_sequences(X_train_final)
X_train_padded = pad_sequences(
    tokenized_word_list, maxlen=maxlen, padding='post')


#EarlyStopping and ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('model_best.h5', monitor='val_accuracy',
                     mode='max', verbose=1, save_best_only=True)

# Building the model
model = Sequential()
embed = Embedding(input_dim=vocab_size, output_dim=20,
                  input_length=X_train_padded.shape[1])
model.add(embed)
model.add(Bidirectional(LSTM(200, return_sequences=True)))
model.add(Dropout(0.3))
model.add(AttentionWithContext())
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.2))
model.add(Dense(1, activation='sigmoid'))
opt = Adam(learning_rate=1e-6)
model.compile(loss='binary_crossentropy',
              optimizer=opt, metrics=['accuracy'])
model.summary()

X_train_final2, X_val, y_train_final2, y_val = train_test_split(
    X_train_padded, y_train, test_size=0.2)
model.fit(X_train_final2, y_train_final2, epochs=50, batch_size=batch_size,
          verbose=1, validation_data=[X_val, y_val], callbacks=[es, mc])

# Padding the test data
tokenized_word_list_test = tokenizer.texts_to_sequences(X_test_final)
X_test_padded = pad_sequences(
    tokenized_word_list_test, maxlen=maxlen, padding='post')

# Evaluating the model
model = load_model('model_best.h5', custom_objects={
                   "AttentionWithContext": AttentionWithContext, "backend": backend})
score, acc = model.evaluate(X_test_padded, y_test)
print('The accuracy of the model on the test set is ', acc*100)
prediction = model.predict(X_test_padded)
y_pred = (prediction > 0.5)
print('F1-score: ', (f1_score(y_pred, y_test)*100))
print('Confusion matrix:')
print(confusion_matrix(y_pred, y_test))
'''
