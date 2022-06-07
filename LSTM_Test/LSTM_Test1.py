# LSTM for sequence classification in the IMDB dataset
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.layers import BatchNormalization


df = pd.read_json('MovieReactionDS_small.json')
tf.convert_to_tensor(df)
X_train, X_test, y_train, y_test = train_test_split(
    df['input'], df['output'], test_size=0.3)

# create the model
model = Sequential()
model.add(LSTM(64, input_shape=(None, 1)))
model.add(Dense(10, activation='sigmoid'))
model.compile(optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(
    X_test, y_test), batch_size=10, epochs=3)
