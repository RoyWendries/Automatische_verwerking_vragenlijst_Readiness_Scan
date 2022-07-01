import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# def to help building plot
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


# load in data
df = pd.read_json('MovieReactionDS_Medium.json')
df['output'] = df['output'].map({'negative': 0, 'positive': 1})
X = df['input']
y = df['output']

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33)

# vectorize the input text
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(X_train.values)

# setup model layers
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

# fit model
history = model.fit(X_train.values, y_train.values,
                    epochs=30, validation_data=[X_test, y_test], validation_steps=5, batch_size=20)

# print test loss and accuracy
test_loss, test_acc = model.evaluate(X_test.values, y_test.values)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# test model on sample input
sample_text = ('The movie was not good. The animation and the graphics '
               'were terrible. I would not recommend this movie.')
predictions = model.predict(np.array([sample_text]))
print(predictions)

# plot model accuracy and loss
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.show()
