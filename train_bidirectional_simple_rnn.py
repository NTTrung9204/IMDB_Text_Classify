import numpy as np
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt

max_features = 20000
embedding_dim = 128
maxlen = 200

(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=max_features)

print(len(x_train))

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

inputs = keras.Input(shape=(maxlen, ), dtype='int32')

x = layers.Embedding(max_features, 128)(inputs)

x = layers.Bidirectional(layers.SimpleRNN(64, return_sequences=True))(x)

x = layers.Bidirectional(layers.SimpleRNN(64))(x)

outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

model.summary()

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=512, epochs=20, validation_data=(x_val, y_val))

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.plot(train_accuracy, label="Train accuracy")
plt.plot(val_accuracy, label='Validation accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()