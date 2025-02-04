import numpy as np
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

max_features = 20000
embedding_dim = 128
maxlen = 200

(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=max_features)

print(len(x_train))

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

inputs = keras.Input(shape=(maxlen, ), dtype='int32')

x = layers.Embedding(max_features, 128)(inputs)

x = layers.LSTM(64, return_sequences=True)(x)

x = layers.LSTM(64)(x)

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

model.save('sentiment_model_lstm.h5')
print("Model saved to 'sentiment_model_lstm.h5'")

print("Saving word index...")
word_index = keras.datasets.imdb.get_word_index()

word_index = {word: (index + 3) for word, index in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

with open('imdb_word_index.pickle', 'wb') as handle:
    pickle.dump(word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Word index saved to 'imdb_word_index.pickle'")