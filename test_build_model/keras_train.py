from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
import numpy as np
import matplotlib.pyplot as plt

def one_hot_encode(num_classes, input_values):
    encoded_values = np.zeros((len(input_values), num_classes))
    for i, value in enumerate(input_values):
        encoded_values[i, value] = 1
    return encoded_values

df = pd.read_csv('archive/train.csv')

df["context"] = df["Title"] + " " + df["Description"]

df.drop(columns=['Title', 'Description'], inplace=True)

context = df['context'].values
max_features = 20000
tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')

tokenizer.fit_on_texts(context)

num_classes = df['Class Index'].max() # max is 4, min is 1 => num_classes = 4

print(df['Class Index'].value_counts())

X = tokenizer.texts_to_sequences(context)
Y = df['Class Index'].values - 1
Y = one_hot_encode(num_classes, Y)

print(Y)

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)

print(len(x_train))

maxlen = 200

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

inputs = keras.Input(shape=(maxlen, ), dtype='int32')

x = layers.Embedding(max_features, 128)(inputs)

x = layers.SimpleRNN(64, return_sequences=True)(x)

x = layers.SimpleRNN(64)(x)

outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.summary()

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=512, epochs=25, validation_data=(x_val, y_val))

loss_history = history.history['loss']
val_loss_history = history.history['val_loss']

plt.figure(figsize=(8, 6))

plt.plot(loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.legend()

plt.grid(True)
plt.tight_layout()
plt.show()

model_save_path = 'model/keras_model_v1.h5'
model.save(model_save_path)