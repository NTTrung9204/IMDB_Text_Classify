import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import pickle
import re

max_features = 20000
embedding_dim = 128
maxlen = 200

def remove_html_tags(text):
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text

def preprocess_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def text_to_sequence(text, word_index, max_features):
    words = text.split()
    sequence = []
    for word in words:
        index = word_index.get(word, 2)
        if index < max_features:
            sequence.append(index)
        else:
            sequence.append(2)
    return sequence

print("Loading model and word index for prediction...")
model = keras.models.load_model('sentiment_model_lstm.h5')
with open('imdb_word_index.pickle', 'rb') as handle:
    word_index = pickle.load(handle)

def predict_sentiment(text, model, word_index, max_features=max_features, maxlen=maxlen):

    text = preprocess_text(text)
    sequence = text_to_sequence(text, word_index, max_features)
    padded = pad_sequences([sequence], maxlen=maxlen)
    prediction = model.predict(padded)[0][0]
    return "negative" if prediction < 0.5 else "positive"

csv_file = 'archive/imdb.csv'
print("Loading CSV file...")
df = pd.read_csv(csv_file)

df['review'] = df['review'].apply(preprocess_text)

df_test = df.head(1000)

correct = 0
total = len(df_test)
predictions = []

# print("Predicting sentiment for CSV reviews...")
# for i, row in df_test.iterrows():
#     review_text = row['review']
#     true_label = row['sentiment'].strip().lower()
#     pred_label = predict_sentiment(review_text, model, word_index)
#     predictions.append(pred_label)
#     if pred_label == true_label:
#         correct += 1

# accuracy = correct / total * 100
# print(f'Accuracy on {total} reviews: {accuracy:.2f}%')

review_text = """
    Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his parents are fighting all the time.<br /><br />This movie is slower than a soap opera... and suddenly, Jake decides to become Rambo and kill the zombie.<br /><br />OK, first of all when you're going to make a film you must Decide if its a thriller or a drama! As a drama the movie is watchable. Parents are divorcing & arguing like in real life. And then we have Jake with his closet which totally ruins all the film! I expected to see a BOOGEYMAN similar movie, and instead i watched a drama with some meaningless thriller spots.<br /><br />3 out of 10 just for the well playing parents & descent dialogs. As for the shots with Jake: just ignore them.

    """
print(predict_sentiment(review_text, model, word_index))