import numpy as np
import pandas as pd
import tensorflow as tf
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

DATA_PATH = '/kaggle/input/yelp-review-dataset/yelp_review_polarity_csv/train.csv'

train_df = pd.read_csv(DATA_PATH, names=['label', 'text'], nrows=50000)

train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 1 else 1)
print(f"Успішно завантажено {len(train_df)} рядків.")

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(train_df['text'].values)
X = tokenizer.texts_to_sequences(train_df['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = train_df['label'].values
# Спліт
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(X_train, Y_train,
                    epochs=5,
                    batch_size=64,
                    validation_split=0.1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, min_delta=0.001)])

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.title("Loss over epochs")
plt.show()

Y_pred_prob = model.predict(X_test)
Y_pred = (Y_pred_prob > 0.5).astype(int)

print("\nAccuracy:", accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


def check_review(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)[0][0]
    label = "POSITIVE" if pred > 0.5 else "NEGATIVE"
    print(f"Text: {text}\nPrediction: {label} ({pred:.4f})\n")


check_review("The service was slow and food was cold")
check_review("Absolutely delicious and great atmosphere")

import spacy
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.model_selection import train_test_split


nlp = spacy.load('en_core_web_sm')


subset_size = 10000

print(f"Беремо перші {subset_size} відгуків для експерименту...")
small_df = train_df.head(subset_size).copy()


def spacy_processor(texts):
    processed = []
    for doc in nlp.pipe(texts, disable=["parser", "ner"], batch_size=1000):
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        processed.append(" ".join(tokens))
    return processed


print("Обробка тексту через SpaCy (це займе 1-2 хвилини)...")
small_df['text_spacy'] = spacy_processor(small_df['text'].values)

print("Готово!")
print(f"Оригінал: {small_df['text'].iloc[0][:50]}...")
print(f"SpaCy:    {small_df['text_spacy'].iloc[0][:50]}...")


def train_and_evaluate(texts, labels, name):
    tok = Tokenizer(num_words=10000)
    tok.fit_on_texts(texts)
    X_seq = tok.texts_to_sequences(texts)
    X_pad = pad_sequences(X_seq, maxlen=150)

    X_tr, X_te, y_tr, y_te = train_test_split(X_pad, labels, test_size=0.2, random_state=42)

    m = Sequential([
        Embedding(10000, 64),
        SpatialDropout1D(0.2),
        LSTM(64, dropout=0.2, recurrent_dropout=0),
        Dense(1, activation='sigmoid')
    ])
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(f"\n--- Навчання методу: {name} ---")
    m.fit(X_tr, y_tr, epochs=4, batch_size=64, verbose=1, validation_split=0.1)

    acc = m.evaluate(X_te, y_te, verbose=0)[1]
    return acc


acc_standard = train_and_evaluate(small_df['text'].values, small_df['label'].values, "Standard Keras")
acc_spacy = train_and_evaluate(small_df['text_spacy'].values, small_df['label'].values, "SpaCy Lemmatization")

results = pd.DataFrame({
    'Method': ['Standard Tokenizer', 'SpaCy Lemmatization'],
    'Accuracy': [acc_standard, acc_spacy]
})

print("\n=== РЕЗУЛЬТАТИ ПОРІВНЯННЯ ===")
print(results)