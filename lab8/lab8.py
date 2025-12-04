import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

wavs_path = "/kaggle/input/the-lj-speech-dataset/LJSpeech-1.1/wavs/"
metadata_path = "/kaggle/input/the-lj-speech-dataset/LJSpeech-1.1/metadata.csv"

metadata_df = pd.read_csv(metadata_path, sep='|', header=None, quoting=3)
metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
metadata_df = metadata_df[["file_name", "normalized_transcription"]]
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

fft_length = 384
frame_length = 256
frame_step = 160


def encode_single_sample(wav_file, label):
    file = tf.io.read_file(wavs_path + wav_file + ".wav")
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)

    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)

    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    label = tf.strings.lower(label)
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    label = char_to_num(label)
    return spectrogram, label


from tensorflow.keras import layers

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]

char_to_num = layers.StringLookup(vocabulary=characters, oov_token="")

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

print(f"Розмір словника: {char_to_num.vocabulary_size()} символів")

metadata_df = metadata_df.iloc[:4500]

split = int(len(metadata_df) * 0.90)
df_train = metadata_df[:split]
df_val = metadata_df[split:]

print(f"Train samples: {len(df_train)}")
print(f"Val samples: {len(df_val)}")

batch_size = 32


def create_dataset(df):
    dataset = tf.data.Dataset.from_tensor_slices(
        (df["file_name"], df["normalized_transcription"])
    )
    dataset = dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


train_dataset = create_dataset(df_train)
validation_dataset = create_dataset(df_val)


def build_model(input_dim, output_dim, rnn_layers=2, rnn_units=128):
    input_spectrogram = layers.Input((None, input_dim), name="input")

    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)

    x = layers.Conv2D(32, kernel_size=[11, 41], strides=[2, 2], padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, kernel_size=[11, 21], strides=[1, 2], padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    new_shape = (-1, x.shape[-2] * x.shape[-1])
    x = layers.Reshape(new_shape)(x)
    x = layers.Dense(rnn_units)(x)

    for i in range(rnn_layers):
        recurrent = layers.Bidirectional(
            layers.LSTM(rnn_units, return_sequences=True, dropout=0.2),
            name=f"bi_lstm_{i}"
        )
        x = recurrent(x)
        x = layers.BatchNormalization()(x)

    x = layers.Dense(output_dim + 1, activation="softmax", name="output")(x)

    output = x
    model = keras.Model(input_spectrogram, output, name="DeepSpeech2")
    return model


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


model = build_model(input_dim=fft_length // 2 + 1, output_dim=char_to_num.vocabulary_size(), rnn_units=256)
opt = keras.optimizers.Adam(learning_rate=1e-4)

model.compile(optimizer=opt, loss=CTCLoss)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


batch = next(iter(validation_dataset))
X_test, y_test = batch

preds = model.predict(X_test)
pred_texts = decode_batch_predictions(preds)

print("\n--- ПОРІВНЯННЯ РЕЗУЛЬТАТІВ ---\n")

for i in range(5):
    prediction = pred_texts[i]

    original = tf.strings.reduce_join(num_to_char(y_test[i])).numpy().decode("utf-8")

    print(f"Аудіо #{i + 1}")
    print(f"Оригінал:   {original}")
    print(f"Прогноз:    {prediction}")
    print("-" * 30)

from spellchecker import SpellChecker
from jiwer import wer
import numpy as np
import tensorflow as tf

spell = SpellChecker()


def correct_text(text):
    if not text: return ""
    words = text.split()

    corrected_words = []
    for word in words:
        correction = spell.correction(word)
        if correction:
            corrected_words.append(correction)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)


print("Генерація прогнозів для розрахунку WER (це займе хвилину)...")

original_transcriptions = []
predicted_raw = []

for batch in validation_dataset.take(5):
    X_val, y_val = batch

    preds = model.predict(X_val, verbose=0)
    pred_texts = decode_batch_predictions(preds)

    for label in y_val:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        original_transcriptions.append(label)

    predicted_raw.extend(pred_texts)

predicted_corrected = [correct_text(txt) for txt in predicted_raw]

wer_raw = wer(original_transcriptions, predicted_raw)
wer_corrected = wer(original_transcriptions, predicted_corrected)

print("-" * 30)
print(f"WER (Сирий вихід):      {wer_raw:.4f}")
print(f"WER (Після корекції):   {wer_corrected:.4f}")
print("-" * 30)

print(f"Оригінал:  {original_transcriptions[0]}")
print(f"Сирий:     {predicted_raw[0]}")
print(f"Виправлений: {predicted_corrected[0]}")