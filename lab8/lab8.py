import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer

import os

data_path = "/kaggle/input/the-lj-speech-dataset/LJSpeech-1.1"

wavs_path = os.path.join(data_path, "wavs")
metadata_path = os.path.join(data_path, "metadata.csv")

if os.path.exists(data_path):
    print("Папку знайдено! Вміст:", os.listdir(data_path))
else:
    print("Папку LJSpeech-1.1 не знайдено, перевіряємо корінь...")
    data_path = "/kaggle/input/the-lj-speech-dataset"
    wavs_path = os.path.join(data_path, "wavs")
    metadata_path = os.path.join(data_path, "metadata.csv")
    print("Вміст:", os.listdir(data_path))


metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
metadata_df = metadata_df[["file_name", "normalized_transcription"]]
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

print(metadata_df.head(3))


# The set of characters accepted in the transcription.
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size ={char_to_num.vocabulary_size()})"
)


# An integer scalar Tensor. The window length in samples.
frame_length = 256
# An integer scalar Tensor. The number of samples to step.
frame_step = 160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 384


def encode_single_sample(wav_file, label):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
    file = tf.io.read_file(wavs_path + "/" + wav_file + ".wav")
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    ###########################################
    ##  Process the label
    ##########################################
    # 7. Convert label to Lower case
    label = tf.strings.lower(label)
    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 9. Map the characters in label to numbers
    label = char_to_num(label)
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label


def preload_data(df):
    X, y = [], []
    for _, row in df.iterrows():
        file = tf.io.read_file(wavs_path + "/" + row["file_name"] + ".wav")
        audio, _ = tf.audio.decode_wav(file)
        spectrogram = tf.signal.stft(tf.squeeze(audio, axis=-1), frame_length, frame_step, fft_length)
        spectrogram = tf.pow(tf.abs(spectrogram), 0.5)
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)

        label = tf.strings.lower(row["normalized_transcription"])
        label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

        X.append(spectrogram)
        y.append(label)
    return X, y


X_train, y_train = preload_data(df_train)
X_val, y_val = preload_data(df_val)


def get_generator(X, y):
    def generator():
        for i in range(len(X)):
            yield X[i], y[i]

    return generator


signature = (
    tf.TensorSpec(shape=(None, fft_length // 2 + 1), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int64)
)

batch_size = 8

train_dataset = tf.data.Dataset.from_generator(
    get_generator(X_train, y_train), output_signature=signature
).padded_batch(batch_size).prefetch(tf.data.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_generator(
    get_generator(X_val, y_val), output_signature=signature
).padded_batch(batch_size).prefetch(tf.data.AUTOTUNE)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display

fig = plt.figure(figsize=(8, 5))

for batch in train_dataset.take(1):
    spectrogram = batch[0][0].numpy()
    spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
    label = batch[1][0]

    # 1. Відображаємо Спектрограму
    label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
    ax = plt.subplot(2, 1, 1)
    ax.imshow(spectrogram, vmax=1)
    ax.set_title(label)
    ax.axis("off")

    wav_file = list(df_train["file_name"])[0]

    file = tf.io.read_file(wavs_path + "/" + wav_file + ".wav")
    audio, _ = tf.audio.decode_wav(file)
    audio = audio.numpy()

    ax = plt.subplot(2, 1, 2)
    plt.plot(audio)
    ax.set_title("Signal Wave")
    ax.set_xlim(0, len(audio))

    display.display(display.Audio(np.transpose(audio), rate=22050))

plt.show()


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def build_model(input_dim, output_dim, rnn_layers=2, rnn_units=256):
    # Вхід
    input_spectrogram = layers.Input((None, input_dim), name="input")
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)

    # Conv 1
    x = layers.Conv2D(32, [11, 41], strides=[2, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Conv 2
    x = layers.Conv2D(32, [11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Reshape для RNN
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # RNN (GRU) - тепер тут буде лише 2 шари
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(recurrent, name=f"bidirectional_{i}", merge_mode="concat")(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)

    # Dense
    x = layers.Dense(units=rnn_units * 2)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(rate=0.5)(x)

    # Вихід
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)

    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=CTCLoss)
    return model


model = build_model(
    input_dim=fft_length // 2 + 1,
    output_dim=char_to_num.vocabulary_size(),
    rnn_layers=2,
    rnn_units=256,
)
model.summary(line_length=110)


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


# A callback class to output a few transcriptions during training
class CallbackEval(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)


# Define the number of epochs.
epochs = 30
# Callback function to check transcription on the val set.
validation_callback = CallbackEval(validation_dataset)
# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[],
    verbose=2
)

import numpy as np
import tensorflow as tf


def test_model_predictions(num_samples=5):
    print(f"--- ТЕСТУВАННЯ МОДЕЛІ НА {num_samples} ПРИКЛАДАХ ---\n")

    for batch in validation_dataset.take(1):
        X, y = batch

        batch_predictions = model.predict(X, verbose=0)

        input_len = np.ones(batch_predictions.shape[0]) * batch_predictions.shape[1]
        results = tf.keras.backend.ctc_decode(batch_predictions, input_length=input_len, greedy=True)[0][0]

        for i in range(min(num_samples, len(X))):
            predicted_text = tf.strings.reduce_join(num_to_char(results[i])).numpy().decode("utf-8")

            target_text = tf.strings.reduce_join(num_to_char(y[i])).numpy().decode("utf-8")


test_model_predictions(5)