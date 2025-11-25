import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Concatenate, GlobalAveragePooling2D, Dense, Dropout, SeparableConv2D
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns



DATA_DIR = '/kaggle/input/catsanddogs8/data'

IMG_SIZE = 299
BATCH_SIZE = 32


try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    class_names = train_ds.class_names
    print(f"Класи: {class_names}")

except Exception as e:
    print("Помилка шляху! Перевірте змінну DATA_DIR. Помилка:", e)


AUTOTUNE = tf.data.AUTOTUNE
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


def conv_block(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name)(x)
    x = BatchNormalization(axis=3, scale=False, name=name + '_bn' if name else None)(x)
    x = Activation('relu', name=name + '_ac' if name else None)(x)
    return x


def build_inception_v3(input_shape=(299, 299, 3)):
    input_layer = Input(shape=input_shape)

    x = conv_block(input_layer, 32, (3, 3), strides=(2, 2), padding='valid')
    x = conv_block(x, 32, (3, 3), padding='valid')
    x = conv_block(x, 64, (3, 3))
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv_block(x, 80, (1, 1), padding='valid')
    x = conv_block(x, 192, (3, 3), padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    branch1x1 = conv_block(x, 64, (1, 1))

    branch5x5 = conv_block(x, 48, (1, 1))
    branch5x5 = conv_block(branch5x5, 64, (5, 5))

    branch3x3dbl = conv_block(x, 64, (1, 1))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_block(branch_pool, 32, (1, 1))

    x = Concatenate(axis=3)([branch1x1, branch5x5, branch3x3dbl, branch_pool])

    branch3x3 = conv_block(x, 384, (3, 3), strides=(2, 2), padding='valid')
    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Concatenate(axis=3)([branch3x3, branch_pool])

    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=outputs, name="My_Inception")
    return model


model = build_inception_v3()


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

train_dataset_aug = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

history = model.fit(
    train_dataset_aug,
    validation_data=val_ds,
    epochs=20,
    verbose=1
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()


y_true = []
y_pred_probs = []

for images, labels in val_ds:
    y_true.extend(labels.numpy())
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds)

y_true = np.array(y_true).flatten()
y_pred_probs = np.array(y_pred_probs).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)


print(classification_report(y_true, y_pred, target_names=class_names))


cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

FOLDER_PATH = '/kaggle/input/testdata'

try:
    file_name = os.listdir(FOLDER_PATH)[0]
    full_path = os.path.join(FOLDER_PATH, file_name)
    print(f"Знайдено файл: {full_path}")

    img = tf.keras.utils.load_img(full_path, target_size=(299, 299))

    img_array = tf.keras.utils.img_to_array(img)

    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    score = prediction[0][0]

    class_names = ['Cat', 'Dog']

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")

    if score > 0.5:
        plt.title(f"Це {class_names[1]} ({100 * score:.2f}%)")
        print(f"Результат: {class_names[1]} з ймовірністю {100 * score:.2f}%")
    else:
        plt.title(f"Це {class_names[0]} ({100 * (1 - score):.2f}%)")
        print(f"Результат: {class_names[0]} з ймовірністю {100 * (1 - score):.2f}%")

    plt.show()

except IndexError:
    print(f"Помилка: Папка '{FOLDER_PATH}' порожня або шлях вказано неправильно.")
except Exception as e:
    print(f"Виникла помилка: {e}")