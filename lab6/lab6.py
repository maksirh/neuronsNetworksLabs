import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

SOURCE_DIR = '/kaggle/input/logo-dataset/data'
WORK_DIR = '/kaggle/working/balanced_dataset'

if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)

os.makedirs(os.path.join(WORK_DIR, 'Starbucks'))
os.makedirs(os.path.join(WORK_DIR, 'Other'))

print("Копіювання та балансування даних...")

starbucks_files = os.listdir(os.path.join(SOURCE_DIR, 'Starbucks'))
other_files = os.listdir(os.path.join(SOURCE_DIR, 'Other'))

min_count = min(len(starbucks_files), len(other_files))
print(f"Балансуємо до {min_count} зображень на кожен клас.")

selected_starbucks = np.random.choice(starbucks_files, min_count, replace=False)
for fname in selected_starbucks:
    src = os.path.join(SOURCE_DIR, 'Starbucks', fname)
    dst = os.path.join(WORK_DIR, 'Starbucks', fname)
    shutil.copyfile(src, dst)

selected_other = np.random.choice(other_files, min_count, replace=False)
for fname in selected_other:
    src = os.path.join(SOURCE_DIR, 'Other', fname)
    dst = os.path.join(WORK_DIR, 'Other', fname)
    shutil.copyfile(src, dst)

print("Готово! Дані підготовлено в /kaggle/working/balanced_dataset")

IMG_SIZE = (299, 299)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

print("\nПеревірка генераторів:")
train_ds = train_datagen.flow_from_directory(
    WORK_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_ds = train_datagen.flow_from_directory(
    WORK_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"Index mapping: {train_ds.class_indices}")


def entry_flow(inputs):
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    previous_block_activation = x

    for size in [128, 256, 728]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

        residual = layers.Conv2D(size, 1, strides=2, padding='same', use_bias=False)(previous_block_activation)
        residual = layers.BatchNormalization()(residual)

        x = layers.add([x, residual])
        previous_block_activation = x
    return x


def middle_flow(x, num_blocks=8):
    previous_block_activation = x
    for _ in range(num_blocks):
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        x = layers.add([x, previous_block_activation])
        previous_block_activation = x
    return x


def exit_flow(x):
    previous_block_activation = x

    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(728, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(1024, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    residual = layers.Conv2D(1024, 1, strides=2, padding='same', use_bias=False)(previous_block_activation)
    residual = layers.BatchNormalization()(residual)

    x = layers.add([x, residual])

    x = layers.SeparableConv2D(1536, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.SeparableConv2D(2048, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return x


inputs = Input(shape=(299, 299, 3))
outputs = exit_flow(middle_flow(entry_flow(inputs)))
model = models.Model(inputs, outputs, name='Manual_Xception')

print("Завантаження ваг ImageNet...")
base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

print("Перенос ваг...")

count = 0
base_weights = [l for l in base_model.layers if len(l.weights) > 0]
my_weights = [l for l in model.layers if len(l.weights) > 0]

for src, dst in zip(base_weights, my_weights):
    if len(src.get_weights()) == len(dst.get_weights()):
        if src.get_weights()[0].shape == dst.get_weights()[0].shape:
            dst.set_weights(src.get_weights())
            count += 1
print(f"Перенесено {count} шарів.")

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

print("Початок навчання...")
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds
)

predictions = model.predict(val_ds)
y_pred = (predictions > 0.5).astype(int)
y_true = val_ds.classes

class_labels = list(train_ds.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=class_labels))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()


def smart_detector_filtered(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Аналіз: {video_path}...")

    scores = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_num % 5 == 0:
            img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (299, 299)) / 255.0
            pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
            scores.append((frame_num / fps, pred))
        frame_num += 1
    cap.release()

    avg = np.mean([s[1] for s in scores])
    targets = [t for t, s in scores if (s < avg - 0.02 if avg > 0.95 else s > 0.8)]

    if targets:
        print(f"\n Логотип знайдено (середня впевненість: {avg:.2f}):")
        start = end = targets[0]

        def print_if_valid(s, e):
            duration = e - s
            if duration >= 0.5:
                print(f"⏰ {s:.1f}с — {e:.1f}с")

        for t in targets[1:]:
            if t - end > 1.0:
                print_if_valid(start, end)
                start = t
            end = t
        print_if_valid(start, end)
    else:
        print("\nНічого не знайдено.")


smart_detector_filtered('/kaggle/input/my-video/rest_video.mp4', model)