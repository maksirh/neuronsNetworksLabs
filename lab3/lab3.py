import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Точність на тестових даних: {accuracy * 100:.2f}")


plt.plot(history.history["accuracy"], label="Точність на train")
plt.plot(history.history["val_accuracy"], label="Точність на test")
plt.xlabel("Епоха")
plt.ylabel("Точність")

plt.legend()
plt.show()

predictions = model.predict(x_test[:5])
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Передбачено: {np.argmax(predictions[i])}")
    plt.show()


def recognize_image(image_path, model):
    img = Image.open(image_path).convert('L')

    img_resized = img.resize((28, 28), Image.LANCZOS)

    img_array = np.array(img_resized).astype("float32") / 255.

    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array

    img_vector = img_array.reshape(1, 784)
    prediction = model.predict(img_vector, verbose=0)

    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Розпізнана цифра: {predicted_digit}")
    print(f"Впевненість: {confidence:.1%}")

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title(f'Вхідне зображення')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    bars = plt.bar(range(10), prediction[0])
    bars[predicted_digit].set_color('red')
    plt.title(f'Прогноз: {predicted_digit}')
    plt.xlabel('Цифра')
    plt.ylabel('Ймовірність')
    plt.xticks(range(10))

    plt.tight_layout()
    plt.show()

    return predicted_digit, confidence

recognize_image("img/img.png", model)
recognize_image("img/img_1.png", model)

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(f"Accuracy: {accuracy:.4f}")

precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Матриця помилок')
plt.xlabel('Предсказані мітки')
plt.ylabel('Справжні мітки')
plt.show()