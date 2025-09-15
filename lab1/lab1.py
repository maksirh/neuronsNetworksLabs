import tensorflow as tf
import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_dim=2, activation="tanh"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=500)

loss, accuracy = model.evaluate(x, y)
print("loss", loss)
print("accuracy", accuracy)

prediction = model.predict(x)
for inp, pred in zip(x, prediction):
    print(inp, round(pred[0]))