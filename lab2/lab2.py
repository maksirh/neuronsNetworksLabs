import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Input, Add
import matplotlib.pyplot as plt



def generate_data(n_samples=1000):
    X = np.random.uniform(-5, 5, (n_samples, 2))
    Y = X[:, 0] * np.sin(X[:, 1]) + X[:, 1] * np.cos(X[:, 0])
    return X, Y


X, Y = generate_data()

X_train, X_test = X[:800], X[800:]
Y_train, Y_test = Y[:800], Y[800:]


def feed_forward_backprop_10():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_dim=2, activation="tanh"),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.01),
        loss='mse',
        metrics=['mae'])

    history = model.fit(X_train, Y_train, epochs=500, verbose=2)

    loss, mae = model.evaluate(X_test, Y_test)
    print("mse", loss)
    print("mae", mae)

    plt.plot(history.history['loss'], label="feed forward backprop 10")
    plt.title("Зміна MSE під час навчання")
    plt.xlabel("Епохи")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def feed_forward_backprop_20():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, input_dim=2, activation="tanh"),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.01),
        loss='mse',
        metrics=['mae'])

    history = model.fit(X_train, Y_train, epochs=500)

    loss, mae = model.evaluate(X_test, Y_test)
    print("mse", loss)
    print("mae", mae)

    plt.plot(history.history['loss'], label="feed forward backprop 20")
    plt.title("Зміна MSE під час навчання")
    plt.xlabel("Епохи")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def cascade_forward_backprop_20():

    inputs = Input(shape=(2,))
    hidden1 = Dense(20, activation='relu')(inputs)
    output = Dense(1)(hidden1)
    output_cascade = Dense(1)(inputs)
    final_output = Add()([output, output_cascade])
    cascade_model_1 = Model(inputs=inputs, outputs=final_output)

    cascade_model_1.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.01),
        loss='mse',
        metrics=['mae'])

    history = cascade_model_1.fit(X_train, Y_train, epochs=500)

    loss, mae = cascade_model_1.evaluate(X_test, Y_test)
    print("mse", loss)
    print("mae", mae)

    plt.plot(history.history['loss'], label="cascade forward backprop 20")
    plt.title("Зміна MSE під час навчання")
    plt.xlabel("Епохи")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

def cascade_forward_backprop_10_10():

    inputs_2 = Input(shape=(2,))
    hidden1_2 = Dense(10, activation='relu')(inputs_2)
    hidden2_2 = Dense(10, activation='relu')(hidden1_2)
    output_2 = Dense(1)(hidden2_2)
    output_cascade_2 = Dense(1)(inputs_2)
    final_output_2 = Add()([output_2, output_cascade_2])
    cascade_model_2 = Model(inputs=inputs_2, outputs=final_output_2)

    cascade_model_2.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.01),
        loss='mse',
        metrics=['mae'])

    history = cascade_model_2.fit(X_train, Y_train, epochs=500)

    loss, mae = cascade_model_2.evaluate(X_test, Y_test)
    print("mse", loss)
    print("mae", mae)

    plt.plot(history.history['loss'], label="cascade forward backprop 10 10")
    plt.title("Зміна MSE під час навчання")
    plt.xlabel("Епохи")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def elman_backprop_15():

    model = Sequential([
        SimpleRNN(15, activation='tanh', input_shape=(1, 2)),
        Dense(1, activation='linear')
    ])

    X_train_simple = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_simple = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mae']
    )

    history = model.fit(X_train_simple, Y_train, epochs=500, verbose=1)

    loss, mae = model.evaluate(X_test_simple, Y_test, verbose=0)
    print("mse", loss)
    print("mae", mae)

    plt.plot(history.history['loss'], label="elman backprop 15")
    plt.title("Зміна MSE під час навчання")
    plt.xlabel("Епохи")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()




def elman_backprop_5_5_5():
    inputs = Input(shape=(1, 2))

    x = SimpleRNN(5, activation='tanh', return_sequences=True)(inputs)
    x = SimpleRNN(5, activation='tanh', return_sequences=True)(x)
    x = SimpleRNN(5, activation='tanh', return_sequences=False)(x)

    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)

    X_train_simple = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_simple = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mae']
    )

    history = model.fit(X_train_simple, Y_train, epochs=500, verbose=1)

    loss, mae = model.evaluate(X_test_simple, Y_test, verbose=0)
    print("mse", loss)
    print("mae", mae)

    plt.plot(history.history['loss'], label="elman backprop 5 5 5")
    plt.title("Зміна MSE під час навчання")
    plt.xlabel("Епохи")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    feed_forward_backprop_10()
    feed_forward_backprop_20()

    cascade_forward_backprop_20()
    cascade_forward_backprop_10_10()

    elman_backprop_15()
    elman_backprop_5_5_5()

