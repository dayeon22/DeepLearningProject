import data_reader
from tensorflow import keras

EPOCHS = 10


def train_model():
    dr = data_reader.DataReader()

    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(rate=0.7),
        keras.layers.Dense(4, activation="softmax")
    ])

    model.compile(optimizer='adam', metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')

    print("\n\n************ TRAINING START ************ ")
    history = model.fit(dr.train_X, dr.train_Y, epochs=EPOCHS,
                        validation_data=(dr.test_X, dr.test_Y))

    model.save('../results/model.h5')
    model.summary()

    data_reader.draw_graph(history)