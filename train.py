import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH, MAPPING_PATH
from typing import List
import json

NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"


def build_model(output_units: int, num_units: List[int], loss: str, learning_rate: float) -> keras.Model:
    """Builds and compiles model

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply

    :return model (tf model): Where the magic happens :D
    """

    # create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compile model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model


def train(num_units: List[int] = NUM_UNITS, loss: str = LOSS,
          learning_rate: float = LEARNING_RATE):
    """Train and save TF model.

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply
    """
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # build the network
    output_units = len(mappings)
    model = build_model(output_units, num_units, loss, learning_rate)

    # train the model
    class_weight = {clss: 1 if clss in (mappings['/'], mappings['_']) else 40 for clss in range(output_units)}
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight)

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()
