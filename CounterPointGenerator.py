import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH, preprocess_song
from typing import List


class CounterPointGenerator:
    """A class that wraps the LSTM model and offers utilities to generate counterpoint melodies."""

    def __init__(self, model_path: str = "model.h5"):
        """Constructor that initialises TensorFlow model"""

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_counterpoint(self, cantus: str, max_sequence_length: int, temperature: float) -> List[str]:
        """Generates a melody using the DL model and returns a midi file.

        :param canus (str): Cantus firmus melody seed with the notation used to encode the dataset
        :param num_steps (int): Number of steps to be generated
        :param max_sequence_len (int): Max number of steps in seed to be considered for generation
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return melody (list of str): List with symbols representing a melody
        """

        # create seed with start symbols
        seed = cantus.split()[:(max_sequence_length - 1)]
        original_length = len(seed)
        melody = seed
        # seed = self._start_symbols + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        seed = seed  # + [self._mappings["<BOC>"]]

        for _ in range(original_length):

            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # update seed
            seed.append(int(output_int))

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # update melody
            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilites: np.ndarray, temperature: float):
        """Samples an index from a probability array reapplying softmax using temperature

        :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return index (int): Selected output symbol
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites))  # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index

    def save_melody(self, cantus_and_counter: List[str], step_duration: float =0.25, format: str ="midi", file_name: str ="mel.mid"):
        """Converts a melody into a MIDI file

        :param melody (list of str):
        :param min_duration (float): Duration of each time step in quarter length
        :param file_name (str): Name of midi file
        :return:
        """

        # create a music21 stream
        s = m21.stream.Score(id='mainScore')
        p0 = m21.stream.Part(id='part0')
        p1 = m21.stream.Part(id='part1')

        split_point = int(len(cantus_and_counter) / 2)
        cantus, counter = cantus_and_counter[:split_point], cantus_and_counter[split_point:]

        def parse_to_part(part: m21.stream.Part, melody):
            start_symbol = None
            step_counter = 1
            # parse all the symbols in the melody and create note/rest objects
            for i, symbol in enumerate(melody):

                # handle case in which we have a note/rest
                if symbol != "_" or i + 1 == len(melody):

                    # ensure we're dealing with note/rest beyond the first one
                    if start_symbol is not None:

                        quarter_length_duration = step_duration * step_counter  # 0.25 * 4 = 1

                        # handle rest
                        if start_symbol == "r":
                            m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                        elif start_symbol == '/':
                            continue
                        # handle note
                        else:
                            m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                        part.append(m21_event)

                        # reset the step counter
                        step_counter = 1

                    start_symbol = symbol

                # handle case in which we have a prolongation sign "_"
                else:
                    step_counter += 1
            return part

        p0 = parse_to_part(p0, cantus)
        p1 = parse_to_part(p1, counter)
        s.insert(0, p0)
        s.insert(0, p1)
        # write the m21 stream to a midi file
        s.write(format, file_name)


if __name__ == "__main__":
    cpg = CounterPointGenerator()
    file_path = "cantus.mxl"
    cantus = m21.converter.parse(file_path).parts[0]
    cantus_processed = preprocess_song(cantus, i=4, sub_dir="test")
    melody = cpg.generate_counterpoint(cantus_processed, SEQUENCE_LENGTH, 0.1)
    print(melody)
    cpg.save_melody(melody, file_name="melody.mid")
