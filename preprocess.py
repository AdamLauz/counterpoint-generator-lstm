import os
import json
import music21 as m21
import numpy as np
import tensorflow.keras as keras
from typing import Tuple, List

MXL_DATASET_PATH = "cmidi/Bach"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET_CANTUS = "cantus_file_dataset"
SINGLE_FILE_DATASET_COUNTER = "counter_file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 65
SONGS_TO_LOAD = 300

# durations are expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25,  # 16th note
    0.5,  # 8th note
    0.75,
    1.0,  # quarter note
    1.5,
    2,  # half note
    3,
    4  # whole note
]


def load_songs_in_mxl(dataset_path: str) -> Tuple[List[m21.stream.Part], List[m21.stream.Part]]:
    """Loads all mxl pieces in dataset using music21.

    :param dataset_path: Path to dataset folder
    :return cantusfirmuses (list of m21 streams): List containing all pieces,
    counterpoints (list of m21 streams): List containing all pieces
    """
    cantusfirmuses = []
    counterpoints = []
    songs_loaded = 0
    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # consider only mxl files
            if file[-3:] == "mxl":
                song = m21.converter.parse(os.path.join(path, file))
                if len(song.parts) >= 2:
                    cantusfirmuses += [song.parts[0]]
                    counterpoints += [song.parts[1]]
                    songs_loaded += 1

            if songs_loaded % 10 == 0:
                print(f"{songs_loaded} songs loaded")

            # load up to SONGS_TO_LOAD
            if songs_loaded == SONGS_TO_LOAD:
                return cantusfirmuses, counterpoints

    return cantusfirmuses, counterpoints


def is_acceptable_song(song: m21.stream.Part, acceptable_durations: List[int]) -> bool:
    """Boolean routine that
    returns True if
    1. piece has all acceptable duration,
    2. piece doesn't have chords (only single melody line)
    False otherwise.

    :param song: music21 Part object
    :param acceptable_durations: list of acceptable durations in quarter length
    :return (bool):
    """
    for note in song.flat.notesAndRests:
        if note.isChord or (note.duration.quarterLength not in acceptable_durations):
            return False
    return True


def transpose(song: m21.stream.Part) -> m21.stream.Part:
    """Transposes song to C maj/A min

    :param song: Song to transpose
    :return transposed_song (m21 stream):
    """
    interval = m21.interval.Interval()
    # get key from the song
    measures_part0 = song.getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # get interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    transposed_song = song.transpose(interval)
    return transposed_song


def encode_song(song: m21.stream.Part, time_step: float = 0.25) -> str:
    """Converts a score into a time-series-like music representation. Each item in the encoded list represents 'min_duration'
    quarter lengths. The symbols used at each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
    for representing notes/rests that are carried over into a new time step. Here's a sample encoding:

        ["r", "_", "60", "_", "_", "_", "72" "_"]

    :param song (m21 stream): Piece to encode
    :param time_step (float): Duration of each time step in quarter length
    :return:
    """

    encoded_song = []

    for event in song.flat.notesAndRests:

        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi  # 60
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):

            # if it's the first time we see a note/rest, let's encode it. Otherwise, it means we're carrying the same
            # symbol in a new time step
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocess_song(song, i: int, sub_dir: str) -> str:
    """
    :param i: index of the song in the song list
    :param song: music21 object representing the song
    :param sub_dir: sub folder to save the preprocessed song under SAVE_DIR
    :return: encoded song (str)
    """

    # filter out songs that have non-acceptable durations
    if not is_acceptable_song(song, ACCEPTABLE_DURATIONS):
        return "failure"

    # transpose songs to Cmaj/Amin
    song = transpose(song)

    # encode songs with music time series representation
    encoded_song = encode_song(song)

    # save songs to text file
    song_dir = os.path.join(SAVE_DIR, sub_dir)

    if not os.path.exists(song_dir):
        os.makedirs(song_dir)

    save_path = os.path.join(song_dir, str(i))
    with open(save_path, "w") as fp:
        fp.write(encoded_song)

    return encoded_song


def preprocess(dataset_path: str):
    # load folk songs
    print("Loading songs...")
    cantusfirmuses, counterpoints = load_songs_in_mxl(dataset_path)
    print(f"Loaded {len(cantusfirmuses)} cantus firmus parts.")
    print(f"Loaded {len(counterpoints)} counter point parts.")

    for i, (cantus, counter) in enumerate(zip(cantusfirmuses, counterpoints)):
        cantus_encoded = preprocess_song(cantus, i, "cantus")
        counter_encoded = preprocess_song(cantus, i, "counter")
        if cantus_encoded:
            continue
        if counter_encoded:
            continue

        # filter out songs with un-equal length
        if len(cantus_encoded.split()) != len(counter_encoded.split()):
            continue

        if i % 10 == 0:
            print(f"Song {i} out of {len(cantusfirmuses)} processed")


def load(file_path) -> str:
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path: str, file_dataset_path: str, sequence_length: int) -> str:
    """Generates a file collating all the encoded songs and adding new piece delimiters.

    :param dataset_path (str): Path to folder containing the encoded songs
    :param file_dataset_path (str): Path to file for saving songs in single file
    :param sequence_length (int): # of time steps to be considered for training
    :return songs (str): String containing all songs in dataset + delimiters
    """

    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    # remove empty space from last character of string
    songs = songs[:-1]

    # save string that contains all the dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


def create_mapping(songs: str, mapping_path: str):
    """Creates a json file that maps the symbols in the song dataset onto integers

    :param songs (str): String with all songs
    :param mapping_path (str): Path where to save mapping
    :return:
    """
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save voabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs: str) -> List[int]:
    int_songs = []

    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # transform songs string to list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generate_training_sequences(sequence_length: int) -> Tuple[List[int], List[int]]:
    """Create input and output data samples for training. Each sample is a sequence.

    :param sequence_length (int): Length of each sequence. With a quantisation at 16th notes, 64 notes equates to 4 bars

    :return inputs (ndarray): Training inputs
    :return targets (ndarray): Training targets
    """

    # load songs and map them to int
    cantuses = load(SINGLE_FILE_DATASET_CANTUS)
    int_cantuses = convert_songs_to_int(cantuses)

    counters = load(SINGLE_FILE_DATASET_COUNTER)
    int_counters = convert_songs_to_int(counters)

    inputs = []
    targets = []

    # generate the training sequences
    num_sequences = len(int_cantuses) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_cantuses[i:i + sequence_length])
        targets.append(int_counters[i])

    # one-hot encode the sequences
    vocabulary_size = len(set(int_cantuses).union(int_counters))

    # inputs size: (# of sequences, sequence length, vocabulary size)
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    print(f"There are {len(inputs)} sequences.")

    return inputs, targets


def main():
    preprocess(MXL_DATASET_PATH)
    cantuses = create_single_file_dataset(os.path.join(SAVE_DIR, "cantus"), SINGLE_FILE_DATASET_CANTUS, SEQUENCE_LENGTH)
    counters = create_single_file_dataset(os.path.join(SAVE_DIR, "counter"), SINGLE_FILE_DATASET_COUNTER,
                                          SEQUENCE_LENGTH)
    create_mapping(cantuses + " " + counters, MAPPING_PATH)

    # inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
