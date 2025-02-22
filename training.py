import mido
from mido import MidiFile, MidiTrack, Message
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


def extract_midi_tracks(file_path):
    """
    Loads a MIDI file and extracts its tracks.

    Parameters:
        file_path (str): Path to the MIDI file.

    Returns:
        dict: A dictionary where keys are track names (or indices) and values are lists of MIDI messages.
    """
    print(f"Loading file {file_path}")
    midi = MidiFile(file_path)

    tracks = {}

    for i, track in enumerate(midi.tracks):
        track_name = track.name if track.name else f"Track {i}"
        tracks[track_name] = [msg for msg in track]

    return tracks


def build_cnn_model(input_shape):
    """
    Builds a convolutional neural network model using Keras.

    Parameters:
        input_shape (tuple): Shape of the input data.

    Returns:
        keras.Model: Compiled CNN model.
    """
    model = keras.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, train_data, train_labels, epochs=10, batch_size=32):
    """
    Trains the CNN model on the given training data.

    Parameters:
        model (keras.Model): The compiled Keras model.
        train_data (np.ndarray): Training input data.
        train_labels (np.ndarray): Corresponding labels for training data.
        epochs (int, optional): Number of training epochs. Default is 10.
        batch_size (int, optional): Batch size for training. Default is 32.

    Returns:
        keras.callbacks.History: Training history object.
    """
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history


def build_harmony_model(input_shape):
    """
    Builds and trains a model to harmonize tracks based on the first track.

    Parameters:
        input_shape (tuple): Shape of the input data.

    Returns:
        keras.Model: Trained harmony model.
    """
    model = keras.Sequential([
        layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(input_shape[0], activation='linear')  # Output harmonized track
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


def harmonize_tracks(model, first_track, other_tracks):
    """
    Harmonizes the given tracks to the first track using the trained model.

    Parameters:
        model (keras.Model): The trained harmony model.
        first_track (np.ndarray): The primary melody track.
        other_tracks (list of np.ndarray): Other tracks to harmonize.

    Returns:
        list of MidiTrack: Harmonized MIDI tracks.
    """
    harmonized_tracks = []

    for track in other_tracks:
        harmonized_notes = []
        for i in range(len(track)):
            input_data = first_track[i:i + 1].reshape(1, -1, 1)
            predicted_notes = model.predict(input_data)[0]
            for note_value in predicted_notes:
                harmonized_notes.append(Message('note_on', note=int(note_value * 127), velocity=64, time=120))

        midi_track = MidiTrack()
        midi_track.extend(harmonized_notes)
        harmonized_tracks.append(midi_track)

    return harmonized_tracks


def create_training_data(files):
    """
    Prepares training data from multiple MIDI tracks for generating new tracks.

    Parameters:
        tracks (dict): Dictionary of extracted MIDI tracks.

    Returns:
        tuple: (training_data, training_labels) as NumPy arrays.
    """
    sequences = []
    labels = []

    for file in files:
        notes = [msg.note for msg in file[0].values if msg.type == 'note_on']
        if len(notes) < 2:
            continue

        for i in range(len(notes) - 1):
            sequences.append(notes[i])
            labels.append(notes[i + 1])

    sequences = np.array(sequences)
    labels = np.array(labels)

    return sequences, labels


def load_midi_from_folder(folder_path):
    """
    Loads all MIDI files from a folder and extracts their tracks.

    Parameters:
        folder_path (str): Path to the folder containing MIDI files.

    Returns:
        list: A list of extracted MIDI tracks from all files.
    """
    all_tracks = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                file_path = os.path.join(root, file)
                tracks = extract_midi_tracks(file_path)
                all_tracks.append(tracks)

    return all_tracks

from tensorflow.keras.utils import to_categorical


def create_training_data_generator(folder_path, sequence_length=32, batch_size=64):
    """
    Generator function to create training data from MIDI files with one-hot encoding.

    Parameters:
        folder_path (str): Path to the folder containing MIDI files.
        sequence_length (int): Number of previous notes used to predict the next note.
        batch_size (int): Number of sequences per batch.

    Yields:
        tuple: (batch_sequences, batch_labels) where sequences are one-hot encoded.
    """
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                 f.endswith('.mid') or f.endswith('.midi')]

    # MIDI note range is 0-127, so we use 128 for one-hot encoding
    num_notes = 128

    while True:  # Infinite loop for generator
        np.random.shuffle(all_files)  # Shuffle files for randomness
        sequences = []
        labels = []

        for file_path in all_files:
            midi = MidiFile(file_path)
            for track in midi.tracks:
                notes = []

                for msg in track:
                    if msg.type == 'note_on':  # Only take note_on messages
                        notes.append(msg.note)

                if len(notes) < sequence_length + 1:
                    continue  # Skip short tracks

                # Convert notes to one-hot encoding
                for i in range(len(notes) - sequence_length):
                    seq = notes[i:i + sequence_length]  # Input sequence
                    label = notes[i + sequence_length]  # Next note (label)

                    # Convert to one-hot
                    seq_one_hot = np.zeros((sequence_length, num_notes))
                    for j, note in enumerate(seq):
                        seq_one_hot[j, note] = 1  # Set one-hot for each note

                    sequences.append(seq_one_hot)
                    labels.append(label)  # Store label as an integer (not one-hot)

                    # Yield batch when batch size is reached
                    if len(sequences) >= batch_size:
                        yield np.array(sequences), np.array(labels)
                        sequences = []
                        labels = []





if __name__ == '__main__':
    subset = 'Q1'
    midis = load_midi_from_folder(f'dataset/{subset}')
    sequences, labels = create_training_data(midis, 32)
    model = build_cnn_model(sequences.shape[1:])
    train_model(model, sequences, labels)
    model.save(f'model{subset}.keras')

