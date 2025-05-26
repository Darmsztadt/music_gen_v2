import mido
from mido import MidiFile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


def build_lstm_model(input_shape, output_shape):
    """
    Builds an LSTM model using Keras.

    Parameters:
        input_shape (tuple): Shape of the input data (e.g., (sequence_length, 128)).
        output_shape (tuple): Shape of the output labels (e.g., (8, 128)).

    Returns:
        keras.Model: Compiled LSTM model.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(256, return_sequences=True),
        layers.LSTM(256),
        layers.Dense(512, activation='relu'),
        layers.Dense(np.prod(output_shape), activation='softmax'),
        layers.Reshape(output_shape)
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def create_training_data_generator(folder_path, sequence_length=32, prediction_length=8, batch_size=64):
    """
    Generator function to create training data from MIDI files with one-hot encoding.

    Parameters:
        folder_path (str): Path to the folder containing MIDI files.
        sequence_length (int): Number of previous notes used to predict the next notes.
        prediction_length (int): Number of future notes to predict.
        batch_size (int): Number of sequences per batch.

    Yields:
        tuple: (batch_sequences, batch_labels) where sequences and labels are one-hot encoded.
    """
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                 f.endswith('.mid') or f.endswith('.midi')]

    num_notes = 128  # MIDI note range

    while True:  # Infinite loop for generator
        np.random.shuffle(all_files)
        sequences = []
        labels = []

        for file_path in all_files:
            midi = MidiFile(file_path)
            for track in midi.tracks:
                notes = []

                for msg in track:
                    if msg.type == 'note_on':
                        notes.append(msg.note)

                if len(notes) < sequence_length + prediction_length:
                    continue  # Skip short tracks

                for i in range(len(notes) - sequence_length - prediction_length):
                    seq = notes[i:i + sequence_length]
                    label = notes[i + sequence_length:i + sequence_length + prediction_length]

                    # Convert input to one-hot
                    seq_one_hot = np.zeros((sequence_length, num_notes))
                    for j, note in enumerate(seq):
                        seq_one_hot[j, note] = 1

                    # Convert labels to one-hot
                    label_one_hot = np.zeros((prediction_length, num_notes))
                    for j, note in enumerate(label):
                        label_one_hot[j, note] = 1

                    sequences.append(seq_one_hot)
                    labels.append(label_one_hot)

                    if len(sequences) >= batch_size:
                        yield np.array(sequences), np.array(labels)
                        sequences = []
                        labels = []


if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)

    for subset in ['Q2', 'Q3', 'Q4']:
        generator = create_training_data_generator(f'dataset/{subset}')
        first_batch = next(generator)

        input_shape = first_batch[0].shape[1:]  # (sequence_length, 128)
        output_shape = first_batch[1].shape[1:]  # (8, 128)

        model = build_lstm_model(input_shape, output_shape)
        model.fit(generator, steps_per_epoch=100000, epochs=5)
        model.save(f'model_{subset}.keras')
