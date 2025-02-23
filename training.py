import mido
from mido import MidiFile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


def build_cnn_model(input_shape, output_shape):
    """
    Builds a convolutional neural network model using Keras.

    Parameters:
        input_shape (tuple): Shape of the input data.
        output_shape (tuple): Shape of the output labels.

    Returns:
        keras.Model: Compiled CNN model.
    """
    model = keras.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(np.prod(output_shape), activation='softmax'),  # Output (8 * 128 classes)
        layers.Reshape(output_shape)  # Reshape to (8, 128)
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
                    seq = notes[i:i + sequence_length]  # Input sequence
                    label = notes[i + sequence_length:i + sequence_length + prediction_length]  # Next 8 notes

                    # Convert input to one-hot
                    seq_one_hot = np.zeros((sequence_length, num_notes))
                    for j, note in enumerate(seq):
                        seq_one_hot[j, note] = 1  # One-hot encode each note

                    # Convert labels to one-hot (shape: (8, 128))
                    label_one_hot = np.zeros((prediction_length, num_notes))
                    for j, note in enumerate(label):
                        label_one_hot[j, note] = 1  # One-hot encode each note

                    sequences.append(seq_one_hot)
                    labels.append(label_one_hot)

                    if len(sequences) >= batch_size:
                        yield np.array(sequences), np.array(labels)
                        sequences = []
                        labels = []


if __name__ == '__main__':
    subset = 'Q3'
    generator = create_training_data_generator(f'dataset/{subset}')
    first_batch = next(generator)  # Fetch first batch to determine shape
    model = build_cnn_model(first_batch[0].shape[1:], first_batch[1].shape[1:])  # Use correct input/output shape
    model.fit(generator, epochs=10)  # Train using generator
    model.save(f'model_{subset}.keras')
