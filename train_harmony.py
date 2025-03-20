import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dense, Flatten, TimeDistributed, Reshape
from mido import MidiFile
from tensorflow.keras.utils import to_categorical


def midi_to_sequences(root_dir, seq_length=64, num_notes=128, batch_size=32):
    def file_generator(root_dir):
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.mid'):
                    yield os.path.join(root, file)

    X_batch, Y_batch = [], []

    for midi_path in file_generator(root_dir):
        midi = MidiFile(midi_path)
        tracks = [list() for _ in range(len(midi.tracks))]

        for i, track in enumerate(midi.tracks):
            time = 0
            for msg in track:
                time += msg.time
                if msg.type == 'note_on' or msg.type == 'note_off':
                    tracks[i].append((msg.note, msg.velocity, time))

        # Ensure at least 5 tracks are available
        if len(tracks) < 5:
            continue

        input_seq = np.zeros((len(tracks[0]), num_notes))
        output_seqs = np.zeros((4, len(tracks[0]), num_notes))

        for i, event in enumerate(tracks[0]):
            input_seq[i, event[0]] = 1  # One-hot encode notes

        for track_idx in range(4):
            for i, event in enumerate(tracks[track_idx + 1]):
                if i < len(tracks[0]):  # Ensure alignment with input_seq length
                    output_seqs[track_idx, i, event[0]] = 1  # One-hot encode notes

        for i in range(len(input_seq) - seq_length):
            X = input_seq[i:i + seq_length]
            Y = output_seqs[:, i:i + seq_length, :]
            X_batch.append(X)
            Y_batch.append(Y)

            if len(X_batch) == batch_size:
                yield np.array(X_batch), np.array(Y_batch)
                X_batch, Y_batch = [], []

    if X_batch:
        yield np.array(X_batch), np.array(Y_batch)

# Define Model
seq_length = 64
num_notes = 128

model = keras.Sequential([
    Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(seq_length, num_notes)),
    Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
    Flatten(),
    Dense(4 * seq_length * num_notes, activation='softmax'),  # Adjust output size
    Reshape((4, seq_length, num_notes))  # Correctly reshape output
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model
midi_root_dir = 'dataset/'  # Replace with actual folder path

for dataset in ['Q1','Q2','Q3','Q4']:
    generator = midi_to_sequences(midi_root_dir+dataset)
    model.fit(generator, epochs=5, steps_per_epoch=5000, batch_size=32)
    model.save(f'harmony_{dataset}.keras')
