import os
import numpy as np
import mido
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


def midi_generator(folder_path, sequence_length=8, batch_size=32):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mid')]
    num_notes = 128  # MIDI note range (0-127)
    while True:
        X, Y = [], []
        for file in files:
            mid = mido.MidiFile(file)
            track = mid.tracks[0] if mid.tracks else None

            if not track:
                continue

            notes = []
            note_lengths = []
            note_volumes = []

            time_elapsed = 0
            for msg in track:
                time_elapsed += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append(msg.note)
                    note_lengths.append(time_elapsed)
                    note_volumes.append(msg.velocity)
                    time_elapsed = 0

            if len(notes) < sequence_length + 1:
                continue

            max_velocity = 127  # MIDI velocity range
            max_length = max(note_lengths) if note_lengths else 1

            note_lengths = np.array(note_lengths) / max_length  # Normalize note lengths
            note_volumes = np.array(note_volumes) / max_velocity  # Normalize volumes

            one_hot_notes = to_categorical(notes, num_classes=num_notes)  # One-hot encode notes

            for i in range(len(notes) - sequence_length):
                X.append(one_hot_notes[i:i + sequence_length])
                avg_length = np.mean(note_lengths[i:i + sequence_length])
                avg_volume = np.mean(note_volumes[i:i + sequence_length])
                Y.append([avg_length, avg_volume])

                if len(X) >= batch_size:
                    yield np.array(X), np.array(Y)
                    X, Y = [], []


def build_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2, padding='same'),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2, padding='same'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)  # Output: avg note length and avg volume
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    dataset_path='dataset/'
    for subset in range(4):
        folder_path = f"{dataset_path}Q{subset+1}"  # Change this to your MIDI folder
        print(folder_path)
        sequence_length = 8
        batch_size = 32

        gen = midi_generator(folder_path, sequence_length, batch_size)
        model = build_model((sequence_length, 128))  # Updated input shape for one-hot encoding

        model.fit(gen, steps_per_epoch=500000, epochs=5)  # Adjust steps and epochs as needed
        model.save(f'parametric_Q{subset+1}.keras')
