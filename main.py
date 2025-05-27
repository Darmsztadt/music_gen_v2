import numpy as np
import mido
import pygame.midi
from mido import MidiFile, MidiTrack, Message, MetaMessage
from pygame import mixer
from tensorflow import keras
import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from midi2audio import FluidSynth
from pydub import AudioSegment
import threading


# --- Core Functions ---

def load_models(model_folder, subset):
    models = {
        'base': keras.models.load_model(os.path.join(model_folder, f'model_{subset}.keras')),
        'harmony': keras.models.load_model(os.path.join(model_folder, f'harmony_{subset}_10x10000x32.keras')),
        'parametric': keras.models.load_model(os.path.join(model_folder, f'parametric_{subset}.keras'))
    }
    return models


def generate_base_melody(base_model, sequence_length=32, total_length=64, num_notes=128):
    seed_notes = np.random.randint(0, num_notes, size=sequence_length)
    generated_notes = list(seed_notes)

    while len(generated_notes) < total_length:
        current_input = np.zeros((1, sequence_length, num_notes))
        input_notes = []
        for i in range(sequence_length):
            idx = len(generated_notes) - sequence_length + i
            note = generated_notes[idx] if idx >= 0 else seed_notes[i]
            current_input[0, i, note] = 1
            input_notes.append(note)

        # Print input notes for this step
        print(f"Current input notes (step {len(generated_notes)}): {input_notes}")

        predicted = base_model.predict(current_input, verbose=0)
        predicted_notes = np.argmax(predicted[0], axis=-1)

        remaining = total_length - len(generated_notes)
        generated_notes.extend(predicted_notes[:remaining].tolist())

    return np.array(generated_notes[sequence_length:total_length])




def generate_harmonies(harmony_model, melody_notes, num_notes=128, chunk_size=64, progress=None):
    """
    Split melody into chunks of 64 notes and predict harmonies per chunk.
    Returns a list of harmony note arrays.
    """

    num_chunks = len(melody_notes) // chunk_size
    harmony_sequences = []

    for i in range(num_chunks):
        chunk = melody_notes[i * chunk_size: (i + 1) * chunk_size]

        # One-hot encode chunk: shape (1, 64, 128)
        melody_one_hot = np.zeros((1, chunk_size, num_notes))
        for j, note in enumerate(chunk):
            melody_one_hot[0, j, note] = 1

        # Predict harmony for this chunk
        harmony_output = harmony_model.predict(melody_one_hot, verbose=0)

        harmony_notes = np.argmax(harmony_output[0], axis=-1)

        harmony_notes = np.squeeze(harmony_notes)  # Make sure it's (64,)

        harmony_sequences.append(harmony_notes)

        if progress:
            progress["value"] += (30 / num_chunks)
            progress.update()

    return harmony_sequences


def predict_parameters(parametric_model, melody_notes, num_notes=128, sequence_length=32, split_length=8,
                       progress=None):
    """
    Predict parameters (avg_length, avg_velocity) from melody notes.
    Splits the sequence into chunks of 8 for prediction.
    """
    if sequence_length % split_length != 0:
        raise ValueError(f"Sequence length {sequence_length} must be divisible by split length {split_length}.")

    splits = sequence_length // split_length
    avg_lengths = []
    avg_velocities = []

    for i in range(splits):
        chunk = melody_notes[i * split_length: (i + 1) * split_length]

        # One-hot encode (1, 8, 128)
        melody_one_hot = np.zeros((1, split_length, num_notes))
        for j, note in enumerate(chunk):
            melody_one_hot[0, j, note] = 1

        # Predict
        param_output = parametric_model.predict(melody_one_hot, verbose=0)
        avg_length, avg_velocity = param_output[0]

        avg_lengths.append(max(avg_length, 0.01))
        avg_velocities.append(np.clip(avg_velocity * 127, 10, 127))

        if progress:
            progress["value"] += 5  # Update progress after each split
            progress.update()

    # Average results
    final_avg_length = np.mean(avg_lengths)
    print(final_avg_length)
    final_avg_velocity = np.mean(avg_velocities)
    print(final_avg_velocity)

    return final_avg_length, final_avg_velocity


def create_track(notes, avg_length, avg_velocity, instrument, ticks_per_beat=480):
    track = MidiTrack()
    track.append(Message('program_change', program=instrument, time=0))
    time_per_note = int(avg_length * ticks_per_beat)

    for note in notes:
        jitter = random.randint(-10, 10)
        time = max(time_per_note + jitter, 1)
        velocity = int(float(avg_velocity))  # Ensures it's a scalar
        track.append(Message('note_on', note=int(note), velocity=velocity, time=0))
        track.append(Message('note_off', note=int(note), velocity=0, time=time))

    return track


def generate_midi(models, output_path, sequence_length=32, total_length=64, num_notes=128,
                  progress=None):
    base_model = models['base']
    harmony_model = models['harmony']
    parametric_model = models['parametric']

    melody_notes = generate_base_melody(base_model, sequence_length, total_length, num_notes)
    harmony_notes = generate_harmonies(harmony_model, melody_notes, num_notes, chunk_size=64, progress=progress)
    avg_length, avg_velocity = predict_parameters(
        parametric_model=parametric_model,
        melody_notes=melody_notes,
        num_notes=128,
        sequence_length=32,
        split_length=8,
        progress=progress
    )

    mid = MidiFile(ticks_per_beat=480)
    tempo_bpm = random.randint(70, 160)
    tempo_us_per_beat = int(60_000_000 / tempo_bpm)

    mid.tracks.append(MidiTrack([MetaMessage('set_tempo', tempo=tempo_us_per_beat)]))

    # Melody Track
    melody_instrument = 0  # Acoustic Grand Piano
    melody_track = create_track(melody_notes, avg_length, avg_velocity, melody_instrument)
    mid.tracks.append(melody_track)

    # Harmony Tracks
    harmony_instruments = [48, 24, 60, 71]  # Strings, Guitar, Horn, Clarinet
    for i, h_notes in enumerate(harmony_notes):
        h_notes = np.ravel(h_notes)
        instrument = harmony_instruments[i % len(harmony_instruments)]
        track = create_track(h_notes, avg_length, avg_velocity, instrument)
        mid.tracks.append(track)

    # Remove the last track (if needed)
    if len(mid.tracks) > 1:
        mid.tracks.pop()

    mid.save(output_path)

    if progress:
        progress["value"] = 100
    return tempo_bpm, melody_instrument


# --- GUI Functions ---

class MidiGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎹 Multi-Track MIDI Generator")
        self.model_folder = ''
        self.output_folder = ''
        self.selected_subset = tk.StringVar(value="Q1")
        self.last_output_path = None
        self.mp3_output_path = None
        self.soundfont_path = "GeneralUser-GS.sf2"
        self.selected_mp3_path = None

        # Initialize MIDI
        self.setup_midi()

        # --- Section: Model and Output Folder Selection ---
        tk.Label(root, text="--- Model & Output Folders ---", font=('Arial', 10, 'bold')).pack(pady=5)

        tk.Button(root, text="Select Model Folder", command=self.select_model_folder, width=30).pack(pady=2)
        self.selected_model_label = tk.Label(root, text="Model Folder: None", fg="gray")
        self.selected_model_label.pack()

        tk.Button(root, text="Select Output Folder", command=self.select_output_folder, width=30).pack(pady=2)
        self.selected_output_label = tk.Label(root, text="Output Folder: None", fg="gray")
        self.selected_output_label.pack()

        # --- Section: Model Subset & Length ---
        tk.Label(root, text="\n--- Model Settings ---", font=('Arial', 10, 'bold')).pack(pady=5)
        tk.Label(root, text="Choose Model Subset (Q1–Q4):").pack()
        self.dropdown = ttk.Combobox(root, textvariable=self.selected_subset, values=["Q1", "Q2", "Q3", "Q4"], width=27)
        self.dropdown.pack(pady=2)

        tk.Label(root, text="Enter Total Length (Number of 64-length segments):").pack()
        self.total_length = tk.Entry(root, width=30)
        self.total_length.pack(pady=2)

        tk.Button(root, text="Generate MIDI", command=self.generate, bg='green', fg='white', width=30).pack(pady=10)

        # --- Section: Preview Existing MIDI ---
        tk.Label(root, text="\n--- Preview Existing MIDI ---", font=('Arial', 10, 'bold')).pack(pady=5)

        tk.Button(root, text="Select MIDI File", command=self.select_existing_midi, bg='blue', fg='white',
                  width=30).pack(pady=2)
        self.selected_midi_label = tk.Label(root, text="Selected MIDI: None", fg="gray")
        self.selected_midi_label.pack()
        tk.Button(root, text="Select SoundFont (.sf2)", command=self.select_soundfont, width=30).pack(pady=2)
        self.selected_sf_label = tk.Label(root, text="SoundFont: default (soundfont.sf2)", fg="gray")
        self.selected_sf_label.pack()

        tk.Button(root, text="Select MP3 Output Path", command=self.select_mp3_path, width=30).pack(pady=2)
        tk.Button(root, text="Convert MIDI to MP3", command=self.convert_midi_to_mp3, bg='orange', fg='black',
                  width=30).pack(pady=5)
        tk.Button(root, text="Select MP3 File", command=self.select_existing_mp3, bg='purple', fg='white',
                  width=30).pack(pady=2)
        self.selected_mp3_label = tk.Label(root, text="Selected MP3: None", fg="gray")
        self.selected_mp3_label.pack()

        tk.Button(root, text="Preview Selected MP3", command=self.preview_midi, bg='blue', fg='white', width=30).pack(
            pady=5)

    def setup_midi(self):
        pygame.midi.init()
        self.player = pygame.midi.Output(0)

    def close_midi(self):
        if hasattr(self, 'player'):
            self.player.close()
        pygame.midi.quit()

    def select_model_folder(self):
        self.model_folder = filedialog.askdirectory(title="Select Model Folder")
        if self.model_folder:
            self.selected_model_label.config(text=f"Model Folder: {self.model_folder}", fg="black")

    def select_output_folder(self):
        self.output_folder = filedialog.askdirectory(title="Select Output Folder")
        if self.output_folder:
            self.selected_output_label.config(text=f"Output Folder: {self.output_folder}", fg="black")

    def select_existing_midi(self):
        midi_file = filedialog.askopenfilename(title="Select MIDI File", filetypes=[("MIDI Files", "*.mid;*.midi")])
        if midi_file:
            self.last_output_path = midi_file
            self.selected_midi_label.config(text=f"Selected MIDI: {os.path.basename(midi_file)}", fg="black")

    def preview_midi(self):
        if not self.selected_mp3_path:
            messagebox.showwarning("Warning", "No MP3 file selected to preview.")
            return

        try:
            mixer.init()
            mixer.music.load(self.selected_mp3_path)
            mixer.music.play()
        except Exception as e:
            messagebox.showerror("Error", f"Could not play MP3: {e}")

    def generate(self):
        if not self.model_folder or not self.output_folder:
            messagebox.showwarning("Error", "Please select both folders first!")
            return

        try:
            total_len_input = int(self.total_length.get())
            total_len = total_len_input * 64
            if total_len_input <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Error", "Total length must be a positive integer.")
            return

        subset = self.selected_subset.get()
        models = load_models(self.model_folder, subset)

        output_path = os.path.join(self.output_folder, f'generated_{subset}_{random.randint(1000, 9999)}.mid')
        tempo, instrument = generate_midi(models, output_path, total_length=total_len)

        self.last_output_path = output_path
        self.selected_midi_label.config(text=f"Selected MIDI: {os.path.basename(output_path)}", fg="black")

        messagebox.showinfo("Success",
                            f"MIDI generated!\nSaved at: {output_path}\nTempo: {tempo} bpm\n")

    def select_mp3_path(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3 files", "*.mp3")],
            title="Select MP3 Output Path"
        )
        if file_path:
            self.mp3_output_path = file_path
            messagebox.showinfo("MP3 Path", f"MP3 will be saved to:\n{file_path}")

    def convert_midi_to_mp3(self):
        try:
            fs = FluidSynth(self.soundfont_path)
            fs.midi_to_audio(self.last_output_path, self.mp3_output_path)
            messagebox.showinfo("Success", f"WAV saved at:\n{self.mp3_output_path}")
        except Exception as e:
            messagebox.showerror("Conversion Error", str(e))

    def select_soundfont(self):
        sf2_path = filedialog.askopenfilename(
            title="Select SoundFont (.sf2)",
            filetypes=[("SoundFont Files", "*.sf2")]
        )
        if sf2_path:
            self.soundfont_path = sf2_path
            self.selected_sf_label.config(text=f"SoundFont: {os.path.basename(sf2_path)}", fg="black")

    def select_existing_mp3(self):
        mp3_file = filedialog.askopenfilename(title="Select MP3 File", filetypes=[("MP3 Files", "*.mp3")])
        if mp3_file:
            self.selected_mp3_path = mp3_file
            self.selected_mp3_label.config(text=f"Selected MP3: {os.path.basename(mp3_file)}", fg="black")


# --- MAIN ---
if __name__ == "__main__":
    root = tk.Tk()
    app = MidiGeneratorApp(root)
    root.mainloop()
