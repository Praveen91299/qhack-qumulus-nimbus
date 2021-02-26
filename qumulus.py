import numpy as np
import sys
from scipy.io.wavfile import write
import pennylane as qml

def get_wave(freq, duration=0.5):
    amplitude = 4096
    t = np.linspace(0, duration, int(samplerate * duration))
    wave = amplitude * np.sin(2 * np.pi * freq * t) * (np.cos(2* np.pi * t / duration - np.pi) + 1)/2
    return wave

def get_piano_notes():
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
    base_freq = 261.63 #Frequency of Note C4
    note_freqs = {octave[i]: base_freq * pow(2,(i/12)) for i in range(len(octave))}        
    note_freqs[''] = 0.0 # silent note
    return note_freqs

def get_song_data(music_notes, tempo):
    note_freqs = get_piano_notes() # Function that we made earlier
    song = [get_wave(note_freqs[note], 60/tempo) for note in music_notes.split('-')]
    song = np.concatenate(song)
    return song

def numbers_to_notes(num_list, mapping):
    notes = ''
    for a in num_list[1:]:
        notes =  notes + '-' +str(mapping[a])
    return notes[1:]

def notes_to_numbers(notes, mapping):
    num_list = [mapping[note] for note in notes.split('-')]
    return num_list
