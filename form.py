# Simple MSAF example
# from __future__ import print_function
import sys
sys.path.append('/Users/ilanashapiro/Documents/msaf0_1_80')
sys.path.append('/Users/ilanashapiro/Library/musicaiz-0.1.2')
sys.path.append('/Users/ilanashapiro/Documents/MIDI-BERT') # https://github.com/wazenmai/MIDI-BERT
sys.path.append('/Users/ilanashapiro/Documents/midi_melody_extraction') # https://github.com/bytedance/midi_melody_extraction
sys.path.append('/Users/ilanashapiro/Documents/motif_discovery') # https://github.com/Tsung-Ping/motif_discovery
sys.path.append('/Users/ilanashapiro/Documents/audio_to_midi_melodia') # https://github.com/justinsalamon/audio_to_midi_melodia
sys.path.append('/Users/ilanashapiro/Documents/functional-harmony') # https://github.com/Tsung-Ping/functional-harmony
sys.path.append('/Users/ilanashapiro/Documents/Harmony-Transformer') # https://github.com/Tsung-Ping/Harmony-Transformer

import msaf
from midi2audio import FluidSynth
import os
import pandas as pd
import mido
import music21
import csv
import numpy as np
import SIA

def midi_to_csv_in_ticks():
	filename = "LOP_database_06_09_17/liszt_classical_archives/0/bl11_solo.mid" # for now
	output_filename = filename[:-4] + ".csv"
	print("Converting " + filename + " to " + output_filename)
	mid = mido.MidiFile(filename)
	df = pd.DataFrame(columns=["onset", "pitch", "duration"])

	for _, track in enumerate(mid.tracks):
		absolute_time = 0 
		note_on_times = {} 
		for msg in track:
			absolute_time += msg.time  # Update absolute time with delta time

			if msg.type == 'note_on' and msg.velocity > 0:
				note_on_times[msg.note] = absolute_time

			elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and msg.note in note_on_times:
				start_time = note_on_times.pop(msg.note)
				duration = absolute_time - start_time
				new_row = pd.DataFrame([[start_time, msg.note, duration]], columns=["onset", "pitch", "duration"]) 
				df = pd.concat([df, new_row], axis=0) 

	df.to_csv(output_filename, index=False) # if I use ticks, TODO: need to add staff numbers
	print(f"Data has been written to {output_filename} in ticks")

# CSV format from https://github.com/Wiilly07/Beethoven_motif 
# code modified from https://github.com/andrewchenk/midi-csv/blob/master/midi_to_csv.py
def midi_to_csv_in_crochets():
	filename = "LOP_database_06_09_17/liszt_classical_archives/0/bl11_solo.mid" # for now
	output_filename = filename[:-4] + ".csv"
	print("Converting " + filename + " to " + output_filename)
	
	mf = music21.midi.MidiFile()
	mf.open(filename)
	mf.read()
	mf.close()
	
	s = music21.midi.translate.midiFileToStream(mf, quantizePost=False).flatten() #quantize is what rounds all note durations to real music note types, not needed for our application

	df = pd.DataFrame(columns=["onset", "pitch", "duration"])
	for g in s.recurse().notes:
		if g.isChord:
			for pitch in g.pitches: 
				x = music21.note.Note(pitch.midi, duration=g.duration, offset=g.offset)
				s.insert(x)
	for note in s.recurse().notes: 
		if note.isNote:
			onset = round(float(note.offset), 3)  # The offset in quarter notes
			pitch = note.pitch.midi
			duration = round(float(note.duration.quarterLength), 3)  # Duration in quarter notes
			staff = note.staff if hasattr(note, 'staff') else 0 # default will be zero (top staff)
			new_row = pd.DataFrame([[onset, pitch, duration, staff]], columns=["onset", "pitch", "duration", "staff"])
			df = pd.concat([df, new_row], axis=0) 

	df.to_csv(output_filename, index=False)
	print(f"Data has been written to {output_filename} in crochets")

# code modified from https://github.com/Tsung-Ping/motif_discovery/blob/main/experiments.py 
def load_notes_csv(filename):
	dt = [
		('onset', float), # onset (in crotchet beats)
		('pitch', float), # MIDI note number
		('duration', float), # duration (in crotchet beats)
		('staff', int) # staff number (integers from zero for the top staff)
	] # datatype

	# Format data as structured array
	with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		notes = np.array([tuple([float(x) for x in row]) for i, row in enumerate(reader) if i > 0], dtype=dt)

	# Get unique notes irrespective of 'staffNum'
	_, unique_indices = np.unique(notes[['onset', 'pitch']], return_index=True)
	notes = notes[unique_indices]
	print('deleted notes:', [i for i in range(notes.size) if i not in unique_indices])

	notes = notes[notes['duration'] > 0]
	return np.sort(notes, order=['onset', 'pitch'])

# midi_to_csv_in_crochets()
def get_motives():
	notes = load_notes_csv("LOP_database_06_09_17/liszt_classical_archives/0/bl11_solo.csv") # for now
	print(SIA.find_motives(notes))

def convert_dataset_midi_to_wav():
	soundfont_filepath = "GeneralUser GS 1.471/GeneralUser GS v1.471.sf2"
	fs = FluidSynth(sound_font=soundfont_filepath)
	directory = "LOP_database_06_09_17"
	for root, _, files in os.walk(directory):
		for filename in files:
			if filename.endswith("_solo.mid"):
				file_path = os.path.join(root, filename)
				fs.midi_to_audio(file_path, file_path[:-3] + "wav")
				print("Converted", file_path, "to WAV")

def segment_audio_MSAF():
	audio_filepath = "LOP_database_06_09_17/liszt_classical_archives/0/bl11_solo.wav"

	# hierarchical: scluster, olda (boundaries only, use with fmc2d), vmo
	# sf (flat, boundaries only) has best performance
	boundaries, labels = msaf.process(audio_filepath, boundaries_id="scluster", labels_id="scluster", hier=True)
	print(boundaries)
	print(labels)

	out_file = audio_filepath[:-4] + '_segments.txt'
	print('Saving output to %s' % out_file)
	msaf.io.write_mirex_hierarchical(boundaries, labels, out_file)

# convert_dataset_midi_to_wav()
# segment_audio_MSAF()

# print(msaf0_1_80.features_registry)
# print(msaf0_1_80.get_all_boundary_algorithms())
# print(msaf0_1_80.get_all_label_algorithms())

