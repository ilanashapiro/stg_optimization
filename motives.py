import sys
sys.path.append('/Users/ilanashapiro/Documents/motif_discovery') # https://github.com/Tsung-Ping/motif_discovery
sys.path.append('/Users/ilanashapiro/Documents/Harmony-Transformer') # https://github.com/Tsung-Ping/Harmony-Transformer -- probably not using this

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import mido
import music21
import csv
import numpy as np
import SIA
import pretty_midi

def midi_to_csv_in_ticks():
	filename = "LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short.mid" # for now
	output_filename = filename[:-4] + ".csv"

	print("Converting " + filename + " to " + output_filename)

	mid = mido.MidiFile(filename)
	df = pd.DataFrame(columns=["onset", "pitch", "duration"])

	# Default MIDI tempo is 500,000 microseconds per beat
	# This can change throughout the piece and needs to be accounted for
	microseconds_per_beat = 500000  # Default value

	for i, track in enumerate(mid.tracks):
		if i % 2 == 0: # for these files, by inspecting them I find that tracks 1 and 2 are sequential, as are 3 and 4. But 1/2 and 3/4 are simulatneous so we reset at track 3
			absolute_time_in_ticks = 0
			absolute_time_in_seconds = 0
		note_ontimes_dict = {} 

		for msg in track:
			if msg.type == 'set_tempo':
				# If there's a tempo change, update the microseconds per beat
				microseconds_per_beat = msg.tempo

			absolute_time_in_ticks += msg.time  # Update absolute time with delta time
			absolute_time_in_seconds = mido.tick2second(absolute_time_in_ticks, mid.ticks_per_beat, microseconds_per_beat)
			
			if msg.type == 'note_on' and msg.velocity > 0:
				note_ontimes_dict[msg.note] = (absolute_time_in_seconds, msg.channel)

			elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and msg.note in note_ontimes_dict:
				onset_seconds, channel = note_ontimes_dict.pop(msg.note)
				duration_seconds = absolute_time_in_seconds - onset_seconds
				staff_num = channel + 1  # Assign the staff number based on the MIDI channel
				new_row = pd.DataFrame([[round(onset_seconds, 3), msg.note, round(duration_seconds, 3), staff_num]], columns=["onset", "pitch", "duration", "staff"]) 
				df = pd.concat([df, new_row], axis=0) 

	df.to_csv(output_filename, index=False) 
	print(f"Data has been written to {output_filename} in ticks")
# midi_to_csv_in_ticks()

# CSV format from https://github.com/Wiilly07/Beethoven_motif 
# code modified from https://github.com/andrewchenk/midi-csv/blob/master/midi_to_csv.py
def midi_to_csv_in_crochets():
	directory = "/Users/ilanashapiro/Documents/constraints_project/LOP_database_06_09_17/liszt_classical_archives"
	def process_midi(file_path):
		output_filepath = file_path[:-4] + "_data.csv"
		print("Converting " + file_path + " to " + output_filepath)
		
		mf = music21.midi.MidiFile()
		mf.open(file_path)
		mf.read()
		mf.close()
		
		s = music21.midi.translate.midiFileToStream(mf, quantizePost=False).flatten() #quantize is what rounds all note durations to real music note types, not needed for our application
		
		df = pd.DataFrame(columns=["onset", "pitch", "duration", "staff"])
		for g in s.recurse().notes:
			if g.isChord:
				for pitch in g.pitches: 
					x = music21.note.Note(pitch.midi, duration=g.duration, offset=g.offset)
					s.insert(x)
		for note in s.recurse().notes: 
			if note.isNote:
				pitch = note.pitch.midi
				onset = round(float(note.offset), 3)  # The offset in quarter notes # IN CROCHETS
				duration = round(float(note.duration.quarterLength), 3)  # Duration in quarter notes
				
				staff = note.staff if hasattr(note, 'staff') else 0 # default will be zero (top staff)
				new_row = pd.DataFrame([[onset, pitch, duration, staff]], columns=["onset", "pitch", "duration", "staff"])
				df = pd.concat([df, new_row], axis=0) 

		df.to_csv(output_filepath, index=False)
		print(f"Data has been written to {output_filepath} in crochets")

	with ThreadPoolExecutor() as executor:
		futures = []
		for root, _, files in os.walk(directory):
			for filename in files:
				# and we haven't already made the CSV file (_data extension since LOP already has CSV files with metadata and we don't want to overwrite)
				if filename.endswith("_solo_short.mid"):# and not any("_data.csv" in file for file in files): 
					file_path = os.path.join(root, filename)
					process_midi(file_path)
					future = executor.submit(process_midi, file_path)
					futures.append(future)
# midi_to_csv_in_crochets()		

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

# as per https://www.music-ir.org/mirex/wiki/2017:Discovery_of_Repeated_Themes_%26_Sections
def write_mirex_motives(motives, out_file):
	out_str = ""
	for idx_p, pattern in enumerate(motives):
		out_str += "pattern" + str(idx_p+1) + "\n" # + 1 because of zero-indexing
		for idx_o, occurrence in enumerate(pattern):
			out_str += "occurrence" + str(idx_o+1) + "\n"
			for ontime, pitch in occurrence:
				out_str += format(ontime, '.5f') + ", " + format(pitch, '.5f') + "\n"
		out_str += "\n"
	with open(out_file, "w") as f:
			f.write(out_str[:-2])

def get_motives():
	directory = "/Users/ilanashapiro/Documents/constraints_project/LOP_database_06_09_17/liszt_classical_archives/0_short_test"
	futures = []
	def process_file(file_path):
		notes = load_notes_csv(file_path)
		motives = SIA.find_motives(notes)
		# motives_test = [[[(174., 84.), (174.5, 57.), (175., 52.), (175.5, 54.)], [(178., 79.), (178.5, 52.), (179., 50.), (179.5, 48.)], [(186., 79.), (186.5, 52.), (187., 50.), (187.5, 48.)], [(194., 79.), (194.5, 52.), (195., 50.), (195.5, 48.)], [(198., 79.), (198.5, 52.), (199., 50.), (199.5, 48.)]], [[(174.5, 57.), (175., 52.), (175.5, 54.), (176., 55.)], [(180., 83.), (180.5, 79.), (181., 79.), (181.5, 79.)], [(188., 83.), (188.5, 79.), (189., 79.), (189.5, 79.)]]]
		write_mirex_motives(motives, file_path[:-9] + "_motives_seconds.txt") # "_data.csv" has length 9
	
	with ThreadPoolExecutor() as executor:
		for root, _, files in os.walk(directory):
			for filename in files:
				if filename.endswith(".csv"):
					file_path = os.path.join(root, filename)
					future = executor.submit(process_file, file_path)
					futures.append(future)
		# as_completed() provides a generator that yields futures as they complete
		# for future in as_completed(futures):
		# 		motives = future.result()
				
get_motives()