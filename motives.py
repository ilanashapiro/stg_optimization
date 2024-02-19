import sys
sys.path.append('/Users/ilanashapiro/Documents/motif_discovery') # https://github.com/Tsung-Ping/motif_discovery

import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import mido
import music21
import csv
import numpy as np
import SIA

# CSV format from https://github.com/Wiilly07/Beethoven_motif 
def midi_to_csv_mido():
	filename = "LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short.mid" # for now
	output_filename = filename[:-4] + ".csv"

	print("Converting " + filename + " to " + output_filename)

	mid = mido.MidiFile(filename)
	df = pd.DataFrame(columns=["onset", "onset_seconds", "pitch", "duration", "staff"])

	# Default MIDI tempo is 500,000 microseconds per beat
	# This can change throughout the piece and needs to be accounted for
	microseconds_per_beat = 500000  # Default value

	def ticks_to_crochets(ticks, ticks_per_beat):
		return ticks / ticks_per_beat
	
	last_channel = None
	absolute_time_in_ticks = 0

	for track in mid.tracks:
		note_ontimes_dict = {} 

		for msg in track:
			if msg.type == 'set_tempo':
				# If there's a tempo change, update the microseconds per beat
				microseconds_per_beat = msg.tempo

			absolute_time_in_ticks += msg.time  # Update absolute time with delta time
			
			if msg.type in ['note_on', 'note_off']:
				# Check if the channel has changed (for messages that have a channel attribute)
				if 'channel' in msg.__dict__ and (last_channel is None or msg.channel != last_channel):
					absolute_time_in_ticks = msg.time  # Reset the time if channel has changed
					last_channel = msg.channel  # Update the last channel seen

				if msg.type == 'note_on' and msg.velocity > 0:
					note_ontimes_dict[msg.note] = (absolute_time_in_ticks, msg.channel)

				elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and msg.note in note_ontimes_dict:
					onset_ticks, channel = note_ontimes_dict.pop(msg.note)
					onset_seconds = mido.tick2second(onset_ticks, mid.ticks_per_beat, microseconds_per_beat)
					onset_crochets = ticks_to_crochets(onset_ticks, mid.ticks_per_beat)
					absolute_time_in_crochets = ticks_to_crochets(absolute_time_in_ticks, mid.ticks_per_beat)
					duration_crochets = absolute_time_in_crochets - onset_crochets
					staff_num = channel + 1  # Assign the staff number based on the MIDI channel, channels are zero-indexed

					new_row = pd.DataFrame([[round(onset_crochets, 3), round(onset_seconds, 3), msg.note, round(duration_crochets, 3), staff_num]], columns=["onset", "onset_seconds", "pitch", "duration", "staff"]) 
					df = pd.concat([df, new_row], axis=0) 

	df.to_csv(output_filename, index=False) 
	print(f"Data has been written to {output_filename}")
midi_to_csv_mido()

# CSV format from https://github.com/Wiilly07/Beethoven_motif 
# code modified from https://github.com/andrewchenk/midi-csv/blob/master/midi_to_csv.py
def midi_to_csv_in_crochets_music21_NOT_USING():
	directory = "/Users/ilanashapiro/Documents/constraints_project/LOP_database_06_09_17/liszt_classical_archives"
	def process_midi(file_path):
		output_filepath = file_path[:-4] + "_crochets.csv"
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
				onset = round(float(note.offset), 3)  # The offset in quarter notes
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
# midi_to_csv_in_crochets_music21_NOT_USING()		

# code modified from https://github.com/Tsung-Ping/motif_discovery/blob/main/experiments.py 
def load_notes_csv(filename):
	dt = [
		('onset', float), # onset (in crotchet beats)
		# ('onset_seconds', float), # onset (in crotchet beats), don't want this for getting motives
		('pitch', float), # MIDI note number
		('duration', float), # duration (in crotchet beats)
		('staff', int) # staff number (integers from zero for the top staff)
	] # datatype
	
	# Format data as structured array
	with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		notes = np.array([tuple([float(x) for i, x in enumerate(row) if i != 1]) for i, row in enumerate(reader) if i > 0], dtype=dt)
	
	# Get unique notes irrespective of 'staffNum'
	_, unique_indices = np.unique(notes[['onset', 'pitch']], return_index=True)
	notes = notes[unique_indices]
	print('deleted notes:', [i for i in range(notes.size) if i not in unique_indices])
	notes = notes[notes['duration'] > 0]
	return np.sort(notes, order=['onset', 'pitch'])

# as per https://www.music-ir.org/mirex/wiki/2017:Discovery_of_Repeated_Themes_%26_Sections
def write_mirex_motives(motives, out_file, csv_file):
	# Read the CSV file
	df = pd.read_csv(csv_file)
	crochets_seconds_dict = df.set_index('onset')['onset_seconds'].to_dict()
	out_str = ""

	for idx_p, pattern in enumerate(motives):
		out_str += "pattern" + str(idx_p+1) + "\n" # + 1 because of zero-indexing
		for idx_o, occurrence in enumerate(pattern):
			out_str += "occurrence" + str(idx_o+1) + "\n"
			for ontime, pitch in occurrence:
				# Convert ontime from crochets to seconds using the mapping
				ontime_seconds = crochets_seconds_dict.get(ontime)  # Fallback to original if not found
				out_str += format(ontime_seconds, '.5f') + ", " + format(pitch, '.5f') + "\n"
		out_str += "\n"
	with open(out_file, "w") as f:
			f.write(out_str[:-2])

def get_motives():
	directory = "LOP_database_06_09_17/liszt_classical_archives/1_short_test"
	futures = []
	def process_file(file_path):
		notes = load_notes_csv(file_path)
		motives = SIA.find_motives(notes, horizontalTolerance=0, verticalTolerance=0, 
														 adjacentTolerance=(2, 12), min_notes=8, min_cardinality=0.8)
		# motives_test = [[[(174., 84.), (174.5, 57.), (175., 52.), (175.5, 54.)], [(178., 79.), (178.5, 52.), (179., 50.), (179.5, 48.)], [(186., 79.), (186.5, 52.), (187., 50.), (187.5, 48.)], [(194., 79.), (194.5, 52.), (195., 50.), (195.5, 48.)], [(198., 79.), (198.5, 52.), (199., 50.), (199.5, 48.)]], [[(174.5, 57.), (175., 52.), (175.5, 54.), (176., 55.)], [(180., 83.), (180.5, 79.), (181., 79.), (181.5, 79.)], [(188., 83.), (188.5, 79.), (189., 79.), (189.5, 79.)]]]
		write_mirex_motives(motives, file_path[:-4] + "_motives.txt", file_path) # "_data.csv" has length 9
	
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