import sys
sys.path.append('/Users/ilanashapiro/Documents/motif_discovery') # https://github.com/Tsung-Ping/motif_discovery

import os
from multiprocessing import Pool
import pandas as pd
import csv
import numpy as np
import SIA
# import vmo

DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project/datasets"
DIRECTORY += '/beethoven/kunstderfuge/biamonti_488_(c)orlandi'
 
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
				if ontime not in crochets_seconds_dict:
					raise Exception("Encountered unassociated ontime for", out_file)
				ontime_seconds = crochets_seconds_dict.get(ontime) 
				out_str += format(ontime_seconds, '.5f') + ", " + format(pitch, '.5f') + "\n"
		out_str += "\n"
	with open(out_file, "w") as f:
			f.write(out_str[:-2])

def process_file(file_path):
	print("PROCESSING", file_path)
	notes = load_notes_csv(file_path)
	if len(notes) < 1500:
		# motives = SIA.find_motives(notes, horizontalTolerance=0, verticalTolerance=0, 
		# 													adjacentTolerance=(2, 6), min_notes=10, min_cardinality=0.4) # I think this is motives4.txt
		# motives = SIA.find_motives(notes, horizontalTolerance=0, verticalTolerance=0, 
		# 													adjacentTolerance=(1, 6), min_notes=5, min_cardinality=0.7) # THIS IS MOTIVES1.TXT
		motives = SIA.find_motives(notes, horizontalTolerance=0, verticalTolerance=0, 
																adjacentTolerance=(1, 6), min_notes=9, min_cardinality=0.5) # THIS SHOULD BE MOTIVES3.TXT, NEED TO CHECK
		# motives_test = [[[(174., 84.), (174.5, 57.), (175., 52.), (175.5, 54.)], [(178., 79.), (178.5, 52.), (179., 50.), (179.5, 48.)], [(186., 79.), (186.5, 52.), (187., 50.), (187.5, 48.)], [(194., 79.), (194.5, 52.), (195., 50.), (195.5, 48.)], [(198., 79.), (198.5, 52.), (199., 50.), (199.5, 48.)]], [[(174.5, 57.), (175., 52.), (175.5, 54.), (176., 55.)], [(180., 83.), (180.5, 79.), (181., 79.), (181.5, 79.)], [(188., 83.), (188.5, 79.), (189., 79.), (189.5, 79.)]]]
		write_mirex_motives(motives, file_path[:-4] + "_motives3.txt", file_path) # "_data.csv" has length 9
		print("WROTE", file_path[:-4] + "_motives3.txt")
	return len(notes)

def check_conditions_for_csv(directory):
	for file in os.listdir(directory):
		if file.endswith("motives4.txt"):
			return False  # Skip this directory as it contains the desired motives file already
	return True

def prepare_file_paths(directory):
	valid_file_paths = []
	for root, _, files in os.walk(directory):
		for filename in files:
			if filename.endswith(".csv") and check_conditions_for_csv(root):
				valid_file_paths.append(os.path.join(root, filename))
	return valid_file_paths

def get_motives():
	# file_paths = prepare_file_paths(DIRECTORY)
	# file_paths = []

	for root, _, files in os.walk(DIRECTORY):
		for filename in files:
			if filename.endswith(".csv") and "melody" not in filename:
				file_path = os.path.join(root, filename)
				process_file(file_path)
				# file_paths.append(file_path)

	# Create a pool of workers and distribute the file processing tasks
	# with Pool() as pool:
	# 	pool.map(process_file, file_paths)

if __name__ == '__main__':
	get_motives()

#---------------------------------------------USING VMO FOR MOTIF EXTXRACTION------------------------------------------------------------
# vmo is MUCH faster than the newer paper I'm using for this, which can detect longer patterns of specified length
# however, VMO seems to almost exclusively detect very very short patterns (2 notes), in this example there's only
# 2 patterns of length 5 and 13 of length 4. when I request min length 4, it only gives me 4 patterns but some patterns
# still have length 1 this is probably a bug in VMO?? requesting length 5 gives me to patterns length 1 and 2....
	
# output format for motifs: list of elements of form [[892, 510, 693, 684, 512], 2]. explanation:
# Occurrences: The pattern repeats five times at different parts of the sequence, specifically starting at indices 892, 510, 693, 684, and 512.
# Pattern Length: Each instance of this repeating pattern includes 2 consecutive elements from the sequence.
	
# midi_file_path = 'LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short.mid'
# score = converter.parse(midi_file_path)
# musicxml_path = 'LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short.musicxml'
# score.write('musicxml', fp=musicxml_path)

# s = converter.parse(musicxml_path)
# notes = []
# for note in s.recurse().notes:
# 	if note.isNote:
# 		# For single notes, append note name and octave
# 		n = note.pitches
# 		notes.append(f"{n[0].nameWithOctave}")
# 	elif note.isChord:
# 		# For chords, append each note in the chord
# 		chord_notes = '.'.join(n.nameWithOctave for n in note.pitches)
# 		notes.append(f"Chord: {chord_notes}")
# oracle = vmo.build_oracle(notes,'f')
# motives = vmo.analysis.find_repeated_patterns(oracle, lower=5)
# print(motives)

# print(msaf0_1_80.features_registry)
# print(msaf0_1_80.get_all_boundary_algorithms())
# print(msaf0_1_80.get_all_label_algorithms())

