import os, json, mido, pandas as pd, sys
from multiprocessing import Pool
import format_conversions as fc

# NOTE: all generation of functional harmony labels is done inside Harmony-Transformer-v2
# This file is purely for post-processing of the computed timesteps (i.e. 1/16 notes) -> seconds

DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project/datasets"
# DIRECTORY = '/Users/ilanashapiro/Documents/constraints_project/project/datasets/chopin/classical_piano_midi_db/chpn-p7'

def get_functional_chord_label(degree1, degree2, quality, key):
		maj_roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
		min_roman = [r.lower() for r in maj_roman]
		quality_map = {'M':'', 'm':'', 'a':'+', 'd':'o', 'M7':'M7', 'm7':'m7', 'D7':'7', 'd7':'o7', 'h7':'ø7', 'a6':'+6', 'a7':'aug7'}
		accidental_map = {'+':'#', '-':'b'}

		def parse_degree(degree):
			deg_num = int(degree[1]) if any(acc in degree for acc in accidental_map.keys()) else int(degree)
			deg_acc = accidental_map[degree[0]] if any(acc in degree for acc in accidental_map.keys()) else ""
			return (deg_num, deg_acc)

		deg1_num, deg1_acc = parse_degree(degree1)
		deg2_num, deg2_acc = parse_degree(degree2)

		if any(deg < 1 or deg > 7 for deg in [deg1_num, deg2_num]):
			raise Exception("Degree not in valid range", degree1, degree2)
		if quality not in quality_map:
			raise Exception("Invalid quality", quality)
		
		secondary_key_roman = None
		if deg1_num > 1: # we're in a secondary chord
			key_letter = key[0]
			secondary_key_qual_is_major = key_letter.isupper() and deg1_num in [4, 5] or not key_letter.isupper() and deg1_num in [3, 5, 6, 7]
			secondary_key_roman = deg1_acc + (maj_roman[deg1_num - 1] if secondary_key_qual_is_major else min_roman[deg1_num - 1]) # 0-indexing

		if quality == "a6":
			chord_roman = "aug6"
		else:
			chord_symbol = maj_roman[deg2_num - 1] if quality[0].isupper() else min_roman[deg2_num - 1] # 0-indexing
			chord_roman = deg2_acc + chord_symbol + quality_map[quality]

		return f'{chord_roman}/{secondary_key_roman}' if secondary_key_roman else chord_roman

# convert timesteps to seconds, i.e. add onset seconds, and add functional/roman chord symbols
def augment_entry(piece_dir):
	def timestep_to_ticks(timestep, ticks_per_beat):
		return timestep * (ticks_per_beat / 4)  # 1 timestep = 1/16 of a quarter note
	
	piece_name = os.path.basename(os.path.normpath(piece_dir))
	fh_file = os.path.join(piece_dir, piece_name + "_functional_harmony.txt")
	midi_csv_file = os.path.join(piece_dir, piece_name + ".csv")
	midi_file = None

	# Check both possible extensions for MIDI file
	for ext in [".mid", ".MID"]:
		midi_filepath = os.path.join(piece_dir, piece_name + ext)
		midi_filepath_caps = os.path.join(piece_dir, piece_name.upper() + ext)
		if os.path.exists(midi_filepath):
			midi_file = midi_filepath
		elif os.path.exists(midi_filepath_caps):
			midi_file = midi_filepath_caps
			break
	if midi_file is None:
		raise FileNotFoundError(f"MIDI file not found in {piece_dir}")

	mid = mido.MidiFile(midi_file)
	tempo_changes = fc.preprocess_tempo_changes(mid)

	mid_df = None
	if os.path.exists(midi_csv_file):
		mid_df = pd.read_csv(midi_csv_file)
	elif os.path.exists(os.path.join(piece_dir, piece_name.upper() + ".csv")):
		mid_df = pd.read_csv(os.path.join(piece_dir, piece_name.upper() + ".csv"))
	if mid_df is None:
		raise FileNotFoundError(f"CSV file not found in {piece_dir}")
	
	# Convert durations to seconds and calculate end times
	mid_df['duration_seconds'] = mid_df['duration'].apply(
			lambda duration: fc.ticks_to_secs_with_tempo_changes(
					timestep_to_ticks(duration * 4, mid.ticks_per_beat), tempo_changes, mid.ticks_per_beat)
	)
	
	with open(fh_file, 'r') as f:
		lines = f.readlines()

	converted_data = []
	for line in lines:
		data = json.loads(line.strip())
		timestep = data["timestep"]
		ticks = timestep_to_ticks(timestep, mid.ticks_per_beat)
		seconds = fc.ticks_to_secs_with_tempo_changes(ticks, tempo_changes, mid.ticks_per_beat)
		data["onset_seconds"] = seconds
		data["roman_chord"] = get_functional_chord_label(data["degree1"], data["degree2"], data["quality"], data["key"])
		converted_data.append(data)
	
	output_file = fh_file
	with open(output_file, 'w') as f:
		for item in converted_data:
			f.write(json.dumps(item) + "\n")
		print("WROTE", output_file)

	return converted_data

def contains_functional_harmony(directory):
	# Check if the directory contains a file ending with "_motives3.txt"
	for filename in os.listdir(directory):
		if filename.endswith("_functional_harmony.txt"):
			return True
	return False

def main():
	fh_dirs = []
	for dir_path, _, _ in os.walk(DIRECTORY):
		if contains_functional_harmony(dir_path):
			fh_dirs.append(dir_path)

	with Pool() as pool:
		pool.map(augment_entry, fh_dirs)

if __name__ == "__main__":
		main()