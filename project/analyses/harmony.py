import os, json, mido, pandas as pd, sys
from multiprocessing import Pool
import format_conversions as fc

# NOTE: all generation of functional harmony labels is done inside Harmony-Transformer-v2
# This file is purely for post-processing of the computed timesteps (i.e. 1/16 notes) -> seconds

DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project/datasets"

def convert_timesteps_to_seconds(piece_dir):
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
	mid_df['end_time'] = mid_df['onset_seconds'] + mid_df['duration_seconds']
	max_end_time = mid_df['end_time'].max()
	
	with open(fh_file, 'r') as f:
		lines = f.readlines()

	converted_data = [{'end_time': max_end_time}]
	for line in lines:
		data = json.loads(line.strip())
		timestep = data["timestep"]
		
		ticks = timestep_to_ticks(timestep, mid.ticks_per_beat)
		seconds = fc.ticks_to_secs_with_tempo_changes(ticks, tempo_changes, mid.ticks_per_beat)
		
		# NOTE: this ONLY seems to be happening at the very end of midi files where there is silence
		# the silence doesn't get translated to the audio. remove these chord predictions because the 
		# file has silence, and these are therefore incorrect.s
		if seconds > max_end_time:
			tolerance = seconds - max_end_time
			print(f"Computed seconds value {seconds} exceeds end_time {max_end_time} in CSV file for {piece_dir} with tolerance {tolerance}")
			continue
	
		data["onset_seconds"] = seconds
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
	for root, dirs, _ in os.walk(DIRECTORY):
		for dir_name in dirs:
			dir_path = os.path.join(root, dir_name)
			if contains_functional_harmony(dir_path):
				fh_dirs.append(dir_path)

	with Pool() as pool:
		pool.map(convert_timesteps_to_seconds, fh_dirs)

if __name__ == "__main__":
		main()