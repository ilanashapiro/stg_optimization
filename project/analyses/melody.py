import pandas as pd
import sys, os
import librosa
import numpy as np 
import subprocess
from multiprocessing import Pool

# -------------------------------- THESE WERE BOTH TRAINED ONLY ON POP99 ------------------------------------------------------
# sys.path.append('/Users/ilanashapiro/Documents/MIDI-BERT') # https://github.com/wazenmai/MIDI-BERT
# sys.path.append('/Users/ilanashapiro/Documents/midi_melody_extraction') # https://github.com/bytedance/midi_melody_extraction
# -----------------------------------------------------------------------------------------------------------------------------

# sys.path.append('/Users/ilanashapiro/Documents/audio_to_midi_melodia') # https://github.com/justinsalamon/audio_to_midi_melodia
# FOR MELODIA VERY IMPORTANT, EACH TERMINAL SESSION NEED TO RUN:
# export VAMP_PATH=$VAMP_PATH:/home/ilshapiro/project/MELODIA
# export PATH=$PATH:~/project/analyses
# confirm with: sonic-annotator -l

DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project/datasets"

def execute_command(command):
	print(f"Running command: {' '.join(command)}")
	subprocess.run(command)

def extract_melody():
	# Define the base directory containing the composer folders
	base_dir = '/home/ilshapiro/project/datasets'
	commands = []

	for composer_folder in os.listdir(base_dir):
		composer_path = os.path.join(base_dir, composer_folder)
		
		if not os.path.isdir(composer_path):
			continue
		
		for subfolder in ['classical_piano_midi_db', 'kunstderfuge']:
			subfolder_path = os.path.join(composer_path, subfolder)
			
			if not os.path.isdir(subfolder_path):
				continue
			
			for piece_folder in os.listdir(subfolder_path):
				piece_path = os.path.join(subfolder_path, piece_folder)
				
				if not os.path.isdir(piece_path):
					continue

				mp3_file_exists = [f for f in os.listdir(piece_path) if f.endswith('.mp3')]
				motives_file_exists = [f for f in os.listdir(piece_path) if f.endswith('motives3.txt') or f.endswith('motives1.txt')]
				melody_file_exists = [f for f in os.listdir(piece_path) if f.endswith('melody.csv')]

				# If there's no mp3 or no motives file in this dir, or if the melody file already exists
				if not mp3_file_exists or not motives_file_exists or melody_file_exists:
					continue
	
				for file in os.listdir(piece_path):
					if file.endswith('.mp3'):
						audiofile_path = os.path.join(piece_path, file)
						command = ['./sonic-annotator', '-d', 'vamp:mtg-melodia:melodia:melody', audiofile_path, '-w', 'csv']
						commands.append(command)

	return commands

def convert_hz_to_midi(file_path, tolerance=0.5):
	data = pd.read_csv(file_path, header=None, names=['timestamp', 'hz'])
	rows_to_write = []
	prev_midi_val = None

	for _, row in data.iterrows():
		hz_val = row['hz']
	
		if hz_val > 0:
			midi_val = librosa.hz_to_midi(hz_val)
			if prev_midi_val is not None:
				if abs(midi_val - prev_midi_val) <= tolerance:
					midi_val = prev_midi_val

			row['hz'] = int(round(midi_val))
			rows_to_write.append(row)
			prev_midi_val = midi_val
	
	df_to_write = pd.DataFrame(rows_to_write)
	out_filepath = file_path[:-4] + "_converted.csv"
	df_to_write.to_csv(out_filepath, index=False, header=False)
	print("Wrote", out_filepath)

def extract_melody_contour(file_path, out_file):
	data = pd.read_csv(file_path, header=None, names=['secs', 'midi'])
	if data.empty:
		print("ERROR: No melody data for", file_path)
		return
	
	notes = list(data['midi'])
	timesteps = list(data['secs'])
	melody_contour = []
	note_start_idx = 0
	prev_midi = None

	while note_start_idx < len(notes):
		curr_midi = notes[note_start_idx]
		note_end_idx = note_start_idx
		
		# Find the end of the current segment
		while note_end_idx < len(notes) and notes[note_end_idx] == curr_midi:
			note_end_idx += 1
		
		start_time = timesteps[note_start_idx]
		end_time = timesteps[note_end_idx] if note_end_idx < len(timesteps) else timesteps[-1]
		
		if prev_midi is not None:
			interval = curr_midi - prev_midi
			melody_contour.append(((start_time, end_time), interval))
		
		prev_midi = curr_midi
		note_start_idx = note_end_idx

	transformed_df = pd.DataFrame(melody_contour)
	transformed_df.to_csv(out_file, index=False, header=False)
	print("Wrote", out_file)

def process_file(filepath):
	print("PROCESSING", filepath)
	
	# if not os.path.exists(filepath[:-4] + "_converted.csv"):
	# 	convert_hz_to_midi(filepath)
	
	# REQUIRES THAT convert_hz_to_midi HAS ALREADY BEEN EXECUTED AND MIDI FILES EXIST
	# OR THE RESULT IS INCORRECT
	out_file = filepath.replace("_converted", "")[:-4] + "_contour.csv"
	if not os.path.exists(out_file):
		extract_melody_contour(filepath, out_file)

if __name__ == "__main__":
	# commands = extract_melody()
	# with Pool() as pool:
	# 	pool.map(execute_command, commands)

	file_queue = []
	for root, _, files in os.walk(DIRECTORY):
		for filename in files:
			# if filename.endswith("_melody.csv"): # for convert_hz_to_midi in process_file
			# 	melody_midi_file = [f for f in os.listdir(root) if f.endswith('_melody_converted.csv')]
			# 	if not melody_midi_file:
			# 		filepath = os.path.join(root, filename)
			# 		file_queue.append(filepath)
			
			if filename.endswith("_melody_converted.csv"): # run this if block for extract_melody_contour
				filepath = os.path.join(root, filename)
				file_queue.append(filepath)
				# process_file(filepath)

	with Pool() as pool:
		pool.map(process_file, file_queue)