import pandas as pd
import sys

from pyparsing import C
import librosa
import csv
import math 

# -------------------------------- THESE WERE BOTH TRAINED ONLY ON POP99 --------------------------------
# sys.path.append('/Users/ilanashapiro/Documents/MIDI-BERT') # https://github.com/wazenmai/MIDI-BERT
# sys.path.append('/Users/ilanashapiro/Documents/midi_melody_extraction') # https://github.com/bytedance/midi_melody_extraction

sys.path.append('/Users/ilanashapiro/Documents/audio_to_midi_melodia') # https://github.com/justinsalamon/audio_to_midi_melodia


file_path = '/Users/ilanashapiro/Documents/constraints_project/project/datasets/albeniz/classical_piano_midi_db/alb_esp1/alb_esp1_vamp_mtg-melodia_melodia_melody.csv' 
converted_file_path_midi = file_path[:-4] + "_converted.csv"
converted_file_path_intervals = converted_file_path_midi[:-4] + "_intervals.csv"
converted_file_path_signs = converted_file_path_intervals[:-4] + "_signs.csv"

def convert_hz_to_midi():
	data = pd.read_csv(file_path, header=None)
	for index, row in data.iterrows():
		hz_val = abs(row[1])
		midi_val = librosa.hz_to_midi(hz_val)
		data.at[index, 1] = int(round(midi_val))
	data.to_csv(converted_file_path_midi, index=False, header=False)
	print("Wrote", converted_file_path_midi)
# convert_hz_to_midi()

def collapse_midi_intervals():
	data = pd.read_csv(converted_file_path_midi, header=None, names=['secs', 'midi'])
	transformed_data = []

	start_time = data['secs'][0]
	start_midi = data['midi'][0]

	for i in range(1, len(data)):
			if data['midi'][i] != data['midi'][i - 1]:
					end_time = data['secs'][i]
					end_midi = data['midi'][i]
					interval = end_midi - start_midi
					transformed_data.append((f"({start_time}, {end_time})", interval))
					start_time = data['secs'][i]
					start_midi = end_midi

	# Add the last interval
	start_time = data['secs'].iloc[-1]
	interval = data['midi'].iloc[-1] - start_midi
	transformed_data.append((f"({start_time}, {end_time})", interval))
	transformed_df = pd.DataFrame(transformed_data)
	transformed_df.to_csv(converted_file_path_intervals, index=False, header=False)
	print("Wrote", converted_file_path_intervals)
# collapse_midi_intervals()

def collapse_interval_signs():
	def sign(x):
		return math.copysign(1, x)
	 
	data = pd.read_csv(converted_file_path_intervals, header=None, names=['time_interval', 'value'])
	transformed_data = []
	curr_time_interval = None
	curr_val = None

	for _, row in data.iterrows():
		time_interval = row['time_interval']
		val = row['value']
		
		if curr_time_interval is None:
			curr_time_interval = time_interval
			curr_val = val
		else:
			if sign(curr_val) != sign(val):
				# Merge intervals
				start_val = float(curr_time_interval.split(',')[0][1:])
				end_val = float(time_interval.split(',')[1][:-1])
				transformed_data.append((f"({start_val}, {end_val})", sign(val - curr_val)))
				curr_time_interval = time_interval
				curr_val = val
			else:
				# Update the end of the current time interval
				curr_time_interval = f"({curr_time_interval.split(',')[0][1:]}, {time_interval.split(',')[1][:-1]})"
				curr_val = val

	# Add the last interval
	if curr_time_interval is not None:
		transformed_data.append((curr_time_interval, curr_val))

	transformed_df = pd.DataFrame(transformed_data, columns=['interval', 'val'])
	transformed_df.to_csv(converted_file_path_signs, index=False, header=False)
	print("Wrote", converted_file_path_signs)
collapse_interval_signs()