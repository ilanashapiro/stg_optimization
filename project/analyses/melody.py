import pandas as pd
import sys
import librosa
import numpy as np 

# -------------------------------- THESE WERE BOTH TRAINED ONLY ON POP99 --------------------------------
# sys.path.append('/Users/ilanashapiro/Documents/MIDI-BERT') # https://github.com/wazenmai/MIDI-BERT
# sys.path.append('/Users/ilanashapiro/Documents/midi_melody_extraction') # https://github.com/bytedance/midi_melody_extraction

sys.path.append('/Users/ilanashapiro/Documents/audio_to_midi_melodia') # https://github.com/justinsalamon/audio_to_midi_melodia

file_path = '/Users/ilanashapiro/Documents/constraints_project/project/datasets/albeniz/classical_piano_midi_db/alb_esp1/alb_esp1_vamp_mtg-melodia_melodia_melody.csv' 
converted_file_path_midi = file_path[:-4] + "_converted.csv"
converted_file_path_intervals = converted_file_path_midi[:-4] + "_intervals.csv"
converted_file_path_signs = converted_file_path_intervals[:-4] + "_signs.csv"

def convert_hz_to_midi(tolerance=0.5):
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
	df_to_write.to_csv(converted_file_path_midi, index=False, header=False)
	print("Wrote", converted_file_path_midi)
# convert_hz_to_midi()

def collapse_midi_interval_signs():
		data = pd.read_csv(converted_file_path_midi, header=None, names=['secs', 'midi'])
		transformed_data = []

		start_time = data['secs'][0]
		prev_val = data['midi'][0]
		prev_sign = 0

		for row in data.iloc[1:].itertuples(index=True):
			curr_time, curr_val = row.secs, row.midi
			curr_sign = np.sign(curr_val - prev_val)

			if curr_sign != prev_sign and curr_sign != 0:
				transformed_data.append(((start_time, curr_time), curr_sign))
				start_time = curr_time
				prev_val = curr_val
				prev_sign = curr_sign
				
		# Handle the last interval and merge it if the sign is the same
		if prev_sign != 0:
			last_start_time = transformed_data[-1][0][0]
			last_sign = transformed_data[-1][1]
			if last_sign == prev_sign:
				transformed_data[-1] = ((last_start_time, data.iloc[-1]['secs']), last_sign)
			else:
				transformed_data.append(((start_time, data.iloc[-1]['secs']), prev_sign))
				
		transformed_df = pd.DataFrame(transformed_data)
		transformed_df.to_csv(converted_file_path_signs, index=False, header=False)
		print("Wrote", converted_file_path_signs)
collapse_midi_interval_signs()

############################# FOR WORKING WITH THE ACTUAL INTERVALS AND THEN DERIVING THE SIGNS ##########################

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
			if curr_val and np.sign(curr_val) != np.sign(val):
				# Merge intervals
				start_val = float(curr_time_interval.split(',')[0][1:])
				end_val = float(time_interval.split(',')[1][:-1])
				transformed_data.append((f"({start_val}, {end_val})", np.sign(val - curr_val)))
				curr_time_interval = time_interval
				curr_val = val
			else:
				# Update the end of the current time interval
				curr_time_interval = f"({curr_time_interval.split(',')[0][1:]}, {time_interval.split(',')[1][:-1]})"
				curr_val = val

	# Add the last interval
	if curr_time_interval is not None and curr_val:
		transformed_data.append((curr_time_interval, np.sign(curr_val)))

	transformed_df = pd.DataFrame(transformed_data, columns=['interval', 'val'])
	transformed_df.to_csv(converted_file_path_signs, index=False, header=False)
	print("Wrote", converted_file_path_signs)
# collapse_interval_signs()