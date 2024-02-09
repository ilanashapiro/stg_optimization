# Simple MSAF example
# from __future__ import print_function
import sys
sys.path.append('/Users/ilanashapiro/Documents/msaf0_1_80')
sys.path.append('/Users/ilanashapiro/Library/musicaiz-0.1.2')
sys.path.append('/Users/ilanashapiro/Library/MIDI-BERT') # https://github.com/wazenmai/MIDI-BERT
sys.path.append('/Users/ilanashapiro/Library/midi_melody_extraction') # https://github.com/bytedance/midi_melody_extraction
sys.path.append('/Users/ilanashapiro/Library/motif_discovery') # https://github.com/Tsung-Ping/motif_discovery
sys.path.append('/Users/ilanashapiro/Library/audio_to_midi_melodia') # https://github.com/justinsalamon/audio_to_midi_melodia
sys.path.append('/Users/ilanashapiro/Library/functional-harmony') # https://github.com/Tsung-Ping/functional-harmony
sys.path.append('/Users/ilanashapiro/Library/Harmony-Transformer') # https://github.com/Tsung-Ping/Harmony-Transformer

import msaf
from midi2audio import FluidSynth
import os
import pandas as pd
import mido

def midi_to_csv():
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

	df.to_csv(output_filename, index=False)
	print(f"Data has been written to {output_filename}")

midi_to_csv()

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

# def midi_to_csv():
# 	midi_filepath = "LOP_database_06_09_17/liszt_classical_archives/0/bl11_solo.mid" # for now
# 	csv_string = pm.midi_to_csv(midi_filepath)

# 	with open(midi_filepath[:-3] + "csv", "w") as f:
# 		f.writelines(csv_string)

# midi_to_csv()

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

	# 4. Evaluate the results
	# evals = msaf.eval.process(audio_file)
	# print(evals)

# convert_dataset_midi_to_wav()
# segment_audio_MSAF()

# print(msaf0_1_80.features_registry)
# print(msaf0_1_80.get_all_boundary_algorithms())
# print(msaf0_1_80.get_all_label_algorithms())

