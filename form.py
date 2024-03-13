import sys

sys.path.append('/Users/ilanashapiro/Documents/msaf0_1_80')
sys.path.append('/Users/ilanashapiro/Library/musicaiz-0.1.2')

import msaf
from midi2audio import FluidSynth
import os
from pydub import AudioSegment

def convert_dataset_midi_to_mp3():
	soundfont_filepath = "GeneralUser GS 1.471/GeneralUser GS v1.471.sf2"
	fs = FluidSynth(sound_font=soundfont_filepath)
	directory = "LOP_database_06_09_17/liszt_classical_archives/1_short_test"
	for root, _, files in os.walk(directory):
		for filename in files:
			if filename.endswith("_solo_short.mid"):
				midi_path = os.path.join(root, filename)
				wav_path = midi_path[:-4] + ".wav"
				fs.midi_to_audio(midi_path, wav_path)
				print("Converted", midi_path, "to WAV")

				mp3_path = midi_path[:-4] + ".mp3"
				audio = AudioSegment.from_wav(wav_path)
				audio.export(mp3_path, format="mp3", bitrate="192k", parameters=["-ar", "44100", "-ac", "2"])
				print("Converted", wav_path, "to MP3")

				os.remove(wav_path)
				print("Removed", wav_path)

# convert_dataset_midi_to_mp3()

def segment_audio_MSAF():
	audio_filepath = "LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short.mp3"

	# hierarchical: scluster, olda (boundaries only, use with fmc2d), vmo
	# sf (flat, boundaries only) has best performance
	boundaries, labels = msaf.process(audio_filepath, boundaries_id="vmo", labels_id="vmo", hier=True)
	print(boundaries)
	print(labels)

	out_file = audio_filepath[:-4] + '_segments.txt'
	print('Saving output to %s' % out_file)
	msaf.io.write_mirex_hierarchical(boundaries, labels, out_file)

# segment_audio_MSAF()