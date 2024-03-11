import sys

sys.path.append('/Users/ilanashapiro/Documents/msaf0_1_80')
sys.path.append('/Users/ilanashapiro/Library/musicaiz-0.1.2')

import msaf
from midi2audio import FluidSynth
import os
from pydub import AudioSegment
import muspy
from music21 import converter, note
import vmo

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


#---------------USING VMO FOR MOTIF EXTXRACTION
# vmo is MUCH faster than the newer paper I'm using for this, which can detect longer patterns of specified length
# however, VMO seems to almost exclusively detect very very short patterns (2 notes), in this example there's only
# 2 patterns of length 5 and 13 of length 4. when I request min length 4, it only gives me 4 patterns but some patterns
# still have length 1 this is probably a bug in VMO?? requesting length 5 gives me to patterns length 1 and 2....
	
# output format for motifs: list of elements of form [[892, 510, 693, 684, 512], 2]. explanation:
# Occurrences: The pattern repeats five times at different parts of the sequence, specifically starting at indices 892, 510, 693, 684, and 512.
# Pattern Length: Each instance of this repeating pattern includes 2 consecutive elements from the sequence.
	
# midi_file_path = 'LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short.mid'
# score = converter.parse(midi_file_path)
musicxml_path = 'LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short.musicxml'
# score.write('musicxml', fp=musicxml_path)

s = converter.parse(musicxml_path)
notes = []
for note in s.recurse().notes:
	if note.isNote:
		# For single notes, append note name and octave
		n = note.pitches
		notes.append(f"{n[0].nameWithOctave}")
	elif note.isChord:
		# For chords, append each note in the chord
		chord_notes = '.'.join(n.nameWithOctave for n in note.pitches)
		notes.append(f"Chord: {chord_notes}")
oracle = vmo.build_oracle(notes,'f')
motives = vmo.analysis.find_repeated_patterns(oracle, lower=5)
print(motives)

# print(msaf0_1_80.features_registry)
# print(msaf0_1_80.get_all_boundary_algorithms())
# print(msaf0_1_80.get_all_label_algorithms())

