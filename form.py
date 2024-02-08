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

import py_midicsv as pm
import msaf
from midi2audio import FluidSynth
import os

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

def midi_to_csv():
  midi_filepath = "LOP_database_06_09_17/liszt_classical_archives/0/bl11_solo.mid" # for now
  csv_string = pm.midi_to_csv(midi_filepath)

  with open(midi_filepath[:-3] + "csv", "w") as f:
    f.writelines(csv_string)

midi_to_csv()

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

