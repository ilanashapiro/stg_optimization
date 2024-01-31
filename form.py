# Simple MSAF example
# from __future__ import print_function
import sys
sys.path.append('/Users/ilanashapiro/Documents/msaf0_1_80')

import msaf0_1_80
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

def segment_audio_MSAF():
  # 1. Select audio file
  audio_filepath = "LOP_database_06_09_17/liszt_classical_archives/0/bl11_solo.wav"

  # 2. Segment the file using the default MSAF parameters (this might take a few seconds)
  boundaries, labels = msaf0_1_80.process(audio_filepath, boundaries_id="olda", labels_id="cnmf", hier=True)
  print(boundaries)
  print(labels)
  # 3. Save segments using the MIREX format
  # out_file = audio_filepath[:-4] + '_segments1.txt'
  # print('Saving output to %s' % out_file)
  # msaf0_1_80.io.write_mirex(boundaries, labels, out_file)

  # 4. Evaluate the results
  # evals = msaf.eval.process(audio_file)
  # print(evals)

# convert_dataset_midi_to_wav()
segment_audio_MSAF()

# print(msaf0_1_80.features_registry)
# print(msaf0_1_80.get_all_boundary_algorithms())
# print(msaf0_1_80.get_all_label_algorithms())

