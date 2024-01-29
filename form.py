# Simple MSAF example
# from __future__ import print_function
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

convert_dataset_midi_to_wav()

def segment_audio_MSAF():
  # 1. Select audio file
  audio_file = "maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.wav"

  # 2. Segment the file using the default MSAF parameters (this might take a few seconds)
  boundaries, labels = msaf.process(audio_file)
  # 3. Save segments using the MIREX format
  out_file = 'segments.txt'
  print('Saving output to %s' % out_file)
  msaf.io.write_mirex(boundaries, labels, out_file)

  # 4. Evaluate the results
  # evals = msaf.eval.process(audio_file)
  # print(evals)




