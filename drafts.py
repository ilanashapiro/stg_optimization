from musicaiz.loaders import Musa
from musicaiz import eval, features

# load file
path = "maestro-v3.0.0/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi"
midi = Musa(path)
structurePredictor = features.StructurePrediction(path)
print(structurePredictor._get_structure_boundaries("high", "BPS"))

# from mido import MidiFile

# midi_file = MidiFile("lmd_matched/M/Q/W/TRMQWXQ128F423D02A/7676add78136d3336cf833372c6fa9be.mid")

# Initialize start and end times
# start_time = None
# end_time = None

# for i, track in enumerate(midi_file.tracks):
#     print("TRACK")
#     cumulative_time = 0

#     for msg in track:
#         print(msg.time)
#         cumulative_time += msg.time

        # # Update start_time for the first event
        # if start_time is None:
        #     start_time = cumulative_time

        # # Update end_time for every event
        # end_time = cumulative_time


