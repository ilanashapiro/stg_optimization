import music21
from networkx import johnson
import numpy as np
import os, sys, pickle, re
import joblib
from scipy.spatial.distance import pdist, squareform

# DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
DIRECTORY = "/home/ilshapiro/project"
# DIRECTORY = "/home/ubuntu/project"
sys.path.append(f"{DIRECTORY}/experiments/structural_distance/structural_distance_music")
import structural_distance_gen_clusters as gen_clusters

MAX_CHORDS = 1000
MAX_PITCHES = 128
MAX_NOTES = 100
MAX_SECTIONS = 10

feature_vector_cache = {}

'''
This file contains the code for the MIDI features Baseline 1 in the structural distance music evaluation experiment in Section 6.1 of the paper
'''

def save_cache(cache_filepath):
		with open(cache_filepath, 'wb') as f:
				pickle.dump(feature_vector_cache, f)

def load_cache(cache_filepath):
	global feature_vector_cache
	if os.path.exists(cache_filepath):
		with open(cache_filepath, 'rb') as f:
				feature_vector_cache = pickle.load(f)
	else:
		feature_vector_cache = {}

def get_feature_vector(midi_filepath):
	if midi_filepath in feature_vector_cache:
			return feature_vector_cache[midi_filepath]
	feature_vector = combined_feature_vector(midi_filepath)
	feature_vector_cache[midi_filepath] = feature_vector
	return feature_vector

def extract_note_features(midi_file):
	score = music21.converter.parse(midi_file)
	notes_info = []
	
	for element in score.flatten().notes:  # Iterate over all notes and chords
		if isinstance(element, music21.note.Note):
			pitch = element.pitch.midi  # MIDI pitch number
			duration = element.quarterLength  # Duration of the note
			start_time = element.offset
			notes_info.append((pitch, duration, start_time))
		elif isinstance(element, music21.chord.Chord):
			for pitch in element.pitches:
				start_time = element.offset
				notes_info.append((pitch.midi, element.quarterLength, start_time))
							
	return notes_info

def vectorize_notes(notes_info, num_pitches=MAX_PITCHES, max_notes=MAX_NOTES):
	pitch_vector = np.zeros((max_notes, num_pitches), dtype=int)
	duration_vector = np.zeros(max_notes, dtype=float)
	start_time_vector = np.zeros(max_notes, dtype=float)

	for i, (pitch, duration, start_time) in enumerate(notes_info[:max_notes]):
		pitch_vector[i, pitch] = 1  # One-hot encoding of pitch
		duration_vector[i] = duration
		start_time_vector[i] = start_time

	feature_vector = np.concatenate([pitch_vector.flatten(), duration_vector, start_time_vector])
	return feature_vector

def extract_harmony_features(midi_file):
	score = music21.converter.parse(midi_file)

	chord_info = []
	for chrd in score.flatten().getElementsByClass('Chord'):
		pitches = [pitch.midi for pitch in chrd.pitches]  # MIDI pitch numbers
		duration = chrd.duration.quarterLength
		chord_info.append((pitches, duration))

	return chord_info

def vectorize_harmony(chord_info, num_pitches=MAX_PITCHES, max_chords=MAX_CHORDS):
	chord_pitches_vector = np.zeros((max_chords, num_pitches), dtype=int)
	chord_duration_vector = np.zeros(max_chords, dtype=float)

	for i, (pitches, duration) in enumerate(chord_info[:max_chords]):
		for pitch in pitches:
			chord_pitches_vector[i, pitch] = 1  # One-hot encoding of pitches
		chord_duration_vector[i] = duration

	feature_vector = np.concatenate([chord_pitches_vector.flatten(), chord_duration_vector])
	return feature_vector

def extract_structural_features(midi_file):
	score = music21.converter.parse(midi_file)
	num_sections = len(score.getElementsByClass('Section'))
	section_boundaries = []
	
	for section in score.getElementsByClass('Section'):
		boundaries = (section.offset, section.offset + section.quarterLength)
		section_boundaries.append(boundaries)
	
	return num_sections, section_boundaries

def vectorize_structural_features(num_sections, section_boundaries, max_sections=MAX_SECTIONS):
	num_sections_vector = np.zeros(max_sections, dtype=int)
	section_boundaries_vector = np.zeros(max_sections * 2, dtype=float)  # Start and end
	num_sections_vector[:num_sections] = 1  # Binary encoding of sections presence

	for i, (start, end) in enumerate(section_boundaries[:max_sections]):
		section_boundaries_vector[2*i] = start
		section_boundaries_vector[2*i + 1] = end

	feature_vector = np.concatenate([num_sections_vector, section_boundaries_vector])
	return feature_vector

def combined_feature_vector(midi_file):
	notes_info = extract_note_features(midi_file)
	chord_info = extract_harmony_features(midi_file)
	num_sections, section_boundaries = extract_structural_features(midi_file)
	
	note_vector = vectorize_notes(notes_info, max_notes=MAX_NOTES)
	harmony_vector = vectorize_harmony(chord_info, num_pitches=MAX_PITCHES, max_chords=MAX_CHORDS)
	structural_vector = vectorize_structural_features(num_sections, section_boundaries, max_sections=MAX_SECTIONS)
	
	return np.concatenate([note_vector, harmony_vector, structural_vector])

def get_midi_filepath(pickle_filepath):
	midi_path_mid = pickle_filepath.replace('_augmented_graph_flat.pickle', '.mid')
	midi_path_MID = pickle_filepath.replace('_augmented_graph_flat.pickle', '.MID')
	if os.path.exists(midi_path_mid):
		return midi_path_mid
	elif os.path.exists(midi_path_MID):
		return midi_path_MID
	else:
		raise ValueError("No midi associated with", pickle_filepath)

def run(clusters_filepath):
	clusters = list(gen_clusters.load_saved_combinations(clusters_filepath))

	cache_filepath = f'{DIRECTORY}/experiments/structural_distance/baseline1_midi_features/feature_vector_cache.pkl'
	load_cache(cache_filepath)

	all_matrices = []
	for cluster in clusters:
		feature_vectors = [get_feature_vector(get_midi_filepath(re.sub(r'^.*?/project', DIRECTORY, piece_stg))) for piece_stg in cluster]
		
		# Calculate cosine distances between feature vectors
		distances = pdist(feature_vectors, metric='cosine')  # Pairwise cosine distances
		distance_matrix = squareform(distances)  # Convert to square matrix

		all_matrices.append(distance_matrix)

	save_cache(cache_filepath)

	return np.mean(np.array(all_matrices), axis=0)

if __name__ == "__main__":
	clusters_filepath = f"{DIRECTORY}/experiments/structural_distance/structural_distance_music/input_clusters_totalordertol2.pkl"
	print(run(clusters_filepath))
	