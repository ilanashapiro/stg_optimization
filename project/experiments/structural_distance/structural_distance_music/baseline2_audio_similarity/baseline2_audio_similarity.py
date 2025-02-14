import os, sys, re, pickle
import numpy as np

DIRECTORY = "/home/ilshapiro/project"
# DIRECTORY = "/home/ubuntu/project"
# DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
sys.path.append(f"{DIRECTORY}/experiments/structural_distance/structural_distance_music/baseline2_audio_similarity/audio-similarity/audio_similarity")
sys.path.append(f"{DIRECTORY}/experiments/structural_distance/structural_distance_music")

import structural_distance_gen_clusters as gen_clusters
from audio_similarity import AudioSimilarity

audio_similarity_cache = {}

'''
This file contains the code for the Audio Similarity SWAS Baseline 2 for the structural distance music evaluation experiment in Section 6.1 of the paper
'''

def get_audio_similarity(mp3_filepath1, mp3_filepath2):
	cache_key = repr((mp3_filepath1, mp3_filepath2))
	cache_key_rev = repr((mp3_filepath2, mp3_filepath1))

	if cache_key in audio_similarity_cache:
		return audio_similarity_cache[cache_key]
	elif cache_key_rev in audio_similarity_cache:
		return audio_similarity_cache[cache_key_rev]
	
	audio_similarity = calculate_audio_similarity(mp3_filepath1, mp3_filepath2)
	audio_similarity_cache[cache_key] = audio_similarity
	return audio_similarity

def calculate_audio_similarity(mp3_filepath1, mp3_filepath2):
	audio_similarity = AudioSimilarity(mp3_filepath1, mp3_filepath2, sample_rate=44100)
	return audio_similarity.stent_weighted_audio_similarity()

def save_cache(cache_filepath):
	with open(cache_filepath, 'wb') as f:
		pickle.dump(audio_similarity_cache, f)

def load_cache(cache_filepath):
	global audio_similarity_cache
	if os.path.exists(cache_filepath):
		with open(cache_filepath, 'rb') as f:
			audio_similarity_cache = pickle.load(f)
	else:
		audio_similarity_cache = {}
		
def run(clusters_filepath):
	clusters = list(gen_clusters.load_saved_combinations(clusters_filepath))

	cache_filepath = f'{DIRECTORY}/experiments/structural_distance/baseline2_audio_similarity/audio_similarity_cache.pkl'
	load_cache(cache_filepath)

	all_matrices = []
	for cluster in clusters:
		# Initialize a matrix to store pairwise similarities
		num_pieces = len(cluster)
		similarity_matrix = np.zeros((num_pieces, num_pieces))
		
		for i in range(num_pieces):
			for j in range(i + 1, num_pieces):  # Compute only the upper triangle
				audio_filepath1 = re.sub(r'^.*?/project', DIRECTORY, cluster[i]).replace('_augmented_graph_flat.pickle', '.mp3')
				audio_filepath2 = re.sub(r'^.*?/project', DIRECTORY, cluster[j]).replace('_augmented_graph_flat.pickle', '.mp3')
				
				similarity = get_audio_similarity(audio_filepath1, audio_filepath2)
				similarity_matrix[i, j] = similarity
				similarity_matrix[j, i] = similarity
		
		all_matrices.append(similarity_matrix)

	save_cache(cache_filepath)

	return np.mean(np.array(all_matrices), axis=0)

if __name__ == "__main__":
	clusters_filepath = f"{DIRECTORY}/experiments/structural_distance/structural_distance_music/input_clusters.pkl"
	print(run(clusters_filepath))
	
