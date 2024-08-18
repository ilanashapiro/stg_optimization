import os, sys
import pandas as pd
import random
import itertools
import json 
import pickle, shelve
import numpy as np
# import cupy as cp

# DIRECTORY = "/home/ilshapiro/project"
# DIRECTORY = '/home/ubuntu/project'
DIRECTORY = '/Users/ilanashapiro/Documents/constraints_project/project'
sys.path.append(DIRECTORY)
sys.path.append(f"{DIRECTORY}/centroid")

import simanneal_centroid_run, simanneal_centroid_helpers

def get_approx_end_time(csv_path):
		df = pd.read_csv(csv_path)
		if 'onset_seconds' in df.columns:
			return df['onset_seconds'].max()
		else:
			raise ValueError(f"'onset_seconds' column not found in {csv_path}")

def build_composers_dict(composer_graphs, base_dir=f"{DIRECTORY}/datasets"):
	for root, _, files in os.walk(base_dir):
		relative_path = os.path.relpath(root, base_dir)
		path_parts = relative_path.split(os.sep)

		if len(path_parts) >= 3:
			source_dir = path_parts[-2]
			composer_dir = path_parts[-3]
			
			for file in files:
				if file.endswith("_augmented_graph_flat.pickle"):
					full_path = os.path.join(root, file)
					csv_path = full_path.replace("_augmented_graph_flat.pickle", ".csv")

					if composer_dir in composer_graphs:
						duration = get_approx_end_time(csv_path)
						size = os.path.getsize(full_path)
						composer_graphs[composer_dir][source_dir].append((full_path, duration, size))
	return composer_graphs

def filter_by_duration_cluster(composer_graphs, window_len):
	# Flatten all values into a single list and sort them
		all_values = []
		value_to_keys = {}
		for key, tuples in composer_graphs.items():
			for tup in tuples:
				duration = tup[1] 
				all_values.append(duration)
				if duration not in value_to_keys:
					value_to_keys[duration] = []
				value_to_keys[duration].append((key, tup))
		all_values.sort()

		# Initialize variables to track the best windows and the corresponding keys and values
		max_count = 0
		best_windows = []

		# Use a sliding window to find all 7-second intervals with the most values
		start_idx = 0
		for end_idx in range(len(all_values)):
				while all_values[end_idx] - all_values[start_idx] > window_len:
					start_idx += 1
				count = end_idx - start_idx + 1
				best_windows.append((all_values[start_idx], all_values[end_idx]))
				if count > max_count:
					max_count = count
					best_windows = [(all_values[start_idx], all_values[end_idx])]
				elif count == max_count:
					best_windows.append((all_values[start_idx], all_values[end_idx]))
						
		# Collect the keys and their corresponding values for each best window
		results = []
		for window_start, window_end in best_windows:
			window_dict = {}
			for duration in all_values:
				if window_start <= duration <= window_end:
					for key, tup in value_to_keys[duration]:
						if key not in window_dict:
							window_dict[key] = []
						window_dict[key].append(tup)
			results.append(window_dict)
		return results

def filter_graphs_by_mean_duration(composer_graphs, max_diff=100000):#3000):
		all_durations = []
		for graphs in composer_graphs.values():
				for graph_list in graphs.values():
						all_durations.extend(duration for _, duration, size in graph_list)
		
		if not all_durations:
				return {}
		
		mean_duration = sum(all_durations) / len(all_durations)
		print("MEAN", mean_duration)
		print("MIN", min(all_durations))

		mean_duration = 160 # 160
		
		filtered_composer_graphs = {}
		for composer, graphs in composer_graphs.items():
				selected_graphs = []
				for source, graph_list in graphs.items():
						for graph, duration, size in graph_list:
								if abs(duration - mean_duration) <= max_diff:
										selected_graphs.append((graph, duration, size))
				
				if selected_graphs:
						filtered_composer_graphs[composer] = selected_graphs
		
		return filtered_composer_graphs

def load_saved_combinations(filename):
	combinations = set()
	try:
		with open(filename, "rb") as f:
			while True:
				try:
					combinations.add(pickle.load(f))
				except EOFError:
					break
	except FileNotFoundError:
		pass  # File doesn't exist yet, so no combinations saved
	return combinations

def log(combination, file):
		with open(file, "ab") as f:  # Open in append-binary mode
				pickle.dump(combination, f)

def update_cache(cache, key, value):
	cache[key] = value
	cache.sync() # Ensure the data is written to disk

def save_combination(combination, saved_combos, combo_file):
	if combination not in saved_combos:
		log(combination, combo_file)

def get_composer_from_path(path):
	parts = path.split(os.sep)
	return parts[parts.index('datasets') + 1]

def dist(G1, G2):
  STG_augmented_list = [G1, G2]
  listA_G, idx_node_mapping, node_metadata_dict = simanneal_centroid_helpers.pad_adj_matrices(STG_augmented_list)
  # A_G1, A_G2 = cp.asarray(listA_G[0]), cp.asarray(listA_G[1]) 
  A_G1, A_G2 = np.asarray(listA_G[0]), np.asarray(listA_G[1]) 
  _, struct_dist = simanneal_centroid_run.align_graph_pair(A_G1, A_G2, idx_node_mapping, node_metadata_dict)
  return struct_dist

# 0 is most similar
# ordering is from https://people.wku.edu/charles.smith/essays/2014EmpStudsArts.pdf table 7
composer_similarity_rank_order_from_composer = {
	"chopin": {
		"bach": 6,
		"mozart": 5,
		"beethoven": 3,
		"schubert": 1,
		"brahms": 2,
		"handel": 7,
		"haydn": 4,
		"chopin": 0
	},
	"haydn": {
		"bach": 6,
		"mozart": 1,
		"beethoven": 2,
		"schubert": 3,
		"brahms": 4,
		"handel": 7,
		"haydn": 0,
		"chopin": 5
	},
	"brahms": {
		"bach": 5,
		"mozart": 6,
		"beethoven": 2,
		"schubert": 1,
		"brahms": 0,
		"handel": 7,
		"haydn": 4,
		"chopin": 3
	},
	"schubert": {
		"bach": 6,
		"mozart": 3,
		"beethoven": 1,
		"schubert": 0,
		"brahms": 3, # mozart brahms are same
		"handel": 7,
		"haydn": 5,
		"chopin": 2
	},
	"beethoven": {
		"bach": 6,
		"mozart": 3,
		"beethoven": 0,
		"schubert": 1, #same
		"brahms": 3, # same
		"handel": 7,
		"haydn": 1,
		"chopin": 5
	},
	"mozart": {
		"bach": 5,
		"mozart": 0,
		"beethoven": 3,
		"schubert": 2,
		"brahms": 4,
		"handel": 7,
		"haydn": 1,
		"chopin": 6
	},
	"bach": {
		"bach": 0,
		"mozart": 4,
		"beethoven": 4,
		"schubert": 7,
		"brahms": 2,
		"handel": 1,
		"haydn": 3,
		"chopin": 6
	},
	"handel": {
		"bach": 1,
		"mozart": 4,
		"beethoven": 4,
		"schubert": 7,
		"brahms": 3,
		"handel": 0,
		"haydn": 2,
		"chopin": 6
	},
}

def is_valid_sequence_for_source_composer(sequence, source_composer_graph_fp, successes_file, failures_file, cache):
	failures = 0
	tolerance = 1
	with open(source_composer_graph_fp, 'rb') as f:
		G_source_composer = pickle.load(f)
	source_composer = get_composer_from_path(source_composer_graph_fp)
	composer_rank_order_from_source = composer_similarity_rank_order_from_composer[source_composer]

	min_distance = float('inf')

	for i in range(len(sequence) - 1):
		graph_fp1 = sequence[i]
		graph_fp2 = sequence[i + 1]

		cache_key1 = repr((source_composer_graph_fp, graph_fp1))
		cache_key1_rev = repr((graph_fp1, source_composer_graph_fp))
		if cache_key1 in cache:
			d1 = cache[cache_key1]
		elif cache_key1_rev in cache:
			d1 = cache[cache_key1_rev]
		else:
			with open(graph_fp1, 'rb') as f:
				G1 = pickle.load(f)
			d1 = float(dist(G_source_composer, G1))
			update_cache(cache, cache_key1, d1)

		cache_key2 = repr((source_composer_graph_fp, graph_fp2))
		cache_key2_rev = repr((graph_fp2, source_composer_graph_fp))
		if cache_key2 in cache:
			d2 = cache[cache_key2]
		elif cache_key2_rev in cache:
			d2 = cache[cache_key2_rev]
		else:
			with open(graph_fp2, 'rb') as f:
				G2 = pickle.load(f)
			d2 = float(dist(G_source_composer, G2))
			update_cache(cache, cache_key2, d2)

		composer1 = get_composer_from_path(graph_fp1)
		rank1 = composer_rank_order_from_source[composer1]
		composer2 = get_composer_from_path(graph_fp2)
		rank2 = composer_rank_order_from_source[composer2]

		# Update the minimum distance
		if min_distance > 0 and d1 > 0 and d2 > 0:
			if d1 < min_distance:
					min_distance = d1
			if d2 < min_distance:
					min_distance = d2

		filtered_composer_rank_order_from_source = {sink_composer: composer_rank_order_from_source[sink_composer] for sink_composer in composer_rank_order_from_source if sink_composer not in ['handel', 'brahms', 'haydn']}
		values = sorted(list(filtered_composer_rank_order_from_source.values()))
		lowest_filtered_rank = sorted(list(filtered_composer_rank_order_from_source.values()))[1] # exclude the singlar 0 value in every composer subdict for symmetric source-sink composer, e.g. mozart-mozart
		# Save the distance for the rank 1 composer
		if rank1 == lowest_filtered_rank:
			lowest_filtered_rank_dist = d1
		elif rank2 == lowest_filtered_rank:
			lowest_filtered_rank_dist = d2

		satisfies_order = d1 <= d2 and rank1 <= rank2 or d1 > d2 and rank1 > rank2
		# always_violates = rank1 == lowest_filtered_rank and d1 != min_distance or rank2 == lowest_filtered_rank and d2 != min_distance # always want to maintain that the no. 1 closest composer stays the closest
		
		# print(source_composer, composer1, d1, rank1, lowest_filtered_rank, failures)
		# print(source_composer, composer2, d2, rank2, lowest_filtered_rank, failures)
		# print(always_violates, rank1, rank2, min_distance, d1, d2)
		# print("---")
		
		if not satisfies_order and failures > tolerance:# always_violates or (not satisfies_order and failures > tolerance):
			# print(source_composer, composer1, composer2, rank1, rank2, d1, d2)
			return (False, failures)
		if not satisfies_order: # always_violates: 
			failures += 1

	if lowest_filtered_rank_dist != min_distance:
		return (False, failures)
	return (True, failures)

def find_valid_combination(clusters):
	for i, composers_dict in enumerate(clusters):
		updated_composers_dict = {}
		for composer, pieces_info_list in composers_dict.items():
			updated_composers_dict[composer] = [info[0] for info in pieces_info_list] # take out duration/graph size info
			# updated_composers_dict['handel'] = ['/home/ubuntu/project/datasets/handel/kunstderfuge/gigue_e_minor_(nc)werths/gigue_e_minor_(nc)werths_augmented_graph_flat.pickle']
		clusters[i] = updated_composers_dict
		
	cache = shelve.open("cache.shelve")
	
	# composers sorted alphabetically in each tuple of pieces
	all_combinations = set()
	for composers_dict in clusters:
		del composers_dict['brahms']
		del composers_dict['haydn']
		all_combinations.update([item for item in list(itertools.product(*(composers_dict[k] for k in sorted(composers_dict.keys()))))])
	
	# these are sets of tuples of strings
	failures_file = "failures.pkl"
	successes_file = "successes.pkl"
	saved_failures = load_saved_combinations(failures_file)
	saved_successes = load_saved_combinations(successes_file)
	
	for combination in all_combinations:
		if True: # combination not in saved_successes and combination not in saved_failures:
			entire_combo_valid = True
			total_failures = 0
			min_dist_fails_tolerance = 2
			total_order_fails_tolerance = float('inf')
			min_dist_fails = 0 
			for source_piece_fp in combination:
				result_bool, num_failures = is_valid_sequence_for_source_composer(combination, source_piece_fp, successes_file, failures_file, cache)
				if total_failures > total_order_fails_tolerance or (not result_bool and min_dist_fails > min_dist_fails_tolerance):
					entire_combo_valid = False
					break
				else:
					if not result_bool:
						min_dist_fails += 1
					total_failures += num_failures
			if entire_combo_valid and total_failures <= total_order_fails_tolerance:
				print("TRUE", total_failures)
				save_combination(combination, saved_successes, successes_file)
				saved_successes.add(combination)
				# sys.exit(0)
			else:
				# print("FALSE")
				save_combination(combination, saved_failures, failures_file)
				saved_failures.add(combination)
			# print()
	cache.close()

if __name__ == "__main__":
	filtered_composer_graphs_path = f"{DIRECTORY}/experiments/structural_distance/filtered_composer_graphs.txt" # this is a LIST of possible clusters
	if os.path.exists(filtered_composer_graphs_path):
			with open(filtered_composer_graphs_path, 'r') as file:
				filtered_composer_graphs = json.load(file)
				print("LOADED", filtered_composer_graphs_path)
	else:
		composer_graphs = {}
		with open(f'{DIRECTORY}/experiments/dataset_composers_in_phylogeny.txt', 'r') as file:
			for line in file:
				composer_graphs[line.strip()] = {'kunstderfuge':[], 'classical_piano_midi_db':[]}
		build_composers_dict(composer_graphs)
		composer_graphs = {composer: graphs for composer, graphs in composer_graphs.items() if len(graphs['classical_piano_midi_db']) + len(graphs['kunstderfuge']) > 0}
		
		selected_composers = ["bach", "mozart", "beethoven", "schubert", "brahms", "wagner", "verdi", "handel", "haydn", "chopin"]
		composer_graphs = {composer: graphs['classical_piano_midi_db'] + graphs['kunstderfuge'] for composer, graphs in composer_graphs.items() if composer in selected_composers}
		# filtered_composer_graphs = filter_graphs_by_mean_duration(composer_graphs)
		filtered_composer_graphs = filter_by_duration_cluster(composer_graphs)
		
		with open(filtered_composer_graphs_path, 'w') as file:
			json.dump(filtered_composer_graphs, file, indent=4)
			print("SAVED", filtered_composer_graphs_path)

	total_composers = 0
	print(find_valid_combination(filtered_composer_graphs[1:]))
	