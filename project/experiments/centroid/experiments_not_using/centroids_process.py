import os, sys, pickle, json
import numpy as np
import shelve
import torch 
from multiprocessing import Pool, current_process

# DIRECTORY = "/home/ubuntu/project"
DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
TIME_PARAM = '50s'
NUM_GPUS = 8 
sys.path.append(DIRECTORY)
sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/structural_distance/structural_distance_experiment")

import centroid_simanneal_gen 
import structural_distance_gen_clusters as st_gen_clusters
import simanneal_centroid_run, simanneal_centroid_helpers

def load_centroids():
	centroids_dict = {}
	for composer in ["bach", "beethoven", "haydn", "mozart"]:
		final_centroid_dir = f"{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}/{composer}"

		final_centroid_path = os.path.join(final_centroid_dir, "final_centroid.txt")
		final_centroid = np.loadtxt(final_centroid_path)
		print(f'Loaded: {final_centroid_path}')

		final_centroid_idx_node_mapping_path = os.path.join(final_centroid_dir, "final_idx_node_mapping.txt")
		with open(final_centroid_idx_node_mapping_path, 'r') as file:
			idx_node_mapping = json.load(file)
			idx_node_mapping = {int(k): v for k, v in idx_node_mapping.items()}
		print(f'Loaded: {final_centroid_idx_node_mapping_path}')

		# we just use the entire node metadata dict from approx_centroid_dir
		approx_centroid_dir = f"{DIRECTORY}/experiments/centroid/approx_centroids/approx_centroid_{TIME_PARAM}/{composer}"
		approx_centroid_node_metadata_dict_path = os.path.join(approx_centroid_dir, "node_metadata_dict.txt")
		with open(approx_centroid_node_metadata_dict_path, 'r') as file:
			node_metadata_dict = json.load(file)
		print(f'Loaded: {approx_centroid_node_metadata_dict_path}')
		
		centroids_dict[composer] = simanneal_centroid_helpers.adj_matrix_to_graph(final_centroid, idx_node_mapping, node_metadata_dict)
	
	return centroids_dict

def get_piece_name_from_path(path):
	filename = os.path.splitext(os.path.basename(path))[0]
	return filename.replace("_augmented_graph_flat", "")

def update_cache(cache, key, value):
	cache[key] = value
	cache.sync() # Ensure the data is written to disk
	
def get_composer_graphs_between_duration(selected_composer, min_duration, max_duration):
	composer_graphs = {}
	with open(f'{DIRECTORY}/experiments/dataset_composers_in_phylogeny.txt', 'r') as file:
		for line in file:
			composer_graphs[line.strip()] = {'kunstderfuge':[], 'classical_piano_midi_db':[]}
	
	composer_graphs = st_gen_clusters.build_composers_dict(composer_graphs)
	composer_graphs = {composer: graphs for composer, graphs in composer_graphs.items() if len(graphs['classical_piano_midi_db']) + len(graphs['kunstderfuge']) > 0}
	composer_graphs = {composer: graphs['classical_piano_midi_db'] + graphs['kunstderfuge'] for composer, graphs in composer_graphs.items() if composer == selected_composer}
	return [piece_info[0] for piece_info in centroid_simanneal_gen.filter_by_max_min_duration_cluster(composer_graphs, max_duration, min_duration=min_duration)[selected_composer]] # remove duration/STG file size info from the tuple

def get_composer_test_graph_filepaths():
	test_data_fp = f'composer_test_graph_filepaths_{TIME_PARAM}.json'
	if False:#os.path.exists(test_data_fp):
		with open(test_data_fp, 'r') as f:
			composer_test_graph_filepaths = json.load(f)
		print(f"Loaded {test_data_fp}")
	else:
		with open(input_pieces_path, 'r') as file:
			composer_training_pieces_dict = json.load(file) # we want to exclude these from the classification experiment as they comprise the centroid we're classifying into
		for composer, input_pieces_paths in composer_training_pieces_dict.items(): # we JUST include the piece name for comparison
			composer_training_pieces_dict[composer] = [get_piece_name_from_path(file_path) for file_path in input_pieces_paths]

		composer_test_graph_filepaths = {}
		for composer in composer_centroids_dict:
			training_pieces = set(composer_training_pieces_dict[composer])
			# remove the pieces we used to make the centroid
			composer_test_graph_filepaths[composer] = [filepath for filepath in get_composer_graphs_between_duration(composer, 0, 120) if get_piece_name_from_path(filepath) not in training_pieces]

		with open(test_data_fp, 'w') as f:
			json.dump(composer_test_graph_filepaths, f, indent=4)
		print(f"Saved {test_data_fp}")

	return composer_test_graph_filepaths

def classify_composer(test_composer, test_graph_filepaths, composer_centroids_dict, gpu_id):
	device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
	print(f"Process {current_process().name} running on GPU {gpu_id} for classifying composer {test_composer}")

	cache = shelve.open(f"{DIRECTORY}/experiments/centroid/cache_classify_{test_composer}_for_{TIME_PARAM}.shelve")
	classifier_counts = {}
	for composer in composer_centroids_dict.keys():
		classifier_counts[composer] = 0

	for STG_filepath in test_graph_filepaths:
		# see if STG_filepath is in cache. if not, proceed
		cache_key = STG_filepath
		if cache_key in cache:
			classified_composer = cache[cache_key]
			print(f"LOADED CLASSIFIED {STG_filepath} AS {classified_composer}")
		else:
			with open(STG_filepath, 'rb') as f:
				G = pickle.load(f)
			centroid_STGs = list(composer_centroids_dict.values())
			centroid_composers = list(composer_centroids_dict.keys())
			# first matrix in listA_G is the test graph and the rest are the centroids
			listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([G]+centroid_STGs)
			listA_G_tensors = [torch.tensor(matrix, device=device, dtype=torch.float32) for matrix in listA_G]
			A_G = listA_G_tensors[0] # the graph we want to test for classifying
			listA_g = listA_G_tensors[2:] # the list of padded centroid adj matrices, [2:] is no bach
			
			min_cost = float('inf')
			best_composer_index = None
			for i, A_g in enumerate(listA_g, start=1):
				_, cost = simanneal_centroid_run.align_graph_pair(A_G, A_g, idx_node_mapping, nodes_features_dict, device)
				if cost < min_cost:
					best_composer_index = i 
					min_cost = cost
				
			classified_composer = centroid_composers[best_composer_index]
			print(f"CLASSIFIED {STG_filepath} AS {classified_composer}")
			update_cache(cache, STG_filepath, classified_composer)

		classifier_counts[classified_composer] += 1
	print(f"RESULTS FOR {test_composer}: {classifier_counts}")
	cache.close()

if __name__ == "__main__":
	input_pieces_path = f"{DIRECTORY}/experiments/centroid/clusters/composer_centroid_input_graphs_{TIME_PARAM}.txt"
	
	composer_centroids_dict = load_centroids() 
	composer_centroids_dict = {k: composer_centroids_dict[k] for k in sorted(composer_centroids_dict)} # to ensure deterministic order
	composer_test_graph_filepaths = get_composer_test_graph_filepaths()

	# cache = shelve.open(f"{DIRECTORY}/experiments/centroid/cache.shelve")

	args = []
	for test_composer, test_graph_filepaths in composer_test_graph_filepaths.items():
		args.append((test_composer, test_graph_filepaths))
	tasks = [(test_composer, test_graph_filepaths, composer_centroids_dict, gpu_id) 
						for gpu_id, (test_composer, test_graph_filepaths) 
						in enumerate(args)]

	# Create a Pool with as many processes as there are GPUs
	pool = Pool(processes=NUM_GPUS)
	
	# Distribute the work across the GPUs
	results = pool.starmap(classify_composer, tasks)
	pool.close()
	pool.join()
			
	# cache.close()
