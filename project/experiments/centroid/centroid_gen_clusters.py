import os, sys, shutil, glob
import pickle, json
import numpy as np
import torch
import cupy as cp
from multiprocessing import Pool, current_process, Queue

DIRECTORY = "/home/ubuntu/project"
sys.path.append(DIRECTORY)
sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/structural_dist")

import build_graph
import structural_distance_gen_clusters as st_gen_clusters
import simanneal_centroid_run, simanneal_centroid_helpers, simanneal_centroid

NUM_GPUS = 8 

def filter_by_max_duration_cluster(composer_graphs, max_duration):
		# Flatten all values into a single list and map durations to their keys and tuples
		value_to_keys = {}
		for key, tuples in composer_graphs.items():
				for tup in tuples:
						duration = tup[1] 
						if duration not in value_to_keys:
								value_to_keys[duration] = []
						value_to_keys[duration].append((key, tup))
		
		# Collect all values whose duration is under the specified limit
		filtered_durations = [duration for duration in value_to_keys.keys() if duration <= max_duration]

		# Create a dictionary to store the results
		results_dict = {}
		for duration in filtered_durations:
				for key, tup in value_to_keys[duration]:
						if key not in results_dict:
								results_dict[key] = []
						results_dict[key].append(tup)

		return results_dict

def filter_by_duration_window_cluster(composer_graphs, window_len):
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

			# we want all 7 composer in the 7-composer cluster, and want each composer to have min 16 pieces
			if len(window_dict) == 7:
				all_have_enough_graphs = True
				for graphs_list in window_dict.values():
					if len(graphs_list) < 16:
						all_have_enough_graphs = False
				if all_have_enough_graphs:
					results.append(window_dict)

	return results

# takes in a single list of pieces info, for a single composer from a cluster
def partition_composer_cluster_for_centroid(composer, pieces_info_list):
	pieces_info_list = sorted(pieces_info_list, key=lambda pieces_tuple: pieces_tuple[1]) 
	centroid_size = min(len(pieces_info_list) // 2, 5) # don't want to try computing centroids beyond 9 pieces rn
	if centroid_size < 5:
		centroid_size = 5

	step = len(pieces_info_list) // centroid_size
	initial_centroid_list = pieces_info_list[:step*centroid_size:step] # we end up with 22 pieces for longer clusters due to slicing
	remaining_pieces_list = [item for item in pieces_info_list if item not in initial_centroid_list]
	initial_centroid_list_durations = [item[1] for item in initial_centroid_list]
	remaining_list_durations = [item[1] for item in remaining_pieces_list]
	
	best_diff = abs(np.mean(initial_centroid_list_durations) - np.mean(remaining_list_durations))
	best_centroid_list = initial_centroid_list[:]
	
	# Try swapping elements to minimize the difference in means
	for i in range(len(initial_centroid_list)):
		for j in range(len(remaining_pieces_list)):
			# Swap
			new_centroid_list = initial_centroid_list[:]
			new_centroid_list[i], remaining_pieces_list[j] = remaining_pieces_list[j], new_centroid_list[i]
			
			new_remaining_list = [item for item in pieces_info_list if item not in new_centroid_list]
			new_centroid_list_durations = [item[1] for item in new_centroid_list]
			new_remaining_list_durations = [item[1] for item in new_remaining_list]
			
			new_diff = abs(np.mean(new_centroid_list_durations) - np.mean(new_remaining_list_durations))
			if new_diff < best_diff:
				best_diff = new_diff
				best_centroid_list = new_centroid_list[:]

	return best_centroid_list

def get_cluster():
	composer_graphs_cluster_path = f"{DIRECTORY}/experiments/centroid/composer_graphs_cluster.txt" # this is a SINGLE cluster
	if os.path.exists(composer_graphs_cluster_path):
		with open(composer_graphs_cluster_path, 'r') as file:
			cluster = json.load(file)
			print("LOADED", composer_graphs_cluster_path)
	else:
		composer_graphs = {}
		with open(f'{DIRECTORY}/experiments/dataset_composers_in_phylogeny.txt', 'r') as file:
			for line in file:
				composer_graphs[line.strip()] = {'kunstderfuge':[], 'classical_piano_midi_db':[]}
		
		composer_graphs = st_gen_clusters.build_composers_dict(composer_graphs)
		composer_graphs = {composer: graphs for composer, graphs in composer_graphs.items() if len(graphs['classical_piano_midi_db']) + len(graphs['kunstderfuge']) > 0}
		selected_composers = ["bach", "mozart", "beethoven", "schubert", "chopin", "brahms", "haydn", "handel"]# "wagner", "verdi"]
		composer_graphs = {composer: graphs['classical_piano_midi_db'] + graphs['kunstderfuge'] for composer, graphs in composer_graphs.items() if composer in selected_composers}

		# filtered_composer_graphs_cluster = filter_by_duration_window_cluster(composer_graphs, window_len=92)
		# for composer_graphs_dict in filtered_composer_graphs_cluster:
		# 	for composer, graphs in composer_graphs_dict.items():
		# 		print(composer, len(graphs))
		# 	print()

		# max_composer_graphs_cluster = None
		# max_graphs = 0
		# for composer_graphs_dict in filtered_composer_graphs_cluster:
		# 	total_graphs = sum(len(graphs) for graphs in composer_graphs_dict.values())
		# 	if total_graphs > max_graphs:
		# 		max_graphs = total_graphs
		# 		max_composer_graphs_cluster = composer_graphs_dict

		max_composer_graphs_cluster = filter_by_max_duration_cluster(composer_graphs, 35)
		# for composer, graphs in max_composer_graphs_cluster.items():
		# 	print(composer, len(graphs))
		# sys.exit(0)
		max_composer_graphs_cluster = {composer: graphs for composer, graphs in max_composer_graphs_cluster.items() if len(graphs) >= 5}
		
		with open(composer_graphs_cluster_path, 'w') as file:
			json.dump(max_composer_graphs_cluster, file, indent=4)
			print("SAVED", composer_graphs_cluster_path)
		
		cluster = max_composer_graphs_cluster
	
	return cluster 

def get_composer_centroid_input_graphs(cluster):
	composer_centroid_input_graphs_path = f"{DIRECTORY}/experiments/centroid/composer_centroid_input_graphs.txt" # this is a SINGLE cluster
	if os.path.exists(composer_centroid_input_graphs_path):
		with open(composer_centroid_input_graphs_path, 'r') as file:
			composer_centroid_input_graphs = json.load(file)
			print("LOADED", composer_centroid_input_graphs_path)
	else:
		composer_centroid_input_graphs = {}
		for composer, pieces_info_list in cluster.items():
			centroid_pieces_list = partition_composer_cluster_for_centroid(composer, pieces_info_list)
			composer_centroid_input_graphs[composer] = centroid_pieces_list

		# take out duration/graph size info so we just have paths
		for composer, pieces_info_list in composer_centroid_input_graphs.items():
			composer_centroid_input_graphs[composer] = [info[0] for info in pieces_info_list] 
		
		with open(composer_centroid_input_graphs_path, 'w') as file:
			json.dump(composer_centroid_input_graphs, file, indent=4)
			print("SAVED", composer_centroid_input_graphs_path)
	return composer_centroid_input_graphs		

def generate_initial_alignments(composer, STG_filepaths_and_augmented_list, gpu_id):
	device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
	print(f"Process {current_process().name} running on GPU {gpu_id} for centroid cluster {composer}")
	
	STG_filepath_list, STG_augmented_list = tuple(map(list, zip(*STG_filepaths_and_augmented_list)))
	listA_G, idx_node_mapping, node_metadata_dict = simanneal_centroid_helpers.pad_adj_matrices(STG_augmented_list)
	listA_G_tensors = [torch.tensor(matrix, device=device, dtype=torch.float32) for matrix in listA_G]
	min_loss_A_G, min_loss_A_G_list_index, min_loss, optimal_alignments = simanneal_centroid_run.initial_centroid_and_alignments(listA_G_tensors, idx_node_mapping, node_metadata_dict, device=device)
	
	# because these are tensors originally
	initial_centroid = min_loss_A_G.cpu().numpy() 
	initial_alignments = [alignment.cpu().numpy() for alignment in optimal_alignments]

	alignments_dir = f"{DIRECTORY}/experiments/centroid/initial_alignments/{composer}"
	initial_alignment_files = []
	for i in range(len(STG_augmented_list)):
		piece_name = os.path.splitext(os.path.basename(STG_filepath_list[i]))[0]
		initial_alignment_files.append(os.path.join(alignments_dir, f'initial_alignment_{piece_name}.txt'))
	
	intial_centroid_piece_name = os.path.splitext(os.path.basename(STG_filepath_list[min_loss_A_G_list_index]))[0]
	initial_centroid_file = os.path.join(alignments_dir, f'initial_centroid_{intial_centroid_piece_name}.txt')

	if not os.path.exists(alignments_dir):
		os.makedirs(alignments_dir)
	print(f"Created directory {alignments_dir}")
	
	np.savetxt(initial_centroid_file, initial_centroid, fmt='%d', delimiter=' ') # since min_loss_A_G is a tensor
	print(f'Saved: {initial_centroid_file}')
	
	for i, alignment in enumerate(initial_alignments):
		file_name = initial_alignment_files[i]
		np.savetxt(file_name, alignment, fmt='%d', delimiter=' ')
		print(f'Saved: {file_name}')
	
def get_saved_initial_alignments_and_centroid(composer, STG_filepaths_list):
	alignments_dir = f"{DIRECTORY}/experiments/centroid/initial_alignments/{composer}"
	initial_alignment_files = []
	for file_path in STG_filepaths_list:
		piece_name = os.path.splitext(os.path.basename(file_path))[0]
		initial_alignment_files.append(os.path.join(alignments_dir, f'initial_alignment_{piece_name}.txt'))
	initial_centroid_file = glob.glob(os.path.join(alignments_dir, '*initial_centroid*'))[0]
	
	alignments = [np.loadtxt(f) for f in initial_alignment_files]
	initial_centroid = np.loadtxt(initial_centroid_file)
	print(f'Loaded existing centroid and alignment files from {alignments_dir}')
	return initial_centroid, alignments

def generate_centroid(composer, initial_centroid, initial_alignments, listA_G, idx_node_mapping, node_metadata_dict, gpu_id):
	device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
	print(f"Process {current_process().name} running on GPU {gpu_id} for centroid cluster {composer}")

	# bc these are originally numpy and we can't convert to tensor till we get the device
	listA_G = [torch.tensor(A_G, device=device, dtype=torch.float32) for A_G in listA_G]
	initial_alignments = [torch.tensor(alignment, device=device, dtype=torch.float32) for alignment in initial_alignments]
	initial_centroid = torch.tensor(initial_centroid, device=device, dtype=torch.float32)
	aligned_listA_G = list(map(simanneal_centroid.align, initial_alignments, listA_G))

	centroid_annealer = simanneal_centroid.CentroidAnnealer(initial_centroid, aligned_listA_G, idx_node_mapping, node_metadata_dict, device=device)
	centroid_annealer.Tmax = 2.5
	centroid_annealer.Tmin = 0.05 
	centroid_annealer.steps = 1000
	approx_centroid, loss = centroid_annealer.anneal()
	approx_centroid = approx_centroid.cpu().numpy() # convert from tensor -> numpy
	loss = loss.item() # convert from tensor -> numpy
	
	approx_centroid, final_idx_node_mapping = simanneal_centroid_helpers.remove_unnecessary_dummy_nodes(approx_centroid, idx_node_mapping, node_metadata_dict)
	approx_centroid_dir = f"{DIRECTORY}/experiments/centroid/approx_centroid/{composer}"
	if not os.path.exists(approx_centroid_dir):
		os.makedirs(approx_centroid_dir)

	approx_centroid_path = os.path.join(approx_centroid_dir, "centroid.txt")
	np.savetxt(approx_centroid_path, approx_centroid, fmt='%d', delimiter=' ')
	print(f'Saved: {approx_centroid_path}')

	approx_centroid_idx_node_mapping_path = os.path.join(approx_centroid_dir, "idx_node_mapping.txt")
	with open(approx_centroid_idx_node_mapping_path, 'w') as file:
		json.dump(final_idx_node_mapping, file)
	print(f'Saved: {approx_centroid_idx_node_mapping_path}')

	approx_centroid_node_metadata_dict_path = os.path.join(approx_centroid_dir, "node_metadata_dict.txt")
	with open(approx_centroid_node_metadata_dict_path, 'w') as file:
		json.dump(node_metadata_dict, file)
	print(f'Saved: {approx_centroid_node_metadata_dict_path}')
	
	approx_centroid_loss_path = os.path.join(approx_centroid_dir, "loss.txt")
	np.savetxt(approx_centroid_loss_path, [loss], fmt='%d')
	print(f'Saved: {approx_centroid_loss_path}')

if __name__ == "__main__":
	# def delete_dirs_with_substring(directory, substring):
	# 	for root, dirs, _ in os.walk(directory, topdown=False):
	# 			for dir_name in dirs:
	# 				if substring == dir_name:
	# 						dir_path = os.path.join(root, dir_name)
	# 						print(f"Deleting directory {dir_path}")
	# 						shutil.rmtree(dir_path)

	# directory = '/home/ubuntu/project/datasets'
	# substring = 'alignments'
	# delete_dirs_with_substring(directory, substring)
	# sys.exit(0)
	
	cluster = get_cluster()
	composer_centroid_input_graphs = get_composer_centroid_input_graphs(cluster)
	

	# ------------------FOR GENERATING INITIAL ALIGNMENTS/CENTROID------------------------------------------------
	# for composer, centroid_input_pieces_list in composer_centroid_input_graphs.items():
	# 	STG_filepaths_and_augmented_list = []
	# 	for graph_filepath in centroid_input_pieces_list:
	# 		with open(graph_filepath, 'rb') as f:
	# 			STG_filepaths_and_augmented_list.append((graph_filepath, pickle.load(f)))
	# 	composer_centroid_input_graphs[composer] = STG_filepaths_and_augmented_list
	# tasks = [(composer, STG_filepaths_and_augmented_list, gpu_id) 
	# 					for gpu_id, (composer, STG_filepaths_and_augmented_list) 
	# 					in enumerate(composer_centroid_input_graphs.items())]

	# # Create a Pool with as many processes as there are GPUs
	# pool = Pool(processes=NUM_GPUS)
	
	# # Distribute the work across the GPUs
	# results = pool.starmap(generate_initial_alignments, tasks)
	# pool.close()
	# pool.join()
	# ----------------------------------------------------------------------------------------------------------------

	# ------FOR LOADING EXISTING INITIAL ALIGNMENTS/CENTROID AFTER THEY'RE GENERATED, AND THEN GENERATING CENTROIDS----
	gen_centroid_info = []
	for composer, centroid_input_pieces_list in list(composer_centroid_input_graphs.items()):
		STG_augmented_list = []
		STG_filepaths_list = []
		
		for graph_filepath in centroid_input_pieces_list:
			with open(graph_filepath, 'rb') as f:
				STG_augmented_list.append(pickle.load(f))
			STG_filepaths_list.append(graph_filepath)
		
		listA_G, idx_node_mapping, node_metadata_dict = simanneal_centroid_helpers.pad_adj_matrices(STG_augmented_list)
		initial_centroid, initial_alignments = get_saved_initial_alignments_and_centroid(composer, STG_filepaths_list)
		gen_centroid_info.append((composer, initial_centroid, initial_alignments, listA_G, idx_node_mapping, node_metadata_dict))

	tasks = [(composer, initial_centroid, initial_alignments, listA_G, idx_node_mapping, node_metadata_dict, gpu_id) 
						for gpu_id, (composer, initial_centroid, initial_alignments, listA_G, idx_node_mapping, node_metadata_dict) 
						in enumerate(gen_centroid_info)]

	# Create a Pool with as many processes as there are GPUs
	pool = Pool(processes=NUM_GPUS)
	
	# Distribute the work across the GPUs
	results = pool.starmap(generate_centroid, tasks)
	pool.close()
	pool.join()
	# ----------------------------------------------------------------------------------------------------------------