import os, sys, shutil, glob, math
import pickle, json
import numpy as np, pandas as pd
import torch
# import cupy as cp
from multiprocessing import Pool, current_process, Queue

# DIRECTORY = "/home/ubuntu/project"
DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
# DIRECTORY = "/home/ilshapiro/project"
sys.path.append(DIRECTORY)
sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/structural_distance/structural_distance_experiment")

import build_graph
import simanneal_centroid_run, simanneal_centroid_helpers, simanneal_centroid

NUM_GPUS = 8 

'''
This file generates the approximate centroids for each composer corpus for the Musical Eval experiment in Section 6.2 of the paper (Alkan, Chopin, Haydn, Mozart) using the
bi-level simulated annealing procedure for the Centroid Annealer in project/centroid/simanneal_centroid.py
First we generate initial alignments for the Centroid Annealer with the Graph Alignment Annealer (2000 steps, max temp 2, min temp 0.01)
Then we run the Centroid Annealer (1000 steps, max temp 2.5, min temp 0.05).
At each iteration of the centroid annealer, we run the Graph Alignment Annealer, starting at 500 steps, max temp 1, min temp 0.01, 
and ending at 5 steps, max temp 0.05, min temp 0.01 as the Centroid Annealer's loss converges
'''
def get_approx_end_time(csv_path):
	df = pd.read_csv(csv_path)
	if 'onset_seconds' in df.columns:
		return df['onset_seconds'].max()
	else:
		raise ValueError(f"'onset_seconds' column not found in {csv_path}")

# from project/centroid/simanneal_centroid_tests.py
def find_n_pickles_within_size_by_composer(min_n=4, max_n=14, max_file_size=math.inf):
	composer_files = {}
	for root, _, files in os.walk(DIRECTORY):
		for file in files:
			if file.endswith('_augmented_graph_flat.pickle'):
				path_parts = root.split(os.sep)
				composer = path_parts[-3]  # assuming the structure is: /home/ubuntu/project/datasets/{COMPOSER}/...
				
				file_path = os.path.join(root, file)
				file_size = os.path.getsize(file_path)
				csv_path = file_path.replace("_augmented_graph_flat.pickle", ".csv")
				duration = get_approx_end_time(csv_path)
				
				if composer not in composer_files:
					composer_files[composer] = []
				
				if file_size <= max_file_size:
					composer_files[composer].append((file_path, file_size, duration))

	smallest_files_by_composer = {}
	for composer, file_info in composer_files.items():
		if len(file_info) >= min_n:
			file_info.sort(key=lambda x: x[1])
			smallest_files_by_composer[composer] = file_info[:max_n]
	
	return smallest_files_by_composer

def get_composer_centroid_input_graphs():
	composer_centroid_input_graphs_dir = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/corpora"
	composer_centroid_input_graphs_path = composer_centroid_input_graphs_dir + "/composer_centroid_input_graphs.txt"
	
	if os.path.exists(composer_centroid_input_graphs_path):
		with open(composer_centroid_input_graphs_path, 'r') as file:
			composer_centroid_input_graphs = json.load(file)
			print("LOADED", composer_centroid_input_graphs_path)
	else:
		if not os.path.exists(composer_centroid_input_graphs_dir):
			os.makedirs(composer_centroid_input_graphs_dir)
		composer_centroid_input_graphs = {}
		corpora_composers_dict = find_n_pickles_within_size_by_composer(min_n=7, max_n=14, max_file_size=50000)

		for composer, files in corpora_composers_dict.items():
			print(f"Composer: {composer.upper()}. Corpus size: {len(files)}")
			for file in files:
					print(f" File: {file[0]}, Size: {file[1]} bytes, Duration: {file[2]} sec")
					
		for composer, info in corpora_composers_dict.items():
			info = tuple(list(x) for x in zip(*info)) # [(filename, size, duration)] -> ([filename], [size], [duration])
			centroid_pieces_list = info[0]
			composer_centroid_input_graphs[composer] = centroid_pieces_list
		
		with open(composer_centroid_input_graphs_path, 'w') as file:
			json.dump(composer_centroid_input_graphs, file, indent=4)
			print("SAVED", composer_centroid_input_graphs_path)
	
	return composer_centroid_input_graphs		

def generate_initial_alignments(composer, STG_filepaths_and_augmented_list, gpu_id):
	device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
	print(f"Process {current_process().name} running on GPU {gpu_id} for centroid cluster {composer}")
	
	STG_filepath_list, STG_augmented_list = tuple(map(list, zip(*STG_filepaths_and_augmented_list)))
	listA_G, idx_node_mapping, node_metadata_dict = simanneal_centroid_helpers.pad_adj_matrices(STG_augmented_list)
	listA_G_tensors = [torch.tensor(matrix, device=device, dtype=torch.float64) for matrix in listA_G]
	min_loss_A_G, min_loss_A_G_list_index, min_loss, optimal_alignments = simanneal_centroid_run.initial_centroid_and_alignments(listA_G_tensors, idx_node_mapping, node_metadata_dict, device=device)

	# because these are tensors originally
	initial_centroid = min_loss_A_G.cpu().numpy() 
	initial_alignments = [alignment.cpu().numpy() for alignment in optimal_alignments]
	alignments_dir = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/initial_alignments/{composer}"
	print("ALIGNMENTS DIR", alignments_dir)

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
	alignments_dir = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/initial_alignments/{composer}"

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
	listA_G = [torch.tensor(A_G, device=device, dtype=torch.float64) for A_G in listA_G]
	initial_alignments = [torch.tensor(alignment, device=device, dtype=torch.float64) for alignment in initial_alignments]
	initial_centroid = torch.tensor(initial_centroid, device=device, dtype=torch.float64)
	aligned_listA_G = list(map(simanneal_centroid.align_torch, initial_alignments, listA_G))

	centroid_annealer = simanneal_centroid.CentroidAnnealer(initial_centroid, aligned_listA_G, idx_node_mapping, node_metadata_dict, device=device)
	centroid_annealer.Tmax = 2.5
	centroid_annealer.Tmin = 0.05 
	centroid_annealer.steps = 1000
	approx_centroid, loss = centroid_annealer.anneal()
	approx_centroid = approx_centroid.cpu().numpy() # convert from tensor -> numpy
	loss = loss.item() # convert from tensor -> numpy
	
	approx_centroid, final_idx_node_mapping = simanneal_centroid_helpers.remove_unnecessary_dummy_nodes(approx_centroid, idx_node_mapping, node_metadata_dict)
	approx_centroid_dir = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/approx_centroids/{composer}"
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
	
	composer_centroid_input_graphs = get_composer_centroid_input_graphs()
	
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