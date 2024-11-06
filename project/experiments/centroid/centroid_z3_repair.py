import os, sys, shutil, glob
import pickle, json
import numpy as np
from multiprocessing import Pool

# DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
DIRECTORY = "/home/ishapiro/project"
sys.path.append(f"{DIRECTORY}")
sys.path.append(f"{DIRECTORY}/centroid")

# import build_graph
import z3_matrix_projection_incremental as z3_repair
import simanneal_centroid_helpers as helpers

TIME_PARAM = "50s"
ABLATION_LEVEL = 4 # set to None if we don't want to do ablation

def repair_centroid(composer):
	if ABLATION_LEVEL:
		approx_centroid_dir = f"{DIRECTORY}/experiments/centroid/approx_centroids/approx_centroid_{TIME_PARAM}_ablation{ABLATION_LEVEL}/{composer}"
	else:
		approx_centroid_dir = f"{DIRECTORY}/experiments/centroid/approx_centroids/approx_centroid_{TIME_PARAM}/{composer}"

	approx_centroid_path = os.path.join(approx_centroid_dir, "centroid.txt")
	approx_centroid = np.loadtxt(approx_centroid_path)
	print(f'Loaded: {approx_centroid_path}')

	approx_centroid_idx_node_mapping_path = os.path.join(approx_centroid_dir, "idx_node_mapping.txt")
	with open(approx_centroid_idx_node_mapping_path, 'r') as file:
		idx_node_mapping = json.load(file)
		idx_node_mapping = {int(k): v for k, v in idx_node_mapping.items()}
	print(f'Loaded: {approx_centroid_idx_node_mapping_path}')

	approx_centroid_node_metadata_dict_path = os.path.join(approx_centroid_dir, "node_metadata_dict.txt")
	with open(approx_centroid_node_metadata_dict_path, 'r') as file:
		node_metadata_dict = json.load(file)
	print(f'Loaded: {approx_centroid_node_metadata_dict_path}')

	z3_repair.initialize_globals(approx_centroid, idx_node_mapping, node_metadata_dict)

	if ABLATION_LEVEL:
		final_centroid_dir = f'{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}_ablation{ABLATION_LEVEL}/{composer}'
	else:
		final_centroid_dir = f'{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}/{composer}'
	if not os.path.exists(final_centroid_dir):
		os.makedirs(final_centroid_dir)
	final_centroid_filename = f'{final_centroid_dir}/final_centroid.txt'
	final_idx_node_mapping_filename = f'{final_centroid_dir}/final_idx_node_mapping.txt'

	z3_repair.run(final_centroid_filename, final_idx_node_mapping_filename)

if __name__ == "__main__":
	if ABLATION_LEVEL:
		approx_centroids_dir = f"{DIRECTORY}/experiments/centroid/approx_centroids/approx_centroid_{TIME_PARAM}_ablation{ABLATION_LEVEL}"
	else:
		approx_centroids_dir = f"{DIRECTORY}/experiments/centroid/approx_centroids/approx_centroid_{TIME_PARAM}"
	approx_centroid_composers = [name for name in os.listdir(approx_centroids_dir) if os.path.isdir(os.path.join(approx_centroids_dir, name)) and name not in ["brahms", "chopin"]]

	with Pool() as pool:
		alignments = pool.map(repair_centroid, approx_centroid_composers)
	
	# uncomment this code for visualizing the centroid
	# centroid = np.loadtxt("final_centroid_50s/beethoven/final_centroid.txt")
	# with open("approx_centroid_50s/beethoven/node_metadata_dict.txt", 'r') as file:
	# 	node_metadata_dict = json.load(file)
	# with open("final_centroid_50s/beethoven/final_idx_node_mapping.txt", 'r') as file:
	# 	centroid_idx_node_mapping = {int(k): v for k, v in json.load(file).items()}
	
	# g = helpers.adj_matrix_to_graph(centroid, centroid_idx_node_mapping, node_metadata_dict)
	
	# layers_g = build_graph.get_unsorted_layers_from_graph_by_index(g)
	# build_graph.visualize([g], [layers_g])
	