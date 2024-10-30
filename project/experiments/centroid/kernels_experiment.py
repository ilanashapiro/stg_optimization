import os, sys, pickle, json, re
import numpy as np
import spectral_experiment

from grakel.kernels import WeisfeilerLehman, NeighborhoodSubgraphPairwiseDistance, RandomWalk
from grakel import Graph, NeighborhoodHash, SubgraphMatching, VertexHistogram, WeisfeilerLehmanOptimalAssignment 

DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
TIME_PARAM = '50s'
NUM_GPUS = 8 
sys.path.append(DIRECTORY)
sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/structural_distance/structural_distance_experiment")

import build_graph
import structural_distance_gen_clusters as st_gen_clusters
import simanneal_centroid_run, simanneal_centroid_helpers, simanneal_centroid
	
if __name__ == "__main__":
	centroid_path = f"{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}"
	training_pieces_path = f"{DIRECTORY}/experiments/centroid/clusters/composer_centroid_input_graphs_{TIME_PARAM}.txt"
	
	composer_centroids_dict = spectral_experiment.load_centroids() 
	composer_centroids_dict = {k: composer_centroids_dict[k] for k in sorted(composer_centroids_dict)} # to ensure deterministic order

	with open(training_pieces_path, 'r') as file:
		composer_training_pieces_paths = json.load(file)
		def load_graph(file_path):
			with open(file_path, 'rb') as f:
				graph = pickle.load(f)
			return graph
		composer_training_pieces_dict = {}
		for composer, filepaths in composer_training_pieces_paths.items():
			composer_training_pieces_dict[composer] = [load_graph(re.sub(r'^.*?/project', DIRECTORY, file_path)) for file_path in filepaths]

	for composer, centroid in composer_centroids_dict.items():
		training_pieces = composer_training_pieces_dict[composer]
		listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([centroid] + training_pieces)
		A_g = listA_G[0] # the centroid graph
		listA_G = listA_G[1:] # separate the input graphs from the centroid
		
		# grakel_graphs = [Graph(A_g)] + [Graph(A_G) for A_G in listA_G]
		grakel_graphs = []
		listl = [A_g] + listA_G
		for adjacency_matrix in listl:
			labels = {i: f'node' for i in range(adjacency_matrix.shape[0])}  # Dummy labels
			edge_labels = {(i, j): 'edge' for i in range(adjacency_matrix.shape[0]) for j in range(adjacency_matrix.shape[1]) if adjacency_matrix[i, j] > 0}  # Dummy edge labels
			graph = Graph(adjacency_matrix, node_labels=labels, edge_labels=edge_labels)
			grakel_graphs.append(graph)

		# Initialize the Weisfeiler-Lehman kernel with a base kernel
		wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=WeisfeilerLehmanOptimalAssignment)

		# Store average similarities for each graph when treated as the centroid
		scores = []

		# Iterate over each graph in the corpus, treating each as the candidate centroid
		for i in range(len(grakel_graphs) - 1):
			# Create a new list where each candidate graph becomes the "test centroid" (first is original centroid, then duplicating the test graphs to replace original centroid)
			test_grakel_graphs = [grakel_graphs[i]] + grakel_graphs[1:]
			
			# Compute the kernel matrix
			kernel_matrix = wl_kernel.fit_transform(test_grakel_graphs)
			
			# Calculate the score, excluding the first graph (the "centroid" itself)
			mean_similarity = np.mean(kernel_matrix[0, 1:])
			std_similarity = np.std(kernel_matrix[0, 1:])
			score = mean_similarity * std_similarity
			scores.append(score)

			print(f"Scoore when graph {i} is treated as the centroid: {score}")

		# Identify which graph has the highest average similarity to the corpus
		best_score = max(scores)
		best_graph_index = scores.index(best_score)

		print(f"The most representative graph in the corpus is graph {best_graph_index} with a score of {score}.")