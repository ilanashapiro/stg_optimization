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
		
		grakel_graphs = []
		for adjacency_matrix in listA_G:
			labels = {i: f'node' for i in range(adjacency_matrix.shape[0])}  # Dummy labels
			edge_labels = {(i, j): 'edge' for i in range(adjacency_matrix.shape[0]) for j in range(adjacency_matrix.shape[1]) if adjacency_matrix[i, j] > 0}  # Dummy edge labels
			graph = Graph(adjacency_matrix, node_labels=labels, edge_labels=edge_labels)
			grakel_graphs.append(graph)

		# Initialize the Weisfeiler-Lehman kernel with a base kernel
		wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=NeighborhoodSubgraphPairwiseDistance)

		# Store average similarities for each graph when treated as the centroid
		scores = []

		# Iterate over each graph in the corpus, treating each as the candidate centroid
		for i in range(len(grakel_graphs) - 1):
			# Create a new list where each candidate graph becomes the "test centroid" (first is original centroid, then duplicating the test graphs to replace original centroid)
			test_grakel_graphs = [grakel_graphs[i]] + grakel_graphs[1:]
			
			# Compute the kernel matrix
			kernel_matrix = wl_kernel.fit_transform(test_grakel_graphs)
			dist_matrix = 1 - kernel_matrix
			
			# Calculate the score, excluding the first graph (the "centroid" itself)
			mean_dist = np.mean(dist_matrix[0, 1:])
			std_dist  = np.std(dist_matrix[0, 1:])
			score = mean_dist # * std_dist 
			scores.append(score)

			print(f"Score when graph {i} is treated as the centroid: {score}")

		best_score = min(scores)
		best_graph_index = scores.index(best_score)

		print(f"The most representative graph in the corpus is graph {best_graph_index} with a score of {best_score}.")


# RESULT WHEN DOING DIFFERENCE MATRIX (1-KERNEL) for WL, NORMALLIZED, 5 ITER, BASE=NeighborhoodSubgraphPairwiseDistance
# graph 0 is the candidate centroid in each group. we want the lowest score (mean * std) of the distances
# Score when graph 0 is treated as the centroid: 0.003151294049742592
# Score when graph 1 is treated as the centroid: 0.1016670509498339
# Score when graph 2 is treated as the centroid: 0.10110109626350668
# Score when graph 3 is treated as the centroid: 0.09997122179854936
# Score when graph 4 is treated as the centroid: 0.10041603586223657
# Score when graph 5 is treated as the centroid: 0.0968413145911526
# The most representative graph in the corpus is graph 0 with a score of 0.003151294049742592.
# Score when graph 0 is treated as the centroid: 0.011726409974786163
# Score when graph 1 is treated as the centroid: 0.09624173461271755
# Score when graph 2 is treated as the centroid: 0.1015314478428872
# Score when graph 3 is treated as the centroid: 0.10471013477566422
# Score when graph 4 is treated as the centroid: 0.1018918443151187
# Score when graph 5 is treated as the centroid: 0.10288991391863234
# Score when graph 6 is treated as the centroid: 0.11193667171465782
# Score when graph 7 is treated as the centroid: 0.09649614434757735
# Score when graph 8 is treated as the centroid: 0.10147834310717792
# The most representative graph in the corpus is graph 0 with a score of 0.011726409974786163.
# Score when graph 0 is treated as the centroid: 0.010636458072726217
# Score when graph 1 is treated as the centroid: 0.11153544713829874
# Score when graph 2 is treated as the centroid: 0.1002269880461082
# Score when graph 3 is treated as the centroid: 0.10047570988006804
# The most representative graph in the corpus is graph 0 with a score of 0.010636458072726217.
# Score when graph 0 is treated as the centroid: 0.0027268160194961408
# Score when graph 1 is treated as the centroid: 0.10671758441211408
# Score when graph 2 is treated as the centroid: 0.10506754513416786
# Score when graph 3 is treated as the centroid: 0.10842399782420022
# Score when graph 4 is treated as the centroid: 0.10800274235973978
# The most representative graph in the corpus is graph 0 with a score of 0.0027268160194961408.