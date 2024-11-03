import os, sys, pickle, json, re
import numpy as np
import spectral_experiment

from grakel.kernels import WeisfeilerLehman, NeighborhoodSubgraphPairwiseDistance, RandomWalk, MultiscaleLaplacian, ShortestPath
from grakel import Graph, NeighborhoodHash, SubgraphMatching, VertexHistogram, WeisfeilerLehmanOptimalAssignment 

# DIRECTORY = "/home/ubuntu/project"
DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
# DIRECTORY = "/home/ilshapiro/project"
TIME_PARAM = '50s'

sys.path.append(f"{DIRECTORY}/centroid")
import simanneal_centroid_helpers

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

		wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=NeighborhoodSubgraphPairwiseDistance)
		# random_walk_kernel = RandomWalk(normalize=True) # Intractable for my graphs
		# shortest_path_kernel = ShortestPath(normalize=True) # Intractable for my graphs
		nspd_kernel = NeighborhoodSubgraphPairwiseDistance(normalize=True)
		# neighborhood_hash_kernel = NeighborhoodHash(normalize=True) # Not accurate enough

		scores = []

		# Iterate over each graph in the corpus, treating each as the candidate centroid
		for i in range(len(grakel_graphs) - 1):
			# Create a new list where each candidate graph becomes the "test centroid" (first is original centroid, then duplicating the test graphs to replace original centroid)
			test_grakel_graphs = [grakel_graphs[i]] + grakel_graphs[1:]
			
			# Compute the kernel matrix
			kernel_matrix = nspd_kernel.fit_transform(test_grakel_graphs)
			dist_matrix = 1 - kernel_matrix
			
			# Calculate the score, excluding the first graph (the "centroid" itself)
			mean_dist = np.mean(dist_matrix[0, 1:])
			std_dist  = np.std(dist_matrix[0, 1:])
			score = mean_dist * std_dist 
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

# RESULT WHEN DOING NeighborhoodSubgraphPairwiseDistance -- probably better than WL (+NSPD) beacuse the graphs are unlabeled
# graph 0 is the candidate centroid in each group. we want the lowest score (mean * std) of the distances
# Score when graph 0 is treated as the centroid: 0.003154250368310091
# Score when graph 1 is treated as the centroid: 0.0604748461059153
# Score when graph 2 is treated as the centroid: 0.05894963562701486
# Score when graph 3 is treated as the centroid: 0.05984226466131365
# Score when graph 4 is treated as the centroid: 0.05876992330008765
# Score when graph 5 is treated as the centroid: 0.060427309621139354
# The most representative graph in the corpus is graph 0 with a score of 0.003154250368310091.
# Score when graph 0 is treated as the centroid: 0.006875488584386819
# Score when graph 1 is treated as the centroid: 0.05791920578209522
# Score when graph 2 is treated as the centroid: 0.06292375085232585
# Score when graph 3 is treated as the centroid: 0.06408684111686977
# Score when graph 4 is treated as the centroid: 0.06288668431496344
# Score when graph 5 is treated as the centroid: 0.06124524001200343
# Score when graph 6 is treated as the centroid: 0.06376956324266032
# Score when graph 7 is treated as the centroid: 0.057058004807151244
# Score when graph 8 is treated as the centroid: 0.06104831512247414
# The most representative graph in the corpus is graph 0 with a score of 0.006875488584386819.
# Score when graph 0 is treated as the centroid: 0.007206712690889715
# Score when graph 1 is treated as the centroid: 0.06993411213175489
# Score when graph 2 is treated as the centroid: 0.06180567372000269
# Score when graph 3 is treated as the centroid: 0.06189796760891367
# The most representative graph in the corpus is graph 0 with a score of 0.007206712690889715.
# Score when graph 0 is treated as the centroid: 0.002299858453899415
# Score when graph 1 is treated as the centroid: 0.06507243529247023
# Score when graph 2 is treated as the centroid: 0.06448303589014688
# Score when graph 3 is treated as the centroid: 0.06738862540061866
# Score when graph 4 is treated as the centroid: 0.06418998623691283
# The most representative graph in the corpus is graph 0 with a score of 0.002299858453899415.