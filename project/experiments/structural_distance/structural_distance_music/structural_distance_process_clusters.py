import os, sys, re
import pickle, shelve
import structural_distance_gen_clusters as gen_clusters
import numpy as np
from collections import defaultdict

from grakel.kernels import WeisfeilerLehman
from grakel import Graph, NeighborhoodHash

import simanneal_centroid_helpers

DIRECTORY = '/home/ilshapiro/project'
# DIRECTORY = "/home/ubuntu/project"
# DIRECTORY = '/Users/ilanashapiro/Documents/constraints_project/project'

'''
This file runs the Graph Alignment Annealer on the STG sets generated in structural_distance_gen_clusters 
to get results for the structural distance music evaluation experiment in Section 6.1 of the paper
The Graph Alignment annealer is implemented in project/centroid/simanneal_centroid.py
Number of annealer steps is 2000, max temp 2, min temp 0.01

Or, if we're doing the WL Kernels basline, the distance metric on the STG sets is the WL Kernel with Neighborhood Hash as base kernel
'''
def create_distance_matrix(cluster, cache, kernel_experiment=False, ablation_level=None):
	n = len(cluster)
	distance_matrix = distance_matrix = np.zeros((n, n))
	
	for i in range(n):
		for j in range(i, n):
			if i == j:
				distance_matrix[i][j] = 0
			else:
				graph_fp1, graph_fp2 = cluster[i], cluster[j]

				if ablation_level:
					if ablation_level < 0 or ablation_level > 4:
						raise Exception("Ablation levels should be 1-4 (5 levels is not an ablation, it's a complete graph)")
					new_suffix = f"_ablation_{ablation_level}level_flat.pickle"
					graph_fp1 = graph_fp1[:-len("_flat.pickle")] + new_suffix
					graph_fp2 = graph_fp2[:-len("_flat.pickle")] + new_suffix

				cache_key = repr((graph_fp1, graph_fp2))
				cache_key_rev = repr((graph_fp2, graph_fp1))
				if cache_key in cache:
					d = cache[cache_key]
				elif cache_key_rev in cache:
					d = cache[cache_key_rev]
				else:
					graph_fp1 = re.sub(r'^.*?/project', DIRECTORY, graph_fp1)
					graph_fp2 = re.sub(r'^.*?/project', DIRECTORY, graph_fp2)
					with open(graph_fp1, 'rb') as f:
						G1 = pickle.load(f)
					with open(graph_fp2, 'rb') as f:
						G2 = pickle.load(f)

					if kernel_experiment:
						listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([G1, G2])
						grakel_graphs = []
						for adjacency_matrix in listA_G:
							labels = {i: f'node{i}' for i in range(adjacency_matrix.shape[0])}  # Dummy labels
							edge_labels = {(i, j): 'edge{i},{j}' for i in range(adjacency_matrix.shape[0]) for j in range(adjacency_matrix.shape[1]) if adjacency_matrix[i, j] > 0}  # Dummy edge labels
							graph = Graph(adjacency_matrix, node_labels=labels, edge_labels=edge_labels)
							grakel_graphs.append(graph)

						wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=NeighborhoodHash)
						kernel_matrix = wl_kernel.fit_transform(grakel_graphs)
						kernel_value = kernel_matrix[0, 1]  # Kernel similarity between G1 and G2
						d = 1 - kernel_value  # Kernel distance
					else:
						d = float(gen_clusters.dist(G1, G2))
					gen_clusters.update_cache(cache, cache_key, d)
				
				# print(graph_fp1, graph_fp2, d)
				distance_matrix[i][j] = d
				distance_matrix[j][i] = d  # Ensure symmetry
	
	return distance_matrix

def reorder_cluster_to_reference(cluster):
	ref_order = ['bach', 'mozart', 'beethoven', 'schubert', 'brahms', 'handel', 'haydn', 'chopin']
	order_map = {substring: index for index, substring in enumerate(ref_order)}
	return tuple(sorted(cluster, key=lambda path: order_map[gen_clusters.get_composer_from_path(path)]))

# incorrect min dist tolerance = 3, no brahms, no haydn
# [[ 0.         70.49215851 69.70609372 73.56806085 74.2404242 ]
#  [70.49215851  0.         58.02445702 62.67288262 63.18674252]
#  [69.70609372 58.02445702  0.         61.78722697 62.41696957]
#  [73.56806085 62.67288262 61.78722697  0.         66.45057542]
#  [74.2404242  63.18674252 62.41696957 66.45057542  0.        ]]
#       Y            N          N          Y           N

# incorrect min dist tolerance = 2, no brahms, no haydn
# [[ 0.         74.55346015 69.94919238 71.11178379 76.84279747]
#  [74.55346015  0.         61.65969699 62.88786701 69.55827472]
#  [69.94919238 61.65969699  0.         57.58932631 64.47282313]
#  [71.11178379 62.88786701 57.58932631  0.         65.4055331 ]
#  [76.84279747 69.55827472 64.47282313 65.4055331   0.        ]]
#     Y(1/2)     N (almost)    Y            Y        N (one off)


# incorrect min dist tolerance = 1, no brahms, no haydn
# [[ 0.         74.55311233 70.63537418 69.45984904 77.83526235]
#  [74.55311233  0.         63.47440429 61.48983656 70.9365914 ]
#  [70.63537418 63.47440429  0.         57.04384279 67.78642932]
#  [69.45984904 61.48983656 57.04384279  0.         66.29479618]
#  [77.83526235 70.9365914  67.78642932 66.29479618  0.        ]]
#      N            Y           Y            Y           Y

# incorrect min dist tolerance = 2, no brahms, no haydn, total ordering fails tolerance = 1
# [[ 0.         74.55364163 70.25934309 71.65572227 80.2026715 ]
#  [74.55364163  0.         61.64754595 63.21782189 72.89690075]
#  [70.25934309 61.64754595  0.         58.31028065 68.60095034]
#  [71.65572227 63.21782189 58.31028065  0.         70.00442844]
#  [80.2026715  72.89690075 68.60095034 70.00442844  0.        ]]
#      Y(1/2)       N           Y          Y          N (one off)

# incorrect min dist tolerance = 2, no brahms, no haydn, total ordering fails tolerance = 1 
# [[ 0.         69.79983433 67.18422074 68.63973048 79.05849818]
#  [69.79983433  0.         60.32597285 62.08864631 73.77669009]
#  [67.18422074 60.32597285  0.         59.65810022 71.40307056]
#  [68.63973048 62.08864631 59.65810022  0.         73.19836064]
#  [79.05849818 73.77669009 71.40307056 73.19836064  0.        ]]
#    Y(1/2)        N           Y            Y           N (one off) 

def run(clusters_path, kernel_experiment=False):
	clusters = gen_clusters.load_saved_combinations(clusters_path)
	ablation_level = None # set to None if we don't want to do ablation
	if ablation_level:
		if kernel_experiment:
			cache = shelve.open(f"{DIRECTORY}/experiments/structural_distance/structural_distance_music/cache_files/cache_postprocess_WL_KERNEL_ablation{ablation_level}.shelve")
		else:
			cache = shelve.open(f"{DIRECTORY}/experiments/structural_distance/structural_distance_music/cache_files/cache_postprocess_ablation{ablation_level}.shelve")
	else:
		if kernel_experiment:
			cache = shelve.open(f"{DIRECTORY}/experiments/structural_distance/structural_distance_music/cache_files/cache_postprocess_WL_KERNEL.shelve")
		else:
			cache = shelve.open(f"{DIRECTORY}/experiments/structural_distance/structural_distance_music/cache_files/cache_postprocess.shelve")
	dist_matrics = [create_distance_matrix(reorder_cluster_to_reference(cluster), cache, kernel_experiment=kernel_experiment, ablation_level=ablation_level) for cluster in clusters]
	cache.close()

	return np.mean(np.stack(dist_matrics), axis=0)

if __name__ == "__main__":
	# renamed from f"{DIRECTORY}/experiments/structural_distance/structural_distance_music/cache_files/input_clusters.pkl" 
	clusters_path = f"{DIRECTORY}/experiments/structural_distance/structural_distance_music/input_clusters.pkl" 
	print(run(clusters_path))

	