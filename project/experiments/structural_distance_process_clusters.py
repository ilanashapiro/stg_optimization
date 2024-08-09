import os, sys
import pickle, shelve
import structural_distance_gen_clusters as gen_clusters
import numpy as np

DIRECTORY = '/home/ubuntu/project'

def create_distance_matrix(cluster, cache):
	n = len(cluster)
	distance_matrix = distance_matrix = np.zeros((n, n))
	
	for i in range(n):
		for j in range(i, n):
			if i == j:
				distance_matrix[i][j] = 0
			else:
				graph_fp1, graph_fp2 = cluster[i], cluster[j]
				cache_key = repr((graph_fp1, graph_fp2))
				cache_key_rev = repr((graph_fp2, graph_fp1))
				if cache_key in cache:
					d = cache[cache_key]
				elif cache_key_rev in cache:
					d = cache[cache_key_rev]
				else:
					with open(graph_fp1, 'rb') as f:
						G1 = pickle.load(f)
					with open(graph_fp2, 'rb') as f:
						G2 = pickle.load(f)
					d = float(gen_clusters.dist(G1, G2))
					gen_clusters.update_cache(cache, cache_key, d)
				
				distance_matrix[i][j] = d
				distance_matrix[j][i] = d  # Ensure symmetry
	
	return distance_matrix

def reorder_cluster_to_reference(cluster):
	ref_order = ['bach', 'mozart', 'beethoven', 'schubert', 'brahms', 'haydn', 'chopin']
	composers_from_paths = [gen_clusters.get_composer_from_path(path) for path in cluster]
	order_map = {substring: index for index, substring in enumerate(ref_order)}
	return tuple(sorted(cluster, key=lambda path: order_map[gen_clusters.get_composer_from_path(path)]))

if __name__ == "__main__":
	clusters_path = f"{DIRECTORY}/experiments/clusters3.pkl"
	clusters = gen_clusters.load_saved_combinations(clusters_path)
	
	cache = shelve.open("cache.shelve")
	dist_matrics = [create_distance_matrix(reorder_cluster_to_reference(cluster), cache) for cluster in clusters]
	cache.close()

	mean_dist_matrix = np.mean(np.stack(dist_matrics), axis=0)
	print(mean_dist_matrix)
	
	

	