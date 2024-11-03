import os, sys, pickle, json, re
import numpy as np
import shelve
import torch, math
from multiprocessing import Pool, current_process
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
import seaborn as sns
from scipy.spatial.distance import euclidean, cosine, cityblock, minkowski, mahalanobis
from sklearn.manifold import SpectralEmbedding
import defining_identifying_optimal_dimension 

# DIRECTORY = "/home/ubuntu/project"
DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
# DIRECTORY = "/home/ilshapiro/project"
TIME_PARAM = '50s'

sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/centroid/defining_identifying_optimal_dimension")
import simanneal_centroid_helpers, embedding_dist

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

def compute_spectra(adj_matrix):
	eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
	return eigenvalues, eigenvectors 

def compute_laplacian(adj_matrix):
	degrees = np.sum(adj_matrix, axis=1)  # Sum rows to get degrees
	D = np.diag(degrees)  # Create the degree matrix
	
	# Compute the Laplacian matrix
	laplacian_matrix = D - adj_matrix
	
	# Compute the eigenvalues and eigenvectors of the Laplacian matrix
	eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
	return eigenvalues, eigenvectors

def adj_matrix_to_edgelist(adj_matrix):
	edgelist = []
	# Iterate through the adjacency matrix
	for i in range(adj_matrix.shape[0]):
		for j in range(i + 1, adj_matrix.shape[1]):  # Only go through upper triangle if undirected
			if adj_matrix[i, j] != 0:  # Only add edges with non-zero weights
				# Append edge (node1, node2, weight)
				edgelist.append((i, j, adj_matrix[i, j]))
	return edgelist

def spectral_embedding(adj_matrix, n_components=8):
	# num components must be <= matrix dims, per the docs. and if drop_first is true (which it is for Laplacian), the library sets num_components += 1. so we must do max dims - 2 for the max components
	max_components = adj_matrix.shape[0] - 2 
	opt_dims, mse_loss = embedding_dist.cal_embedding_distance(adj_matrix_to_edgelist(adj_matrix))
	embedding = SpectralEmbedding(n_components=n_components)
	return embedding.fit_transform(adj_matrix)

def plot_spectra(composer, centroid_features, input_features, title='Spectra'):
	plt.figure(figsize=(8, 6))
	plt.plot(centroid_features, label='Centroid STG', marker='o')
	for i, features in enumerate(input_features):
			plt.plot(features, label=f"{composer.capitalize()} STG {i+1}", linestyle='--')
	plt.xlabel('Eigenvalue Index')
	plt.ylabel('Eigenvalue')
	plt.title(title)
	plt.legend()
	plt.xlim(left=500)
	plt.grid(True)
	plt.show()

def spectra_inv_correlations(centroid_features, input_features):
	inv_correlations = []
	for features in input_features:
		# corr, _ = pearsonr(centroid_features, features)
		# inv_correlations.append(1 - corr)

		# distance = cityblock(centroid_features, features) # tried euclidean, cosine, cityblock, minkowski -- results weren't good
		distance = minkowski(centroid_features, features, p=2)  # p=2 for Euclidean
		inv_correlations.append(distance)

		# print(f"Pearson's inv_correlations for {composer}: {corr}")
	return inv_correlations

def spectra_correlations(centroid_features, input_features):
	correlations = []
	for features in input_features:
		corr, _ = pearsonr(centroid_features, features)
		correlations.append(corr)
		# print(f"Pearson's correlation for {composer}: {corr}")
	return correlations

def plot_spectral_embeddings(embeddings):
	colors = plt.cm.viridis(np.linspace(0, 1, len(embeddings)))
	plt.figure(figsize=(8, 8))

	for i, embedding in enumerate(embeddings):
		plt.scatter(embedding[:, 0], embedding[:, 1], color=colors[i], label=f'DAG {i + 1}')

	plt.title('Spectral Embedding of DAGs')
	plt.xlabel('Dimension 1')
	plt.ylabel('Dimension 2')
	plt.legend()
	plt.grid()
	plt.show()

if __name__ == "__main__":
	centroid_path = f"{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}"
	training_pieces_path = f"{DIRECTORY}/experiments/centroid/clusters/composer_centroid_input_graphs_{TIME_PARAM}.txt"
	
	composer_centroids_dict = load_centroids() 
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
	
		def embedding_distances():
			for composer, centroid in composer_centroids_dict.items():
				training_pieces = composer_training_pieces_dict[composer]
				listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([centroid] + training_pieces)
				best_score = math.inf
				best_centroid_idx = -1  # Track the index of the best centroid (default is -1, i.e., original)
				
				for idx, test_centroid in enumerate(listA_G):
					test_centroid_embedding = spectral_embedding(test_centroid)
			
					# Exclude the original candidate centroid from input_embeddings for this test
					other_embeddings = [spectral_embedding(A_G) for i, A_G in enumerate(listA_G) if i > 0]

					# plot_spectral_embeddings([test_centroid_embedding] + other_embeddings)
					# sys.exit(0)
					sys.exit(0)
					distances = euclidean_distances(test_centroid_embedding.flatten().reshape(1, -1), [embedding.flatten() for embedding in other_embeddings])
					score = np.mean(distances) * np.std(distances)
					
					# print(f"Score for test centroid at index {idx} (mean * std): {score}")
					
					if score < best_score:
						best_score = score
						best_centroid_idx = idx  # Store the index of the best test centroid

				print(f"The graph at index {best_centroid_idx} has the best score of {best_score}.")
	
	def spectral_correlation():
		for composer, centroid in composer_centroids_dict.items():
				training_pieces = composer_training_pieces_dict[composer]
				
				# Prepare adjacency matrices and separate the original centroid
				listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([centroid] + training_pieces)
				best_score = math.inf # math.inf for inv_correlations, -math.inf for corrlations
				best_centroid_idx = -1
				
				# Loop through each graph in listA_G as a test centroid
				for idx, test_centroid in enumerate(listA_G):
					test_centroid_features = compute_laplacian(test_centroid)[0] # replace with compute_spectra to examine spectral decomposition instead
					other_features = [compute_laplacian(A_G)[0] for j, A_G in enumerate(listA_G) if j > 0] # j > 0 means we're excluding the candidate centroid. also, replace with compute_spectra to examine spectral decomposition instead
					
					# Calculate mean correlation and standard deviation for this test centroid
					inv_correlations = spectra_inv_correlations(test_centroid_features, other_features)
					# correlations = spectra_correlations(test_centroid_features, other_features)
					score = np.mean(inv_correlations) * np.std(inv_correlations)
					# print(f"Test centroid at index {idx}: Mean correlation = {mean_correlation}, Std Dev = {std_dev}, Score = {score}")
					
					# Update the best centroid if this test centroid has a higher score
					if score < best_score: # < for inv_correlations, > for corrlations
						best_score = score
						best_centroid_idx = idx

				print(f"The graph at index {best_centroid_idx} has the best score with a value of {best_score}.")
												
	def plot_spectra_wrapper():
		for composer, centroid in composer_centroids_dict.items():
			training_pieces = composer_training_pieces_dict[composer]
			listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([centroid] + training_pieces)
			A_g = listA_G[0] # the centroid graph
			listA_G = listA_G[1:] # separate the input graphs from the centroid

			centroid_features = compute_spectra(A_g)[0]
			input_features = [compute_spectra(A_G)[0] for A_G in listA_G]

			print(f"Mean Pearson correlation for {composer}: {np.mean(spectra_correlations(centroid_features, input_features))}")
			# plot_spectra(composer, centroid_features, input_features, title=f'Spectral Decomposition for Composer {composer.capitalize()}')

	# spectral_correlation()
	embedding_distances()
	# plot_spectra_wrapper()