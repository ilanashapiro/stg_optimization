import os, sys, pickle, json, re
import numpy as np
import shelve
import torch 
from multiprocessing import Pool, current_process
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import seaborn as sns

# DIRECTORY = "/home/ubuntu/project"
DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
TIME_PARAM = '50s'
NUM_GPUS = 8 
sys.path.append(DIRECTORY)
sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/structural_distance/structural_distance_experiment")

import build_graph
import structural_distance_gen_clusters as st_gen_clusters
import simanneal_centroid_run, simanneal_centroid_helpers, simanneal_centroid

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

def clustering_experiment_NOTUSING():
	features = np.array([compute_spectra(A) for A in listA_G])
	scaler = StandardScaler()
	features_scaled = scaler.fit_transform(features)
	centroid_features = compute_spectra(A_g)
	centroid_features_scaled = scaler.transform([centroid_features])

	num_clusters = 1 
	n_neighbors = min(5, len(features_scaled))  # Ensure n_neighbors <= n_samples
	spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=n_neighbors, random_state=0)
	labels = spectral.fit_predict(features_scaled)
	cluster_centroid = np.mean(features_scaled, axis=0)

	plot_feature_space(features_scaled, centroid_features_scaled, cluster_centroid, labels)
	
	features = np.array([compute_spectra(A) for A in listA_G])

	# Standardize the feature vectors
	scaler = StandardScaler()
	features_scaled = scaler.fit_transform(features)

	# Perform spectral clustering
	num_clusters = 1 
	n_neighbors = min(5, len(features_scaled))  # Ensure n_neighbors <= n_samples
	spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=n_neighbors, random_state=0)
	labels = spectral.fit_predict(features_scaled)

	centroid_features = compute_spectra(A_g)
	centroid_features_scaled = scaler.transform([centroid_features])

	pca = PCA(n_components=2)  # Reduce to 2 components for visualization
	features_pca = pca.fit_transform(features_scaled)

	# Get the PCA components (each row corresponds to a principal component)
	components = pca.components_

	# Calculate the contribution of each original feature to each principal component
	# The higher the absolute value, the more influence the feature has on that component
	feature_contributions = np.abs(components)

	# Sum contributions across all components to get a measure of overall importance
	salient_feature_scores = feature_contributions.sum(axis=0)

	# Get indices of the most salient features
	num_top_features = 2  # Number of top features to select
	top_feature_indices = np.argsort(salient_feature_scores)[::-1][:num_top_features] # project back to original dimensions

	cluster_centroid = np.mean(features_scaled, axis=0)

	print(top_feature_indices)
	# Plot feature vectors
	print(features_scaled.shape)
	plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=labels, cmap='viridis', marker='o')
	plt.scatter(centroid_features_scaled[0, 0], centroid_features_scaled[0, 1], color='red', marker='x', s=100)# label='Centroid Graph')
	plt.scatter(cluster_centroid[0, 0], cluster_centroid[0, 1], color='blue', marker='^', s=100)
	plt.title('Feature Space Clustering')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.legend()
	plt.show()

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

def spectral_embedding(adj_matrix, n_components=5):
	eigenvalues, eigenvectors = compute_spectra(adj_matrix)  # Get both values and vectors
	# Sort eigenvalues and corresponding eigenvectors
	idx = np.argsort(eigenvalues)
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:, idx]

	# Return the first n_components eigenvectors as the embedding
	return eigenvectors[:, :n_components]

def eval_embedding_similarity(embedding1, embeddings):
    # Calculate cosine similarity between embedding1 and each embedding in embeddings
    similarities = cosine_similarity(embedding1.reshape(1, -1), embeddings)
    return np.mean(similarities)

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

def plot_feature_space(features_scaled, centroid_features_scaled, cluster_centroid, labels):
	# Make sure features are 2D if they are not already
	if features_scaled.shape[1] != 2:
			raise ValueError("Features need to be 2-dimensional for direct plotting.")

	plt.figure(figsize=(12, 6))
	plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=labels, cmap='viridis', marker='o', label='Training Graphs')
	plt.scatter(centroid_features_scaled[0, 0], centroid_features_scaled[0, 1], color='red', marker='x', s=100, label='Centroid Graph')
	
	# Compute cluster centroid in the feature space
	cluster_centroid = np.mean(features_scaled, axis=0)
	plt.scatter(cluster_centroid[0], cluster_centroid[1], color='blue', marker='^', s=100, label='Cluster Centroid')
	
	plt.title('Feature Space Clustering')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.legend()
	plt.grid(True)
	plt.show()

def eval_spectra_correlation(centroid_features, input_features):
	correlations = []
	for features in input_features:
		corr, _ = pearsonr(centroid_features, features)
		correlations.append(corr)
		# print(f"Pearson's correlation for {composer}: {corr}")
	return np.mean(correlations)

def visualize_embeddings(embeddings, labels=None):
	if isinstance(embeddings, list):
			embeddings = np.array(embeddings)

	plt.figure(figsize=(10, 8))
	
	if labels is not None:
		# Ensure that the labels have the same length as the number of embeddings
		if len(labels) != embeddings.shape[0]:
			raise ValueError(f"Length of labels {len(labels)} must match the number of embeddings {embeddings.shape[0]}.")
		
		unique_labels = np.unique(labels)
		palette = sns.color_palette("hsv", len(unique_labels))
		
		# Create a mapping from label to color
		label_color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
		
		# Plot each point with its corresponding color
		for i in range(len(labels)):
			plt.scatter(embeddings[i, 0], embeddings[i, 1], 
									color=label_color_map[labels[i]], s=100, alpha=0.6, edgecolors='w')
	else:
		plt.scatter(embeddings[:, 0], embeddings[:, 1], s=100, alpha=0.6, edgecolors='w')

	plt.title('Spectral Embedding Visualization')
	plt.xlabel('First Component')
	plt.ylabel('Second Component')
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
	
		def embedding_similarity():
			for composer, centroid in composer_centroids_dict.items():
				training_pieces = composer_training_pieces_dict[composer]

				# Prepare adjacency matrices and separate the original centroid
				listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([centroid] + training_pieces)
				A_g = listA_G[0]  # the centroid graph
				listA_G = listA_G[1:]  # separate the input graphs from the centroid
				
				# Compute spectral embeddings for the centroid and input graphs
				centroid_embedding = spectral_embedding(A_g)
				input_embeddings = [spectral_embedding(A_G) for A_G in listA_G]
				visualize_embeddings([centroid_embedding] + input_embeddings, labels=["centroid"] + [f"input_piece_{i}" for i in range(0, len(input_embeddings))])
				
				# Print and compute original centroid similarity
				original_similarity = eval_embedding_similarity(centroid_embedding.flatten(), [embedding.flatten() for embedding in input_embeddings])
				
				# Initialize variables to store the best centroid and its similarity score
				best_similarity = original_similarity
				best_centroid_idx = -1  # Track the index of the best centroid (default is -1, i.e., original)
				
				# Loop through each graph in listA_G as a test centroid
				for idx, test_centroid in enumerate(listA_G):
						test_centroid_embedding = spectral_embedding(test_centroid)
						
						# Exclude the test centroid from input_embeddings for this test
						other_embeddings = [spectral_embedding(A_G) for i, A_G in enumerate(listA_G) if i != idx]
						
						# Calculate mean similarity for this test centroid using cosine similarity
						mean_similarity = eval_embedding_similarity(test_centroid_embedding.flatten(), [embedding.flatten() for embedding in other_embeddings])
						
						print(f"Mean cosine similarity for test centroid at index {idx}: {mean_similarity}")
						
						# Update the best centroid if this test centroid has a higher similarity
						if mean_similarity > best_similarity:
								best_similarity = mean_similarity
								best_centroid_idx = idx  # Store the index of the best test centroid

				# Print the best centroid's similarity results
				if best_centroid_idx == -1:
						print("The original centroid is the closest to the corpus!")
				else:
						print(f"The graph at index {best_centroid_idx} has the highest overall similarity with an average of {best_similarity}.")
	
	def laplacian_correlation():
		for composer, centroid in composer_centroids_dict.items():
			training_pieces = composer_training_pieces_dict[composer]
			
			# Prepare adjacency matrices and separate the original centroid
			listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([centroid] + training_pieces)
			A_g = listA_G[0]  # the centroid graph
			listA_G = listA_G[1:]  # separate the input graphs from the centroid
			
			# Calculate spectra for the original centroid and input graphs
			centroid_features = compute_laplacian(A_g)[0] # replace with compute_spectra to examine spectral decomposition instead
			input_features = [compute_laplacian(A_G)[0] for A_G in listA_G] # replace with compute_spectra to examine spectral decomposition instead
			
			# Print and plot original centroid correlation
			original_correlation = eval_spectra_correlation(centroid_features, input_features)
			print(f"Mean Pearson correlation for original {composer} centroid: {original_correlation}")
			plot_spectra(composer, centroid_features, input_features, title=f'Laplacian Decomposition for Composer {composer.capitalize()}')
			
			# Initialize variables to store the best centroid and its correlation score
			best_correlation = original_correlation
			best_centroid_idx = -1  # Track the index of the best centroid (default is -1, i.e., original)
			
			# Loop through each graph in listA_G as a test centroid
			for idx, test_centroid in enumerate(listA_G):
					test_centroid_features = compute_laplacian(test_centroid)[0] # replace with compute_spectra to examine spectral decomposition instead
					
					# Exclude the test centroid from input_features for this test
					other_features = [compute_laplacian(A_G)[0] for i, A_G in enumerate(listA_G) if i != idx] # replace with compute_spectra to examine spectral decomposition instead
					
					# Calculate mean correlation for this test centroid
					mean_correlation = eval_spectra_correlation(test_centroid_features, other_features)
					
					print(f"Mean Pearson correlation for test centroid at index {idx}: {mean_correlation}")
					
					# Update the best centroid if this test centroid has a higher correlation
					if mean_correlation > best_correlation:
							best_correlation = mean_correlation
							best_centroid_idx = idx  # Store the index of the best test centroid

			# Print and plot the best centroid's correlation results
			if best_centroid_idx == -1:
					print("The original centroid is the closest to the corpus!")
			else:
					print(f"The graph at index {best_centroid_idx} has the highest overall correlation with an average of {best_correlation}.")
						
	def spectra_correlation():
		for composer, centroid in composer_centroids_dict.items():
			training_pieces = composer_training_pieces_dict[composer]
			listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([centroid] + training_pieces)
			A_g = listA_G[0] # the centroid graph
			listA_G = listA_G[1:] # separate the input graphs from the centroid

			centroid_features = compute_spectra(A_g)[0]
			input_features = [compute_spectra(A_G)[0] for A_G in listA_G]

			print(f"Mean Pearson correlation for {composer}: {eval_spectra_correlation(centroid_features, input_features)}")
			plot_spectra(composer, centroid_features, input_features, title=f'Spectral Decomposition for Composer {composer.capitalize()}')

	laplacian_correlation()
	# embedding_similarity()
	# spectra_correlation()