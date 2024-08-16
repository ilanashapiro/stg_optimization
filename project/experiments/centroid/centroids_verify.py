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

# DIRECTORY = "/home/ubuntu/project"
DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
TIME_PARAM = '50s'
NUM_GPUS = 8 
sys.path.append(DIRECTORY)
sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/structural_dist/struct_dist_data")

import build_graph
import centroid_gen_clusters 
import structural_distance_gen_clusters as st_gen_clusters
import simanneal_centroid_run, simanneal_centroid_helpers, simanneal_centroid
import centroids_process

def compute_spectral_features(adj_matrix):
	G = nx.from_numpy_array(adj_matrix)
	laplacian = nx.laplacian_matrix(G).toarray()
	eigenvalues = np.linalg.eigvalsh(laplacian)
	return eigenvalues


def plot_spectral_features(composer, centroid_features, training_features, title='Spectral Features Comparison'):
		plt.figure(figsize=(8, 6))
		plt.plot(centroid_features, label='Centroid STG', marker='o')
		for i, features in enumerate(training_features):
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

		# Scatter plot of the features
		plt.figure(figsize=(12, 6))
		plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=labels, cmap='viridis', marker='o', label='Training Graphs')
		
		# Scatter plot of the centroid features
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

if __name__ == "__main__":
	centroid_path = f"{DIRECTORY}/experiments/centroid/final_centroid_{TIME_PARAM}"
	training_pieces_path = f"{DIRECTORY}/experiments/centroid/clusters/composer_centroid_input_graphs_{TIME_PARAM}.txt"
	
	composer_centroids_dict = centroids_process.load_centroids(centroid_path) 
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
		A_g = listA_G[0] # the graph we want to test for classifying
		listA_G = listA_G[1:] # separate the input graphs from the centroid

		centroid_features = compute_spectral_features(A_g)
		training_features = [compute_spectral_features(adj_matrix) for adj_matrix in listA_G]

		correlations = []
		for features in training_features:
				corr, _ = pearsonr(centroid_features, features)
				correlations.append(corr)

		# Print out the correlations
		print(f"Pearson correlations for {composer}: {correlations}")
		
		# Optional: Calculate the mean correlation for a summary statistic
		mean_correlation = np.mean(correlations)
		print(f"Mean Pearson correlation for {composer}: {mean_correlation}")
		continue
		# Plot spectral features for comparison
		# plot_spectral_features(composer, centroid_features, training_features, title=f'Spectral Features Comparison for Composer {composer}')
		# continue
		features = np.array([compute_spectral_features(A) for A in listA_G])
		scaler = StandardScaler()
		features_scaled = scaler.fit_transform(features)
		centroid_features = compute_spectral_features(A_g)
		centroid_features_scaled = scaler.transform([centroid_features])

		num_clusters = 1 
		n_neighbors = min(5, len(features_scaled))  # Ensure n_neighbors <= n_samples
		spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=n_neighbors, random_state=0)
		labels = spectral.fit_predict(features_scaled)
		# Compute cluster centroid
		cluster_centroid = np.mean(features_scaled, axis=0)

		# Plot feature vectors
		plot_feature_space(features_scaled, centroid_features_scaled, cluster_centroid, labels)
		
		continue
		features = np.array([compute_spectral_features(A) for A in listA_G])

		# Standardize the feature vectors
		scaler = StandardScaler()
		features_scaled = scaler.fit_transform(features)

		# Perform spectral clustering
		num_clusters = 1 
		n_neighbors = min(5, len(features_scaled))  # Ensure n_neighbors <= n_samples
		spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=n_neighbors, random_state=0)
		labels = spectral.fit_predict(features_scaled)

		# Compute features for the centroid graph
		centroid_features = compute_spectral_features(A_g)
		centroid_features_scaled = scaler.transform([centroid_features])

		pca = PCA(n_components=2)  # Reduce to 2 components for visualization, adjust as needed
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
		top_feature_indices = np.argsort(salient_feature_scores)[::-1][:num_top_features]


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

		
