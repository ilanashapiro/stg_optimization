import os, sys, pickle, json, re
import numpy as np
import shelve
import torch, math
from multiprocessing import Pool, current_process
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances, pairwise_kernels
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
import seaborn as sns
from scipy.spatial.distance import euclidean, cosine, cityblock, minkowski, mahalanobis
from scipy.optimize import minimize
from sklearn.manifold import SpectralEmbedding
import defining_identifying_optimal_dimension 


# DIRECTORY = "/home/ubuntu/project"
DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
# DIRECTORY = "/home/ilshapiro/project"
TIME_PARAM = '50s'
ABLATION_LEVEL = None

sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/centroid/defining_identifying_optimal_dimension")
import simanneal_centroid_helpers, embedding_dist

def load_centroids():
	centroids_dict = {}
	for composer in ["bach", "beethoven", "haydn", "mozart"]:
		if ABLATION_LEVEL:
			final_centroid_dir = f"{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}_ablation{ABLATION_LEVEL}/{composer}"
		else:
			final_centroid_dir = f"{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}/{composer}"

		final_centroid_path = os.path.join(final_centroid_dir, "final_centroid.txt")
		final_centroid = np.loadtxt(final_centroid_path)
		# print(f'Loaded: {final_centroid_path}')

		final_centroid_idx_node_mapping_path = os.path.join(final_centroid_dir, "final_idx_node_mapping.txt")
		with open(final_centroid_idx_node_mapping_path, 'r') as file:
			idx_node_mapping = json.load(file)
			idx_node_mapping = {int(k): v for k, v in idx_node_mapping.items()}
		# print(f'Loaded: {final_centroid_idx_node_mapping_path}')

		# we just use the entire node metadata dict from approx_centroid_dir
		if ABLATION_LEVEL:
			approx_centroid_dir = f"{DIRECTORY}/experiments/centroid/approx_centroids/approx_centroid_{TIME_PARAM}_ablation{ABLATION_LEVEL}/{composer}"
		else:
			approx_centroid_dir = f"{DIRECTORY}/experiments/centroid/approx_centroids/approx_centroid_{TIME_PARAM}/{composer}"

		approx_centroid_node_metadata_dict_path = os.path.join(approx_centroid_dir, "node_metadata_dict.txt")
		with open(approx_centroid_node_metadata_dict_path, 'r') as file:
			node_metadata_dict = json.load(file)
		# print(f'Loaded: {approx_centroid_node_metadata_dict_path}')
		
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

def get_opt_embedding_dim(adj_matrix):
	opt_dim, mse_loss = embedding_dist.cal_embedding_distance(adj_matrix_to_edgelist(adj_matrix))
	print(opt_dim, mse_loss)
	return opt_dim
	
def spectral_embedding(adj_matrix, n_components=8):
	# num components must be <= matrix dims, per the docs. and if drop_first is true (which it is for Laplacian), the library sets num_components += 1. so we must do max dims - 2 for the max components
	max_components = adj_matrix.shape[0] - 2 
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

def plot_embeddings_for_composer(composer, embeddings_dict):
		# Get the embeddings for the selected composer
		embeddings = embeddings_dict[composer]

		candidate_centroid = embeddings[0]
		embeddings = embeddings[1:]
		
		# Apply PCA to reduce dimensionality to 2D
		pca = PCA(n_components=2, svd_solver='full')
		reduced_embeddings = pca.fit_transform(np.array([embedding.flatten() for embedding in embeddings]))
		
		# Compute the initial guess (mean embedding) for this composer
		initial_guess = np.mean(embeddings, axis=0)
		
		# Convert the initial guess to 2D using PCA
		initial_guess_2d = pca.transform(np.array([initial_guess.flatten()]))  # Single point, reshape to 2D
		candidate_centroid_2d = pca.transform(np.array([candidate_centroid.flatten()]))

		# Plot the embeddings
		plt.figure(figsize=(10, 8))
		
		# Plot the embeddings for the pieces
		plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', label=f'{composer} Pieces')

		# Plot the initial guess
		plt.scatter(initial_guess_2d[:, 0], initial_guess_2d[:, 1], c='red', marker='x', label='Initial Guess')
		plt.scatter(candidate_centroid_2d[:, 0], candidate_centroid_2d[:, 1], c='green', marker='*', label='Candidate Centroid')

		# Adding labels for each piece
		for i, label in enumerate([f"Piece {i+1}" for i in range(len(embeddings))]):
				plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label, fontsize=8, ha='right', va='bottom')
		
		
		# Add title and labels
		plt.title(f'2D Plot of Embeddings and Initial Guess for Composer: {composer}')
		plt.xlabel('PCA Component 1')
		plt.ylabel('PCA Component 2')
		plt.legend(loc='upper right')
		plt.savefig(f'output_plot_{composer}.png')
		plt.close()


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
			composer_idx = 0
			for composer, centroid in composer_centroids_dict.items():
				training_pieces = composer_training_pieces_dict[composer]
				listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([centroid] + training_pieces)
				best_score = math.inf
				best_centroid_idx = -1  # Track the index of the best centroid (default is -1, i.e., original)
				
				# the embeddings must be the same size, so we do the max of the opt embeddings so as to not lose structural info
				# embedding_dim = int(np.median([get_opt_embedding_dim(A_G) for A_G in listA_G]))
				# print(f"EMBEDDING DIM (MEDIAN) FOR {composer}: {embedding_dim}")
				embedding_dim = [51, 26, 70, 48][composer_idx] # no ablation
				# embedding_dim = [53, 21, 77, 55][composer_idx] # ablation 4
				# embedding_dim = [57, 49, 56, 24][composer_idx] # ablation 3
				# embedding_dim = [57, 49, 56, 24][composer_idx] # ablation 2
				# embedding_dim = [56, 23, 61, 57][composer_idx] # ablation 1
				
				for idx, test_centroid in enumerate(listA_G):
					test_centroid_embedding = spectral_embedding(test_centroid, n_components=embedding_dim)
			
					# Exclude the original candidate centroid from input_embeddings for this test
					other_embeddings = [spectral_embedding(A_G, n_components=embedding_dim) for i, A_G in enumerate(listA_G) if i > 0]

					# plot_spectral_embeddings([test_centroid_embedding] + other_embeddings)

					# distances = np.array([1 - pairwise_kernels([test_centroid_embedding.flatten()], [embedding.flatten() for embedding in other_embeddings], metric='rbf')])
					distances = cdist([test_centroid_embedding.flatten()], [embedding.flatten() for embedding in other_embeddings], metric='euclidean')
					score = np.mean(distances) * np.std(distances)
					# score = np.mean([float(pearsonr(test_centroid_embedding.flatten(), other_embedding.flatten())[0]) for other_embedding in other_embeddings])
					
					print(f"Score for test centroid at index {idx} (mean * std): {score} with ablation {ABLATION_LEVEL}")
					# sys.exit(0)
					
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

	def spectral_centroid():
		composer_idx = 0
		embeddings_dict = {}
		for composer, centroid in composer_centroids_dict.items():
			training_pieces = composer_training_pieces_dict[composer]
			listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([centroid] + training_pieces)
			embeddings_dict[composer] = []

			# the embeddings must be the same size, so we do the max of the opt embeddings so as to not lose structural info
			# embedding_dim = int(np.median([get_opt_embedding_dim(A_G) for A_G in listA_G]))
			# print(f"EMBEDDING DIM (MEDIAN) FOR {composer}: {embedding_dim}")
			embedding_dim = [51, 26, 70, 48][composer_idx] # no ablation
			
			def objective_function(corpus_embeddings):
				dists = cdist([corpus_embeddings[0].flatten()], [embedding.flatten() for embedding in corpus_embeddings[1:]], metric='euclidean')
				return np.mean(dists) * np.std(dists)

			for A_G in listA_G:
				embedding = spectral_embedding(A_G, n_components=embedding_dim)
				embeddings_dict[composer].append(embedding)
			
			plot_embeddings_for_composer(composer, embeddings_dict)

			# sys.exit(0)
			# initial_guess = np.mean(embeddings_dict[composer], axis=0)
			# print("MINIMIZING")
			# result = minimize(objective_function, initial_guess, method='L-BFGS-B', options={'disp': True})

			# optimized_centroid = result.x

			# print("Optimized Centroid Embedding:", optimized_centroid)
			# print("Optimized Function Value:", result.fun)

				
				


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
	# embedding_distances()
	# plot_spectra_wrapper()
	spectral_centroid()

#--------------------------------------------------------------------------------------------------------------------------------------------
# EUCLIDEAN DISTANCE RESULTS (full STG, no ablation):

# graph size smaller than the default end dimension, thus has been automatically set to 92
# the optimal dimension at 0.05 accuracy level is 32
# the MSE of curve fitting is 2.710455981557946e-05
# 32 2.710455981557946e-05
# graph size smaller than the default end dimension, thus has been automatically set to 111
# the optimal dimension at 0.05 accuracy level is 47
# the MSE of curve fitting is 0.00010975787397100909
# 47 0.00010975787397100909
# graph size smaller than the default end dimension, thus has been automatically set to 140
# the optimal dimension at 0.05 accuracy level is 34
# the MSE of curve fitting is 4.426486827246422e-05
# 34 4.426486827246422e-05
# graph size smaller than the default end dimension, thus has been automatically set to 179
# the optimal dimension at 0.05 accuracy level is 69
# the MSE of curve fitting is 0.0001151441312731053
# 69 0.0001151441312731053
# graph size smaller than the default end dimension, thus has been automatically set to 134
# the optimal dimension at 0.05 accuracy level is 49
# the MSE of curve fitting is 7.562297260643182e-05
# 49 7.562297260643182e-05
# graph size smaller than the default end dimension, thus has been automatically set to 346
# the optimal dimension at 0.05 accuracy level is 143
# the MSE of curve fitting is 4.136423205525769e-05
# 143 4.136423205525769e-05
# graph size smaller than the default end dimension, thus has been automatically set to 173
# the optimal dimension at 0.05 accuracy level is 65
# the MSE of curve fitting is 8.723696916253633e-05
# 65 8.723696916253633e-05
# EMBEDDING DIM (MEDIAN) FOR bach: 49
# Score for test centroid at index 0 (mean * std): 0.008574461422436126
# Score for test centroid at index 1 (mean * std): 0.4565080427122601
# Score for test centroid at index 2 (mean * std): 0.506996934425738
# Score for test centroid at index 3 (mean * std): 0.40286174862827867
# Score for test centroid at index 4 (mean * std): 0.5050712964613594
# Score for test centroid at index 5 (mean * std): 0.615436108264228
# Score for test centroid at index 6 (mean * std): 0.6136893531835792
# The graph at index 0 has the best score of 0.008574461422436126.

# graph size smaller than the default end dimension, thus has been automatically set to 127
# the optimal dimension at 0.05 accuracy level is 69
# the MSE of curve fitting is 7.945919151826464e-05
# 69 7.945919151826464e-05
# graph size smaller than the default end dimension, thus has been automatically set to 200
# the optimal dimension at 0.05 accuracy level is 110
# the MSE of curve fitting is 5.640084314862057e-05
# 110 5.640084314862057e-05
# graph size smaller than the default end dimension, thus has been automatically set to 40
# the optimal dimension at 0.05 accuracy level is 13
# the MSE of curve fitting is 6.993271166166618e-05
# 13 6.993271166166618e-05
# graph size smaller than the default end dimension, thus has been automatically set to 48
# the optimal dimension at 0.05 accuracy level is 11
# the MSE of curve fitting is 4.79958710763192e-05
# 11 4.79958710763192e-05
# graph size smaller than the default end dimension, thus has been automatically set to 73
# the optimal dimension at 0.05 accuracy level is 29
# the MSE of curve fitting is 3.45823259164068e-05
# 29 3.45823259164068e-05
# graph size smaller than the default end dimension, thus has been automatically set to 55
# the optimal dimension at 0.05 accuracy level is 15
# the MSE of curve fitting is 3.5061761940651e-05
# 15 3.5061761940651e-05
# graph size smaller than the default end dimension, thus has been automatically set to 34
# the optimal dimension at 0.05 accuracy level is 10
# the MSE of curve fitting is 9.515193416303498e-05
# 10 9.515193416303498e-05
# graph size smaller than the default end dimension, thus has been automatically set to 63
# the optimal dimension at 0.05 accuracy level is 17
# the MSE of curve fitting is 2.8707516553239562e-05
# 17 2.8707516553239562e-05
# graph size smaller than the default end dimension, thus has been automatically set to 104
# the optimal dimension at 0.05 accuracy level is 31
# the MSE of curve fitting is 5.934929548428081e-05
# 31 5.934929548428081e-05
# graph size smaller than the default end dimension, thus has been automatically set to 147
# the optimal dimension at 0.05 accuracy level is 64
# the MSE of curve fitting is 5.9713638117520164e-05
# 64 5.9713638117520164e-05
# EMBEDDING DIM (MEDIAN) FOR beethoven: 23
# Score for test centroid at index 0 (mean * std): 0.015035301886421111
# Score for test centroid at index 1 (mean * std): 0.3745473824719996
# Score for test centroid at index 2 (mean * std): 0.2738294381806822
# Score for test centroid at index 3 (mean * std): 0.2698867737494362
# Score for test centroid at index 4 (mean * std): 0.37822992759334256
# Score for test centroid at index 5 (mean * std): 0.27670954219527955
# Score for test centroid at index 6 (mean * std): 0.2635396609059994
# Score for test centroid at index 7 (mean * std): 0.2727027809700001
# Score for test centroid at index 8 (mean * std): 0.3796284282736056
# Score for test centroid at index 9 (mean * std): 0.378488173943061
	
# The graph at index 0 has the best score of 0.015035301886421111.
# graph size smaller than the default end dimension, thus has been automatically set to 115
# the optimal dimension at 0.05 accuracy level is 56
# the MSE of curve fitting is 7.038564861744082e-05
# 56 7.038564861744082e-05
# graph size smaller than the default end dimension, thus has been automatically set to 102
# the optimal dimension at 0.05 accuracy level is 29
# the MSE of curve fitting is 7.772750260992701e-05
# 29 7.772750260992701e-05
# graph size smaller than the default end dimension, thus has been automatically set to 163
# the optimal dimension at 0.05 accuracy level is 95
# the MSE of curve fitting is 5.146907757390221e-05
# 95 5.146907757390221e-05
# graph size smaller than the default end dimension, thus has been automatically set to 182
# the optimal dimension at 0.05 accuracy level is 103
# the MSE of curve fitting is 8.126296724510065e-05
# 103 8.126296724510065e-05
# graph size smaller than the default end dimension, thus has been automatically set to 158
# the optimal dimension at 0.05 accuracy level is 68
# the MSE of curve fitting is 6.993003041435008e-05
# 68 6.993003041435008e-05
# EMBEDDING DIM (MEDIAN) FOR haydn: 68
# Score for test centroid at index 0 (mean * std): 0.041445182887735625
# Score for test centroid at index 1 (mean * std): 0.8285262918412071
# Score for test centroid at index 2 (mean * std): 1.052813759695979
# Score for test centroid at index 3 (mean * std): 1.222437786337313
# Score for test centroid at index 4 (mean * std): 1.4615183438743837
# The graph at index 0 has the best score of 0.041445182887735625.

# graph size smaller than the default end dimension, thus has been automatically set to 113
# the optimal dimension at 0.05 accuracy level is 50
# the MSE of curve fitting is 7.260695629526462e-05
# 50 7.260695629526462e-05
# graph size smaller than the default end dimension, thus has been automatically set to 177
# the optimal dimension at 0.05 accuracy level is 72
# the MSE of curve fitting is 4.7238794232087796e-05
# 72 4.7238794232087796e-05
# graph size smaller than the default end dimension, thus has been automatically set to 133
# the optimal dimension at 0.05 accuracy level is 42
# the MSE of curve fitting is 4.145676008638775e-05
# 42 4.145676008638775e-05
# graph size smaller than the default end dimension, thus has been automatically set to 98
# the optimal dimension at 0.05 accuracy level is 25
# the MSE of curve fitting is 2.569817339641731e-05
# 25 2.569817339641731e-05
# graph size smaller than the default end dimension, thus has been automatically set to 137
# the optimal dimension at 0.05 accuracy level is 61
# the MSE of curve fitting is 5.9920605734343336e-05
# 61 5.9920605734343336e-05
# graph size smaller than the default end dimension, thus has been automatically set to 150
# the optimal dimension at 0.05 accuracy level is 54
# the MSE of curve fitting is 5.575161178008929e-05
# 54 5.575161178008929e-05
# EMBEDDING DIM (MEDIAN) FOR mozart: 52
# Score for test centroid at index 0 (mean * std): 0.014712376513727393
# Score for test centroid at index 1 (mean * std): 1.0282015684591146
# Score for test centroid at index 2 (mean * std): 1.0204491714930022
# Score for test centroid at index 3 (mean * std): 0.8980017616418008
# Score for test centroid at index 4 (mean * std): 0.8104844781196467
# Score for test centroid at index 5 (mean * std): 0.8647802490312058
# The graph at index 0 has the best score of 0.014712376513727393.

#--------------------------------------------------------------------------------------------------------------------------------------------
# ABLATION 1 LEVELS RESULTS

# graph size smaller than the default end dimension, thus has been automatically set to 9
# the optimal dimension at 0.05 accuracy level is 489
# the MSE of curve fitting is 1.5407439555097887e-33
# 489 1.5407439555097887e-33
# graph size smaller than the default end dimension, thus has been automatically set to 111
# the optimal dimension at 0.05 accuracy level is 36
# the MSE of curve fitting is 5.2511883701769414e-05
# 36 5.2511883701769414e-05
# graph size smaller than the default end dimension, thus has been automatically set to 140
# the optimal dimension at 0.05 accuracy level is 36
# the MSE of curve fitting is 6.643736924589481e-05
# 36 6.643736924589481e-05
# graph size smaller than the default end dimension, thus has been automatically set to 179
# the optimal dimension at 0.05 accuracy level is 56
# the MSE of curve fitting is 9.77714946677505e-05
# 56 9.77714946677505e-05
# graph size smaller than the default end dimension, thus has been automatically set to 134
# the optimal dimension at 0.05 accuracy level is 42
# the MSE of curve fitting is 5.181357175901629e-05
# 42 5.181357175901629e-05
# graph size smaller than the default end dimension, thus has been automatically set to 346
# the optimal dimension at 0.05 accuracy level is 153
# the MSE of curve fitting is 3.69343615732918e-05
# 153 3.69343615732918e-05
# graph size smaller than the default end dimension, thus has been automatically set to 173
# the optimal dimension at 0.05 accuracy level is 72
# the MSE of curve fitting is 0.00010311214520682333
# 72 0.00010311214520682333
# EMBEDDING DIM (MEDIAN) FOR bach: 56
# Score for test centroid at index 0 (mean * std): 0.008727339611986676
# Score for test centroid at index 1 (mean * std): 0.5080489292113205
# Score for test centroid at index 2 (mean * std): 0.42393905887287153
# Score for test centroid at index 3 (mean * std): 0.515711002762139
# Score for test centroid at index 4 (mean * std): 0.5877421058479286
# Score for test centroid at index 5 (mean * std): 0.7147425364922209
# Score for test centroid at index 6 (mean * std): 0.7065490699758724
# The graph at index 0 has the best score of 0.008727339611986676.

# graph size smaller than the default end dimension, thus has been automatically set to 7
# the optimal dimension at 0.05 accuracy level is 142
# the MSE of curve fitting is 7.703719777548943e-33
# 142 7.703719777548943e-33s
# graph size smaller than the default end dimension, thus has been automatically set to 200
# the optimal dimension at 0.05 accuracy level is 99
# the MSE of curve fitting is 5.594589023181938e-05
# 99 5.594589023181938e-05
# graph size smaller than the default end dimension, thus has been automatically set to 40
# the optimal dimension at 0.05 accuracy level is 18
# the MSE of curve fitting is 0.00011077955163300875
# 18 0.00011077955163300875
# graph size smaller than the default end dimension, thus has been automatically set to 48
# the optimal dimension at 0.05 accuracy level is 14
# the MSE of curve fitting is 4.706860690806917e-05
# 14 4.706860690806917e-05
# graph size smaller than the default end dimension, thus has been automatically set to 73
# the optimal dimension at 0.05 accuracy level is 25
# the MSE of curve fitting is 9.626607766339998e-05
# 25 9.626607766339998e-05
# graph size smaller than the default end dimension, thus has been automatically set to 55
# the optimal dimension at 0.05 accuracy level is 16
# the MSE of curve fitting is 4.4840113822865186e-05
# 16 4.4840113822865186e-05
# graph size smaller than the default end dimension, thus has been automatically set to 34
# the optimal dimension at 0.05 accuracy level is 8
# the MSE of curve fitting is 0.0006190493034994979
# 8 0.0006190493034994979
# graph size smaller than the default end dimension, thus has been automatically set to 63
# the optimal dimension at 0.05 accuracy level is 21
# the MSE of curve fitting is 5.8370479410150465e-05
# 21 5.8370479410150465e-05
# graph size smaller than the default end dimension, thus has been automatically set to 104
# the optimal dimension at 0.05 accuracy level is 30
# the MSE of curve fitting is 6.452674675127743e-05
# 30 6.452674675127743e-05
# graph size smaller than the default end dimension, thus has been automatically set to 147
# the optimal dimension at 0.05 accuracy level is 62
# the MSE of curve fitting is 5.6538531960260055e-05
# 62 5.6538531960260055e-05
# EMBEDDING DIM (MEDIAN) FOR beethoven: 23
# The graph at index 0 has the best score of 0.039270824163033305.
# Score for test centroid at index 0 (mean * std): 0.008900807163130197
# Score for test centroid at index 1 (mean * std): 0.9323338721394995
# Score for test centroid at index 2 (mean * std): 1.1915983436748676
# Score for test centroid at index 3 (mean * std): 1.2123599719378857
# Score for test centroid at index 4 (mean * std): 1.2335888475663561

# graph size smaller than the default end dimension, thus has been automatically set to 8
# the optimal dimension at 0.05 accuracy level is 1
# the MSE of curve fitting is 1.5407439555097887e-33
# 1 1.5407439555097887e-33
# graph size smaller than the default end dimension, thus has been automatically set to 102
# the optimal dimension at 0.05 accuracy level is 33
# the MSE of curve fitting is 9.519288459784999e-05
# 33 9.519288459784999e-05
# graph size smaller than the default end dimension, thus has been automatically set to 163
# the optimal dimension at 0.05 accuracy level is 77
# the MSE of curve fitting is 5.702469716007259e-05
# 77 5.702469716007259e-05
# graph size smaller than the default end dimension, thus has been automatically set to 182
# the optimal dimension at 0.05 accuracy level is 99
# the MSE of curve fitting is 6.409057107609141e-05
# 99 6.409057107609141e-05
# graph size smaller than the default end dimension, thus has been automatically set to 158
# the optimal dimension at 0.05 accuracy level is 61
# the MSE of curve fitting is 8.020003101210238e-05
# 61 8.020003101210238e-05
# EMBEDDING DIM (MEDIAN) FOR haydn: 61
# Score for test centroid at index 0 (mean * std): 693.9404019998244
# Score for test centroid at index 1 (mean * std): 9286.441425857802
# Score for test centroid at index 2 (mean * std): 10392.8326644266
# Score for test centroid at index 3 (mean * std): 11454.85848582048
# Score for test centroid at index 4 (mean * std): 11365.283039676553
# The graph at index 0 has the best score of 693.9404019998244.

# graph size smaller than the default end dimension, thus has been automatically set to 9
# the optimal dimension at 0.05 accuracy level is 928
# the MSE of curve fitting is 7.703719777548943e-33
# 928 7.703719777548943e-33
# graph size smaller than the default end dimension, thus has been automatically set to 177
# the optimal dimension at 0.05 accuracy level is 82
# the MSE of curve fitting is 9.17216590642881e-05
# 82 9.17216590642881e-05
# graph size smaller than the default end dimension, thus has been automatically set to 133
# the optimal dimension at 0.05 accuracy level is 44
# the MSE of curve fitting is 4.0275951597613984e-05
# 44 4.0275951597613984e-05
# graph size smaller than the default end dimension, thus has been automatically set to 98
# the optimal dimension at 0.05 accuracy level is 27
# the MSE of curve fitting is 5.755120367997809e-05
# 27 5.755120367997809e-05
# graph size smaller than the default end dimension, thus has been automatically set to 137
# the optimal dimension at 0.05 accuracy level is 56
# the MSE of curve fitting is 6.254403966062365e-05
# 56 6.254403966062365e-05
# graph size smaller than the default end dimension, thus has been automatically set to 150
# the optimal dimension at 0.05 accuracy level is 59
# the MSE of curve fitting is 9.438046185574712e-05
# 59 9.438046185574712e-05
# EMBEDDING DIM (MEDIAN) FOR mozart: 57
# Score for test centroid at index 0 (mean * std): 0.011036194623772306
# Score for test centroid at index 1 (mean * std): 1.106541692601886
# Score for test centroid at index 2 (mean * std): 1.0947133752166596
# Score for test centroid at index 3 (mean * std): 0.8231013029535288
# Score for test centroid at index 4 (mean * std): 0.91698717196777
# Score for test centroid at index 5 (mean * std): 0.9362294182141782
# The graph at index 0 has the best score of 0.011036194623772306.

#--------------------------------------------------------------------------------------------------------------------------------------------
# ABLATION 2 LEVELS RESULTS

# graph size smaller than the default end dimension, thus has been automatically set to 35
# the optimal dimension at 0.05 accuracy level is 62
# the MSE of curve fitting is 5.206399410464953e-05
# 62 5.206399410464953e-05
# graph size smaller than the default end dimension, thus has been automatically set to 111
# the optimal dimension at 0.05 accuracy level is 48
# the MSE of curve fitting is 7.991682444624433e-05
# 48 7.991682444624433e-05
# graph size smaller than the default end dimension, thus has been automatically set to 140
# the optimal dimension at 0.05 accuracy level is 42
# the MSE of curve fitting is 4.415057330922452e-05
# 42 4.415057330922452e-05
# graph size smaller than the default end dimension, thus has been automatically set to 179
# the optimal dimension at 0.05 accuracy level is 88
# the MSE of curve fitting is 0.00010464397035439035
# 88 0.00010464397035439035
# graph size smaller than the default end dimension, thus has been automatically set to 134
# the optimal dimension at 0.05 accuracy level is 48
# the MSE of curve fitting is 0.00010736291742577732
# 48 0.00010736291742577732
# graph size smaller than the default end dimension, thus has been automatically set to 346
# the optimal dimension at 0.05 accuracy level is 146
# the MSE of curve fitting is 3.8272382342457617e-05
# 146 3.8272382342457617e-05
# graph size smaller than the default end dimension, thus has been automatically set to 173
# the optimal dimension at 0.05 accuracy level is 57
# the MSE of curve fitting is 8.794090830434336e-05
# 57 8.794090830434336e-05
# EMBEDDING DIM (MEDIAN) FOR bach: 57
# Score for test centroid at index 0 (mean * std): 0.008316705882346464
# Score for test centroid at index 1 (mean * std): 0.5411077827396861
# Score for test centroid at index 2 (mean * std): 0.44906762299461184
# Score for test centroid at index 3 (mean * std): 0.5278727361413726
# Score for test centroid at index 4 (mean * std): 0.7145621808206053
# Score for test centroid at index 5 (mean * std): 0.7269980780161366
# Score for test centroid at index 6 (mean * std): 0.7197158733915672
# The graph at index 0 has the best score of 0.008316705882346464.

# graph size smaller than the default end dimension, thus has been automatically set to 47
# the optimal dimension at 0.05 accuracy level is 72
# the MSE of curve fitting is 5.330455137963835e-05
# 72 5.330455137963835e-05
# graph size smaller than the default end dimension, thus has been automatically set to 200
# the optimal dimension at 0.05 accuracy level is 85
# the MSE of curve fitting is 5.1576884995317923e-05
# 85 5.1576884995317923e-05
# graph size smaller than the default end dimension, thus has been automatically set to 40
# the optimal dimension at 0.05 accuracy level is 20
# the MSE of curve fitting is 0.00021259625746559868
# 20 0.00021259625746559868
# graph size smaller than the default end dimension, thus has been automatically set to 48
# the optimal dimension at 0.05 accuracy level is 19
# the MSE of curve fitting is 8.720097686040469e-05
# 19 8.720097686040469e-05
# graph size smaller than the default end dimension, thus has been automatically set to 73
# the optimal dimension at 0.05 accuracy level is 26
# the MSE of curve fitting is 6.27019186616165e-05
# 26 6.27019186616165e-05
# graph size smaller than the default end dimension, thus has been automatically set to 55
# the optimal dimension at 0.05 accuracy level is 13
# the MSE of curve fitting is 2.0974662542841698e-05
# 13 2.0974662542841698e-05
# graph size smaller than the default end dimension, thus has been automatically set to 34
# the optimal dimension at 0.05 accuracy level is 9
# the MSE of curve fitting is 0.00014363117586999968
# 9 0.00014363117586999968
# graph size smaller than the default end dimension, thus has been automatically set to 63
# the optimal dimension at 0.05 accuracy level is 23
# the MSE of curve fitting is 3.9100856611079495e-05
# 23 3.9100856611079495e-05
# graph size smaller than the default end dimension, thus has been automatically set to 104
# the optimal dimension at 0.05 accuracy level is 27
# the MSE of curve fitting is 2.418617463970868e-05
# 27 2.418617463970868e-05
# graph size smaller than the default end dimension, thus has been automatically set to 147
# the optimal dimension at 0.05 accuracy level is 67
# the MSE of curve fitting is 6.0657945680792984e-05
# 67 6.0657945680792984e-05
# EMBEDDING DIM (MEDIAN) FOR beethoven: 24
# Score for test centroid at index 0 (mean * std): 0.005971634490450888
# Score for test centroid at index 1 (mean * std): 0.3992714929289605
# Score for test centroid at index 2 (mean * std): 0.399747202789218
# Score for test centroid at index 3 (mean * std): 0.2901229857071547
# Score for test centroid at index 4 (mean * std): 0.4023120824542109
# Score for test centroid at index 5 (mean * std): 0.24266516079181788
# Score for test centroid at index 6 (mean * std): 0.25436613565457594
# Score for test centroid at index 7 (mean * std): 0.29184798978870313
# Score for test centroid at index 8 (mean * std): 0.40365836380712106
# Score for test centroid at index 9 (mean * std): 0.40109647322928854
# The graph at index 0 has the best score of 0.005971634490450888.

# graph size smaller than the default end dimension, thus has been automatically set to 36
# the optimal dimension at 0.05 accuracy level is 34
# the MSE of curve fitting is 7.364579400519923e-05
# 34 7.364579400519923e-05
# graph size smaller than the default end dimension, thus has been automatically set to 102
# the optimal dimension at 0.05 accuracy level is 26
# the MSE of curve fitting is 6.665146540179774e-05
# 26 6.665146540179774e-05
# graph size smaller than the default end dimension, thus has been automatically set to 163
# the optimal dimension at 0.05 accuracy level is 83
# the MSE of curve fitting is 5.4253221893465133e-05
# 83 5.4253221893465133e-05
# graph size smaller than the default end dimension, thus has been automatically set to 182
# the optimal dimension at 0.05 accuracy level is 114
# the MSE of curve fitting is 7.143502537731286e-05
# 114 7.143502537731286e-05
# graph size smaller than the default end dimension, thus has been automatically set to 158
# the optimal dimension at 0.05 accuracy level is 56
# the MSE of curve fitting is 6.907871925187393e-05
# 56 6.907871925187393e-05
# EMBEDDING DIM (MEDIAN) FOR haydn: 56
# Score for test centroid at index 0 (mean * std): 0.014030182792217488
# Score for test centroid at index 1 (mean * std): 0.9290382861970637
# Score for test centroid at index 2 (mean * std): 1.1908589232080062
# Score for test centroid at index 3 (mean * std): 1.2133069693134506
# Score for test centroid at index 4 (mean * std): 1.2337415122277966
# The graph at index 0 has the best score of 0.014030182792217488.

# graph size smaller than the default end dimension, thus has been automatically set to 24
# the optimal dimension at 0.05 accuracy level is 17
# the MSE of curve fitting is 0.0004589956565186793
# 17 0.0004589956565186793
# graph size smaller than the default end dimension, thus has been automatically set to 177
# the optimal dimension at 0.05 accuracy level is 63
# the MSE of curve fitting is 7.127694646196501e-05
# 63 7.127694646196501e-05
# graph size smaller than the default end dimension, thus has been automatically set to 133
# the optimal dimension at 0.05 accuracy level is 42
# the MSE of curve fitting is 3.578064328037125e-05
# 42 3.578064328037125e-05
# graph size smaller than the default end dimension, thus has been automatically set to 98
# the optimal dimension at 0.05 accuracy level is 33
# the MSE of curve fitting is 7.239625561330044e-05
# 33 7.239625561330044e-05
# graph size smaller than the default end dimension, thus has been automatically set to 137
# the optimal dimension at 0.05 accuracy level is 60
# the MSE of curve fitting is 5.8278883175286464e-05
# 60 5.8278883175286464e-05
# graph size smaller than the default end dimension, thus has been automatically set to 150
# the optimal dimension at 0.05 accuracy level is 57
# the MSE of curve fitting is 7.463463380722458e-05
# 57 7.463463380722458e-05
# EMBEDDING DIM (MEDIAN) FOR mozart: 49
# Score for test centroid at index 0 (mean * std): 0.010902131851484798
# Score for test centroid at index 1 (mean * std): 0.9694873525502807
# Score for test centroid at index 2 (mean * std): 0.9643141496665758
# Score for test centroid at index 3 (mean * std): 0.9611154881238142
# Score for test centroid at index 4 (mean * std): 0.8029068269485747
# Score for test centroid at index 5 (mean * std): 0.9724303858296832
# The graph at index 0 has the best score of 0.010902131851484798.

#--------------------------------------------------------------------------------------------------------------------------------------------
# ABLATION 3 LEVELS EUCLIDEAN DIST RESULTS

# graph size smaller than the default end dimension, thus has been automatically set to 36
# the optimal dimension at 0.05 accuracy level is 35
# the MSE of curve fitting is 0.00010739595702196074
# 35 0.00010739595702196074
# graph size smaller than the default end dimension, thus has been automatically set to 111
# the optimal dimension at 0.05 accuracy level is 33
# the MSE of curve fitting is 4.555192399863674e-05
# 33 4.555192399863674e-05
# graph size smaller than the default end dimension, thus has been automatically set to 140
# the optimal dimension at 0.05 accuracy level is 38
# the MSE of curve fitting is 5.262432483680921e-05
# 38 5.262432483680921e-05
# graph size smaller than the default end dimension, thus has been automatically set to 179
# the optimal dimension at 0.05 accuracy level is 65
# the MSE of curve fitting is 7.558483673484633e-05
# 65 7.558483673484633e-05
# graph size smaller than the default end dimension, thus has been automatically set to 134
# the optimal dimension at 0.05 accuracy level is 55
# the MSE of curve fitting is 9.487565041240426e-05
# 55 9.487565041240426e-05
# graph size smaller than the default end dimension, thus has been automatically set to 346
# the optimal dimension at 0.05 accuracy level is 144
# the MSE of curve fitting is 3.849205868625104e-05
# 144 3.849205868625104e-05
# graph size smaller than the default end dimension, thus has been automatically set to 173
# the optimal dimension at 0.05 accuracy level is 51
# the MSE of curve fitting is 7.950745321805397e-05
# 51 7.950745321805397e-05
# EMBEDDING DIM (MEDIAN) FOR bach: 51
# Score for test centroid at index 0 (mean * std): 0.007853084411191535
# Score for test centroid at index 1 (mean * std): 0.4945054352998346
# Score for test centroid at index 2 (mean * std): 0.4811083732455434
# Score for test centroid at index 3 (mean * std): 0.42354343019769625
# Score for test centroid at index 4 (mean * std): 0.531765109168171
# Score for test centroid at index 5 (mean * std): 0.649207902465212
# Score for test centroid at index 6 (mean * std): 0.6436003273618688

# The graph at index 0 has the best score of 0.007853084411191535.
# graph size smaller than the default end dimension, thus has been automatically set to 58
# the optimal dimension at 0.05 accuracy level is 57
# the MSE of curve fitting is 3.718530007972513e-05
# 57 3.718530007972513e-05
# graph size smaller than the default end dimension, thus has been automatically set to 200
# the optimal dimension at 0.05 accuracy level is 111
# the MSE of curve fitting is 6.978430339866881e-05
# 111 6.978430339866881e-05
# graph size smaller than the default end dimension, thus has been automatically set to 40
# the optimal dimension at 0.05 accuracy level is 8
# the MSE of curve fitting is 2.003124176380592e-05
# 8 2.003124176380592e-05
# graph size smaller than the default end dimension, thus has been automatically set to 48
# the optimal dimension at 0.05 accuracy level is 16
# the MSE of curve fitting is 3.390388975076219e-05
# 16 3.390388975076219e-05
# graph size smaller than the default end dimension, thus has been automatically set to 73
# the optimal dimension at 0.05 accuracy level is 35
# the MSE of curve fitting is 8.568244646114773e-05
# 35 8.568244646114773e-05
# graph size smaller than the default end dimension, thus has been automatically set to 55
# the optimal dimension at 0.05 accuracy level is 17
# the MSE of curve fitting is 3.9191239381903444e-05
# 17 3.9191239381903444e-05
# graph size smaller than the default end dimension, thus has been automatically set to 34
# the optimal dimension at 0.05 accuracy level is 7
# the MSE of curve fitting is 0.00012125679320425614
# 7 0.00012125679320425614
# graph size smaller than the default end dimension, thus has been automatically set to 63
# the optimal dimension at 0.05 accuracy level is 16
# the MSE of curve fitting is 1.4533442524253327e-05
# 16 1.4533442524253327e-05
# graph size smaller than the default end dimension, thus has been automatically set to 104
# the optimal dimension at 0.05 accuracy level is 36
# the MSE of curve fitting is 4.4188836535100804e-05
# 36 4.4188836535100804e-05
# graph size smaller than the default end dimension, thus has been automatically set to 147
# the optimal dimension at 0.05 accuracy level is 64
# the MSE of curve fitting is 8.050074750539301e-05
# 64 8.050074750539301e-05
# EMBEDDING DIM (MEDIAN) FOR beethoven: 26
# Score for test centroid at index 0 (mean * std): 0.016526311290681706
# Score for test centroid at index 1 (mean * std): 0.43199541353651283
# Score for test centroid at index 2 (mean * std): 0.3177279710974212
# Score for test centroid at index 3 (mean * std): 0.42643886603217657
# Score for test centroid at index 4 (mean * std): 0.4347420451643818
# Score for test centroid at index 5 (mean * std): 0.3180664385672385
# Score for test centroid at index 6 (mean * std): 0.2805464464608046
# Score for test centroid at index 7 (mean * std): 0.43217095877502343
# Score for test centroid at index 8 (mean * std): 0.4372879243525327
# Score for test centroid at index 9 (mean * std): 0.4344731316009534
# The graph at index 0 has the best score of 0.016526311290681706.

# graph size smaller than the default end dimension, thus has been automatically set to 24
# the optimal dimension at 0.05 accuracy level is 10
# the MSE of curve fitting is 0.00015808519352129854
# 10 0.00015808519352129854
# graph size smaller than the default end dimension, thus has been automatically set to 102
# the optimal dimension at 0.05 accuracy level is 33
# the MSE of curve fitting is 0.00011424812055742955
# 33 0.00011424812055742955
# graph size smaller than the default end dimension, thus has been automatically set to 163
# the optimal dimension at 0.05 accuracy level is 101
# the MSE of curve fitting is 6.310250870736487e-05
# 101 6.310250870736487e-05
# graph size smaller than the default end dimension, thus has been automatically set to 182
# the optimal dimension at 0.05 accuracy level is 94
# the MSE of curve fitting is 5.592055029918522e-05
# 94 5.592055029918522e-05
# graph size smaller than the default end dimension, thus has been automatically set to 158
# the optimal dimension at 0.05 accuracy level is 70
# the MSE of curve fitting is 8.077254505179799e-05
# 70 8.077254505179799e-05
# EMBEDDING DIM (MEDIAN) FOR haydn: 70
# Score for test centroid at index 0 (mean * std): 0.007066520564892582
# Score for test centroid at index 1 (mean * std): 0.8335720036617511
# Score for test centroid at index 2 (mean * std): 1.0511658606803989
# Score for test centroid at index 3 (mean * std): 1.2020472678305887
# Score for test centroid at index 4 (mean * std): 1.5345848498779717
# The graph at index 0 has the best score of 0.007066520564892582.

# graph size smaller than the default end dimension, thus has been automatically set to 22
# the optimal dimension at 0.05 accuracy level is 9
# the MSE of curve fitting is 8.219087944907621e-05
# 9 8.219087944907621e-05
# graph size smaller than the default end dimension, thus has been automatically set to 177
# the optimal dimension at 0.05 accuracy level is 63
# the MSE of curve fitting is 6.041499790127287e-05
# 63 6.041499790127287e-05
# graph size smaller than the default end dimension, thus has been automatically set to 133
# the optimal dimension at 0.05 accuracy level is 44
# the MSE of curve fitting is 7.320319094647193e-05
# 44 7.320319094647193e-05
# graph size smaller than the default end dimension, thus has been automatically set to 98
# the optimal dimension at 0.05 accuracy level is 22
# the MSE of curve fitting is 3.5152229463530233e-05
# 22 3.5152229463530233e-05
# graph size smaller than the default end dimension, thus has been automatically set to 137
# the optimal dimension at 0.05 accuracy level is 61
# the MSE of curve fitting is 8.718157135703527e-05
# 61 8.718157135703527e-05
# graph size smaller than the default end dimension, thus has been automatically set to 150
# the optimal dimension at 0.05 accuracy level is 53
# the MSE of curve fitting is 8.499048186866481e-05
# 53 8.499048186866481e-05
# EMBEDDING DIM (MEDIAN) FOR mozart: 48
# Score for test centroid at index 0 (mean * std): 0.009851476829442537
# Score for test centroid at index 1 (mean * std): 0.9494769291043287
# Score for test centroid at index 2 (mean * std): 0.9452212945298389
# Score for test centroid at index 3 (mean * std): 0.9419470682078197
# Score for test centroid at index 4 (mean * std): 0.7844146778020202
# Score for test centroid at index 5 (mean * std): 0.7918264344708834
# The graph at index 0 has the best score of 0.009851476829442537.

#--------------------------------------------------------------------------------------------------------------------------------------------
# ABLATION 4 LEVELS RESULTS
# graph size smaller than the default end dimension, thus has been automatically set to 38
# the optimal dimension at 0.05 accuracy level is 53
# the MSE of curve fitting is 0.00014270917370611642
# 53 0.00014270917370611642
# graph size smaller than the default end dimension, thus has been automatically set to 111
# the optimal dimension at 0.05 accuracy level is 38
# the MSE of curve fitting is 8.575327167099736e-05
# 38 8.575327167099736e-05
# graph size smaller than the default end dimension, thus has been automatically set to 140
# the optimal dimension at 0.05 accuracy level is 28
# the MSE of curve fitting is 5.942421241551215e-05
# 28 5.942421241551215e-05
# graph size smaller than the default end dimension, thus has been automatically set to 179
# the optimal dimension at 0.05 accuracy level is 66
# the MSE of curve fitting is 0.00010142934885219916
# 66 0.00010142934885219916
# graph size smaller than the default end dimension, thus has been automatically set to 134
# the optimal dimension at 0.05 accuracy level is 43
# the MSE of curve fitting is 0.00013729338425951432
# 43 0.00013729338425951432
# graph size smaller than the default end dimension, thus has been automatically set to 346
# the optimal dimension at 0.05 accuracy level is 139
# the MSE of curve fitting is 2.948900136995238e-05
# 139 2.948900136995238e-05
# graph size smaller than the default end dimension, thus has been automatically set to 173
# the optimal dimension at 0.05 accuracy level is 54
# the MSE of curve fitting is 7.635987437935418e-05
# 54 7.635987437935418e-05
# EMBEDDING DIM (MEDIAN) FOR bach: 53
# Score for test centroid at index 0 (mean * std): 0.01125368117461237
# Score for test centroid at index 1 (mean * std): 0.5007300230417491
# Score for test centroid at index 2 (mean * std): 0.46856646307722977
# Score for test centroid at index 3 (mean * std): 0.4490965608490357
# Score for test centroid at index 4 (mean * std): 0.6644597500680848
# Score for test centroid at index 5 (mean * std): 0.6755118948952035
# Score for test centroid at index 6 (mean * std): 0.6686214036078975
# The graph at index 0 has the best score of 0.01125368117461237.

# graph size smaller than the default end dimension, thus has been automatically set to 32
# the optimal dimension at 0.05 accuracy level is 33
# the MSE of curve fitting is 0.0002693695567374304
# 33 0.0002693695567374304
# graph size smaller than the default end dimension, thus has been automatically set to 200
# the optimal dimension at 0.05 accuracy level is 99
# the MSE of curve fitting is 4.0619430971936813e-05
# 99 4.0619430971936813e-05
# graph size smaller than the default end dimension, thus has been automatically set to 40
# the optimal dimension at 0.05 accuracy level is 10
# the MSE of curve fitting is 6.260497927029932e-05
# 10 6.260497927029932e-05
# graph size smaller than the default end dimension, thus has been automatically set to 48
# the optimal dimension at 0.05 accuracy level is 14
# the MSE of curve fitting is 6.475198560919985e-05
# 14 6.475198560919985e-05
# graph size smaller than the default end dimension, thus has been automatically set to 73
# the optimal dimension at 0.05 accuracy level is 25
# the MSE of curve fitting is 7.111908629174981e-05
# 25 7.111908629174981e-05
# graph size smaller than the default end dimension, thus has been automatically set to 55
# the optimal dimension at 0.05 accuracy level is 12
# the MSE of curve fitting is 5.153425701037339e-05
# 12 5.153425701037339e-05
# graph size smaller than the default end dimension, thus has been automatically set to 34
# the optimal dimension at 0.05 accuracy level is 6
# the MSE of curve fitting is 0.0003570651227482032
# 6 0.0003570651227482032
# graph size smaller than the default end dimension, thus has been automatically set to 63
# the optimal dimension at 0.05 accuracy level is 17
# the MSE of curve fitting is 6.006843363420701e-05
# 17 6.006843363420701e-05
# graph size smaller than the default end dimension, thus has been automatically set to 104
# the optimal dimension at 0.05 accuracy level is 29
# the MSE of curve fitting is 2.530558315471385e-05
# 29 2.530558315471385e-05
# graph size smaller than the default end dimension, thus has been automatically set to 147
# the optimal dimension at 0.05 accuracy level is 67
# the MSE of curve fitting is 8.073433103119955e-05
# 67 8.073433103119955e-05
# EMBEDDING DIM (MEDIAN) FOR beethoven: 21
# Score for test centroid at index 0 (mean * std): 0.012056755498368505
# Score for test centroid at index 1 (mean * std): 0.34930717909610187
# Score for test centroid at index 2 (mean * std): 0.25052389137039055
# Score for test centroid at index 3 (mean * std): 0.2482166695426522
# Score for test centroid at index 4 (mean * std): 0.352789708224622
# Score for test centroid at index 5 (mean * std): 0.24937050409553926
# Score for test centroid at index 6 (mean * std): 0.34479213164215516
# Score for test centroid at index 7 (mean * std): 0.24828657472827884
# Score for test centroid at index 8 (mean * std): 0.3528950611432443
# Score for test centroid at index 9 (mean * std): 0.35020789873819946
# The graph at index 0 has the best score of 0.012056755498368505.

# graph size smaller than the default end dimension, thus has been automatically set to 68
# the optimal dimension at 0.05 accuracy level is 76
# the MSE of curve fitting is 7.991821019300072e-05
# 76 7.991821019300072e-05
# graph size smaller than the default end dimension, thus has been automatically set to 102
# the optimal dimension at 0.05 accuracy level is 27
# the MSE of curve fitting is 4.1927130901876246e-05
# 27 4.1927130901876246e-05
# graph size smaller than the default end dimension, thus has been automatically set to 163
# the optimal dimension at 0.05 accuracy level is 77
# the MSE of curve fitting is 4.4094615706498876e-05
# 77 4.4094615706498876e-05
# graph size smaller than the default end dimension, thus has been automatically set to 182
# the optimal dimension at 0.05 accuracy level is 103
# the MSE of curve fitting is 7.198594935768476e-05
# 103 7.198594935768476e-05
# graph size smaller than the default end dimension, thus has been automatically set to 158
# the optimal dimension at 0.05 accuracy level is 79
# the MSE of curve fitting is 9.583306906881717e-05
# 79 9.583306906881717e-05
# EMBEDDING DIM (MEDIAN) FOR haydn: 77
# Score for test centroid at index 0 (mean * std): 0.021187877697135944
# Score for test centroid at index 1 (mean * std): 0.8182770398103711
# Score for test centroid at index 2 (mean * std): 1.002879095924139
# Score for test centroid at index 3 (mean * std): 1.126474818959798
# Score for test centroid at index 4 (mean * std): 1.6866800277018692
# The graph at index 0 has the best score of 0.021187877697135944.

# graph size smaller than the default end dimension, thus has been automatically set to 65
# the optimal dimension at 0.05 accuracy level is 76
# the MSE of curve fitting is 5.5241394354278625e-05
# 76 5.5241394354278625e-05
# graph size smaller than the default end dimension, thus has been automatically set to 177
# the optimal dimension at 0.05 accuracy level is 78
# the MSE of curve fitting is 0.0001020570142840232
# 78 0.0001020570142840232
# graph size smaller than the default end dimension, thus has been automatically set to 133
# the optimal dimension at 0.05 accuracy level is 37
# the MSE of curve fitting is 6.227010587369685e-05
# 37 6.227010587369685e-05
# graph size smaller than the default end dimension, thus has been automatically set to 98
# the optimal dimension at 0.05 accuracy level is 32
# the MSE of curve fitting is 9.373560677988344e-05
# 32 9.373560677988344e-05
# graph size smaller than the default end dimension, thus has been automatically set to 137
# the optimal dimension at 0.05 accuracy level is 53
# the MSE of curve fitting is 7.154297960245186e-05
# 53 7.154297960245186e-05
# graph size smaller than the default end dimension, thus has been automatically set to 150
# the optimal dimension at 0.05 accuracy level is 58
# the MSE of curve fitting is 5.944082325153983e-05
# 58 5.944082325153983e-05
# EMBEDDING DIM (MEDIAN) FOR mozart: 55
# Score for test centroid at index 0 (mean * std): 0.03174971448927805
# Score for test centroid at index 1 (mean * std): 1.087165793310889
# Score for test centroid at index 2 (mean * std): 1.0782992918157888
# Score for test centroid at index 3 (mean * std): 0.8306162450794055
# Score for test centroid at index 4 (mean * std): 0.8611404747782938
# Score for test centroid at index 5 (mean * std): 1.0892656830960392
# The graph at index 0 has the best score of 0.03174971448927805.



