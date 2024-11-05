import os, sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skbio.stats.distance import mantel
from skbio import DistanceMatrix

# DIRECTORY = "/home/ilshapiro/project"
DIRECTORY = "/home/ubuntu/project"
# DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
sys.path.append(f"{DIRECTORY}/experiments/structural_distance/structural_distance_experiment")
sys.path.append(f"{DIRECTORY}/experiments/structural_distance/baseline1_midi_features")
sys.path.append(f"{DIRECTORY}/experiments/structural_distance/baseline2_audio_similarity")

import structural_distance_process_clusters as gen_data
import baseline1_features_vec as baseline1
import baseline2_audio_similarity as baseline2

def normalize_upper_triangle(matrix, scaler):
	upper_tri_indices = np.triu_indices_from(matrix, k=1)
	upper_tri_values = matrix[upper_tri_indices]
	upper_tri_values_normalized = scaler.fit_transform(upper_tri_values.reshape(-1, 1)).flatten()
	matrix[upper_tri_indices] = upper_tri_values_normalized
	matrix.T[upper_tri_indices] = upper_tri_values_normalized  # Reflect to lower triangle
	return matrix

def ensure_hollow(matrix):
	for i in range(matrix.shape[0]):
		matrix[i][i] = 0
	return matrix

if __name__ == "__main__":
	scaler = MinMaxScaler()
	clusters_path = f"{DIRECTORY}/experiments/structural_distance/structural_distance_experiment/clusters_totalnobrahmsnohaydn_mindisttol2.pkl" 

	ground_truth_normalized_similarity_matrix = normalize_upper_triangle(np.array([
			[ 0.  ,  2.58,  2.58,  0.59,  0.61],
			[ 2.58,  0.  , 62.78, 64.83,  0.87],
			[ 2.58, 62.78,  0.  , 95.73, 32.75],
			[ 0.59, 64.83, 95.73,  0.  , 81.57],
			[ 0.61,  0.87, 32.75, 81.57,  0.  ]
	]), scaler)
	ground_truth_normalized_dist_matrix = ensure_hollow(1 - ground_truth_normalized_similarity_matrix)
	
	struct_dist_normalized_dist_matrix = ensure_hollow(normalize_upper_triangle(gen_data.run(clusters_path), scaler)) # structural distance
	# baseline1_normalized_dist_matrix = ensure_hollow(normalize_upper_triangle(baseline1.run(clusters_path), scaler)) # cosine distance between MIDI feature vectors
	# baseline2_normalized_dist_matrix = ensure_hollow(1 - normalize_upper_triangle(baseline2.run(clusters_path), scaler)) # audio *similarity*, not distance

	method = 'spearman' 
	struct_dist_mantel_result = mantel(ground_truth_normalized_dist_matrix, struct_dist_normalized_dist_matrix, method)
	# baseline1_mantel_result = mantel(ground_truth_normalized_dist_matrix, baseline1_normalized_dist_matrix, method)
	# baseline2_mantel_result = mantel(ground_truth_normalized_dist_matrix, baseline2_normalized_dist_matrix, method)

	print(f"Mantel Test Result for Structural Distance Matrix with {method}:")
	print(f"Correlation: {struct_dist_mantel_result[0]:.4f}, p-value: {struct_dist_mantel_result[1]:.4f}")

	# print(f"Mantel Test Result for Baseline1 (Cosine Similarity from MIDI Features Vector) Matrix with {method}:")
	# print(f"Correlation: {baseline1_mantel_result[0]:.4f}, p-value: {baseline1_mantel_result[1]:.4f}")

	# print(f"Mantel Test Result for Baseline2 (Stent-Weighted Audio Similarity) Matrix with {method}:")
	# print(f"Correlation: {baseline2_mantel_result[0]:.4f}, p-value: {baseline2_mantel_result[1]:.4f}")

# for cluster: f"{DIRECTORY}/experiments/structural_distance/structural_distance_experiment/clusters_totalnobrahmsnohaydn_mindisttol2.pkl" 
# Mantel Test Result for Structure Distance Matrix with spearman:
# Correlation: 0.8207, p-value: 0.0140
# Mantel Test Result for Baseline1 (Cosine Similarity from MIDI Features Vector) Matrix with spearman:
# Correlation: 0.4681, p-value: 0.3150
# Mantel Test Result for Baseline2 (Stent-Weighted Audio Similarity) Matrix with spearman:
# Correlation: 0.5775, p-value: 0.1760

# ABLATION EXPERIMENTS for cluster: f"{DIRECTORY}/experiments/structural_distance/structural_distance_experiment/clusters_totalnobrahmsnohaydn_mindisttol2.pkl" :
# 1 level:
# Mantel Test Result for Structural Distance Matrix with spearman:
# Correlation: -0.4377, p-value: 0.2810

# 2 levels:
# Mantel Test Result for Structural Distance Matrix with spearman:
# Correlation: 0.6930, p-value: 0.1150

# 3 levels:
# Mantel Test Result for Structural Distance Matrix with spearman:
# Correlation: 0.7173, p-value: 0.0680

# 4 levels:
# Mantel Test Result for Structural Distance Matrix with spearman:
# Correlation: 0.7842, p-value: 0.0390