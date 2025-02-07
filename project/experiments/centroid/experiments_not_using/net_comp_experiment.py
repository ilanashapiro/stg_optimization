import os, sys, pickle, json, re
import numpy as np
import netcomp
import spectral_experiment
import networkx as nx

# DIRECTORY = "/home/ubuntu/project"
# DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
DIRECTORY = "/home/ilshapiro/project"
TIME_PARAM = '50s'
ABLATION_LEVEL = 1

sys.path.append(f"{DIRECTORY}/centroid")
import simanneal_centroid_helpers

def adjacency_matrix_to_nx(adjacency_matrix):
	return nx.from_numpy_array(adjacency_matrix, create_using=nx.Graph)

if __name__ == "__main__":
		if ABLATION_LEVEL:
			centroid_path = f"{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}_ablation{ABLATION_LEVEL}"
		else:
			centroid_path = f"{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}"

		training_pieces_path = f"{DIRECTORY}/experiments/centroid/clusters/composer_centroid_input_graphs_{TIME_PARAM}.txt"

		# Load the centroid and training pieces
		composer_centroids_dict = spectral_experiment.load_centroids()
		composer_centroids_dict = {k: composer_centroids_dict[k] for k in sorted(composer_centroids_dict)}

		with open(training_pieces_path, 'r') as file:
			composer_training_pieces_paths = json.load(file)
			def load_graph(file_path):
				with open(file_path, 'rb') as f:
					graph = pickle.load(f)
				return graph
			composer_training_pieces_dict = {}
			for composer, filepaths in composer_training_pieces_paths.items():
				if ABLATION_LEVEL:
					new_suffix = f"_ablation_{ABLATION_LEVEL}level_flat.pickle"
					composer_training_pieces_dict[composer] = [load_graph(re.sub(r'^.*?/project', DIRECTORY, file_path)[:-len("_flat.pickle")] + new_suffix) for file_path in filepaths]
				else:
					composer_training_pieces_dict[composer] = [load_graph(re.sub(r'^.*?/project', DIRECTORY, file_path)) for file_path in filepaths]


		for composer, centroid in composer_centroids_dict.items():
				training_pieces = composer_training_pieces_dict[composer]
				listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([centroid] + training_pieces)
				scores = []

				# Iterate over each graph in the corpus, treating each as the candidate centroid
				for i, test_centroid in enumerate(listA_G):
						distances = [
								netcomp.deltacon0(test_centroid, test_graph)
								for j, test_graph in enumerate(listA_G) if j > 0 # exclude the original centroid
						]
						score = np.mean(distances) * np.std(distances)
						scores.append(score)

						print(f"Score when graph {i} is treated as the centroid: {score:.4f}")

				# Identify the graph with the highest score as the best centroid
				best_score = min(scores)
				best_graph_index = scores.index(best_score)

				print(f"The most representative graph in the corpus is graph {best_graph_index} with an average score of {best_score:.4f}, with ablation level {ABLATION_LEVEL}")

#--------------------------------------------------------------------------------------------------------------------------------------------
# RESULTS FOR DELTACON0 DIST, NO ABLATION/FULL STG
# Bach
# graph 0 is the candidate centroid in each group. we want the lowest score (mean * std) of the distances
# Score when graph 0 is treated as the centroid: 5758.7216
# Score when graph 1 is treated as the centroid: 15103.2771
# Score when graph 2 is treated as the centroid: 15377.8457
# Score when graph 3 is treated as the centroid: 18000.2300
# Score when graph 4 is treated as the centroid: 14918.1085
# Score when graph 5 is treated as the centroid: 23056.7677
# Score when graph 6 is treated as the centroid: 17413.2004
# The most representative graph in the corpus is graph 0 with an average score of 5758.7216.

# Beethoven
# Score when graph 0 is treated as the centroid: 3041.2497
# Score when graph 1 is treated as the centroid: 11112.1401
# Score when graph 2 is treated as the centroid: 8484.5735
# Score when graph 3 is treated as the centroid: 8190.1356
# Score when graph 4 is treated as the centroid: 8759.3727
# Score when graph 5 is treated as the centroid: 7479.4534
# Score when graph 6 is treated as the centroid: 6483.1636
# Score when graph 7 is treated as the centroid: 7325.4205
# Score when graph 8 is treated as the centroid: 10469.6367
# Score when graph 9 is treated as the centroid: 11878.4789
# The most representative graph in the corpus is graph 0 with an average score of 3041.2497.

# Haydn
# Score when graph 0 is treated as the centroid: 3001.7722
# Score when graph 1 is treated as the centroid: 12434.9219
# Score when graph 2 is treated as the centroid: 12621.0919
# Score when graph 3 is treated as the centroid: 11697.2503
# Score when graph 4 is treated as the centroid: 13467.9854
# The most representative graph in the corpus is graph 0 with an average score of 3001.7722.

# Mozart
# Score when graph 0 is treated as the centroid: 4121.1771
# Score when graph 1 is treated as the centroid: 14803.7125
# Score when graph 2 is treated as the centroid: 15808.8600
# Score when graph 3 is treated as the centroid: 13480.9526
# Score when graph 4 is treated as the centroid: 11228.4564
# Score when graph 5 is treated as the centroid: 14857.0757
# The most representative graph in the corpus is graph 0 with an average score of 4121.1771.

#--------------------------------------------------------------------------------------------------------------------------------------------
# RESULTS FOR DELTACON0 DIST, ABLATION 4 LEVEL
# BACH
# Score when graph 0 is treated as the centroid: 2730.2237
# Score when graph 1 is treated as the centroid: 3436.6001
# Score when graph 2 is treated as the centroid: 3296.0670
# Score when graph 3 is treated as the centroid: 4092.2290
# Score when graph 4 is treated as the centroid: 3480.9912
# Score when graph 5 is treated as the centroid: 5831.0557
# Score when graph 6 is treated as the centroid: 4016.0979
# The most representative graph in the corpus is graph 0 with an average score of 2730.2237, with ablation level 4

# Beethoven
# Score when graph 0 is treated as the centroid: 1939.7410
# Score when graph 1 is treated as the centroid: 3877.8020
# Score when graph 2 is treated as the centroid: 2227.3566
# Score when graph 3 is treated as the centroid: 2688.2116
# Score when graph 4 is treated as the centroid: 2647.9585
# Score when graph 5 is treated as the centroid: 1914.6180
# Score when graph 6 is treated as the centroid: 1846.7619
# Score when graph 7 is treated as the centroid: 2144.6716
# Score when graph 8 is treated as the centroid: 3268.0577
# Score when graph 9 is treated as the centroid: 3702.7038
# The most representative graph in the corpus is graph 6 with an average score of 1846.7619, with ablation level 4

# Haydn
# Score when graph 0 is treated as the centroid: 1937.3421
# Score when graph 1 is treated as the centroid: 3639.2744
# Score when graph 2 is treated as the centroid: 4037.1142
# Score when graph 3 is treated as the centroid: 3730.6989
# Score when graph 4 is treated as the centroid: 3845.3842
# The most representative graph in the corpus is graph 0 with an average score of 1937.3421, with ablation level 4

# Mozart
# Score when graph 0 is treated as the centroid: 2197.9949
# Score when graph 1 is treated as the centroid: 3401.0980
# Score when graph 2 is treated as the centroid: 4258.3645
# Score when graph 3 is treated as the centroid: 2578.7043
# Score when graph 4 is treated as the centroid: 3258.7692
# Score when graph 5 is treated as the centroid: 3447.1943
# The most representative graph in the corpus is graph 0 with an average score of 2197.9949, with ablation level 4

#--------------------------------------------------------------------------------------------------------------------------------------------
# RESULTS FOR DELTACON0 DIST, ABLATION 3 LEVEL
# BACH
# Score when graph 0 is treated as the centroid: 999.1480
# Score when graph 1 is treated as the centroid: 787.0172
# Score when graph 2 is treated as the centroid: 984.1282
# Score when graph 3 is treated as the centroid: 938.7673
# Score when graph 4 is treated as the centroid: 655.2547
# Score when graph 5 is treated as the centroid: 1511.9829
# Score when graph 6 is treated as the centroid: 932.1942
# The most representative graph in the corpus is graph 4 with an average score of 655.2547, with ablation level 3

# BEETHOVEN
# Score when graph 0 is treated as the centroid: 678.7409
# Score when graph 1 is treated as the centroid: 903.0421
# Score when graph 2 is treated as the centroid: 465.1311
# Score when graph 3 is treated as the centroid: 688.4590
# Score when graph 4 is treated as the centroid: 799.9320
# Score when graph 5 is treated as the centroid: 479.4630
# Score when graph 6 is treated as the centroid: 540.8579
# Score when graph 7 is treated as the centroid: 573.1249
# Score when graph 8 is treated as the centroid: 758.8314
# Score when graph 9 is treated as the centroid: 1113.0981
# The most representative graph in the corpus is graph 2 with an average score of 465.1311, with ablation level 3

# HAYDN
# Score when graph 0 is treated as the centroid: 300.2609
# Score when graph 1 is treated as the centroid: 1214.5085
# Score when graph 2 is treated as the centroid: 1363.4954
# Score when graph 3 is treated as the centroid: 1536.2001
# Score when graph 4 is treated as the centroid: 1647.7821
# The most representative graph in the corpus is graph 0 with an average score of 300.2609, with ablation level 3

# MOZART
# Score when graph 0 is treated as the centroid: 591.3670
# Score when graph 1 is treated as the centroid: 985.5883
# Score when graph 2 is treated as the centroid: 1078.2906
# Score when graph 3 is treated as the centroid: 622.7225
# Score when graph 4 is treated as the centroid: 989.1205
# Score when graph 5 is treated as the centroid: 1134.4431
# The most representative graph in the corpus is graph 0 with an average score of 591.3670, with ablation level 3

#--------------------------------------------------------------------------------------------------------------------------------------------
# RESULTS FOR DELTACON0 DIST, ABLATION 2 LEVEL
# BACH
# Score when graph 0 is treated as the centroid: 787.1044
# Score when graph 1 is treated as the centroid: 581.9835
# Score when graph 2 is treated as the centroid: 758.2824
# Score when graph 3 is treated as the centroid: 706.9336
# Score when graph 4 is treated as the centroid: 452.1776
# Score when graph 5 is treated as the centroid: 1044.4496
# Score when graph 6 is treated as the centroid: 668.8930
# The most representative graph in the corpus is graph 4 with an average score of 452.1776, with ablation level 2

# BEETHOVEN
# Score when graph 0 is treated as the centroid: 685.0950
# Score when graph 1 is treated as the centroid: 559.1432
# Score when graph 2 is treated as the centroid: 217.0641
# Score when graph 3 is treated as the centroid: 223.4581
# Score when graph 4 is treated as the centroid: 229.3700
# Score when graph 5 is treated as the centroid: 202.0061
# Score when graph 6 is treated as the centroid: 217.4382
# Score when graph 7 is treated as the centroid: 302.5385
# Score when graph 8 is treated as the centroid: 217.4382
# Score when graph 9 is treated as the centroid: 600.3571
# The most representative graph in the corpus is graph 5 with an average score of 202.0061, with ablation level 2

# HAYDN
# Score when graph 0 is treated as the centroid: 881.1399
# Score when graph 1 is treated as the centroid: 587.7407
# Score when graph 2 is treated as the centroid: 879.5931
# Score when graph 3 is treated as the centroid: 993.3021
# Score when graph 4 is treated as the centroid: 850.1741
# The most representative graph in the corpus is graph 1 with an average score of 587.7407, with ablation level 2

# MOZART
# Score when graph 0 is treated as the centroid: 362.8245
# Score when graph 1 is treated as the centroid: 444.0123
# Score when graph 2 is treated as the centroid: 534.4845
# Score when graph 3 is treated as the centroid: 335.9017
# Score when graph 4 is treated as the centroid: 555.6425
# Score when graph 5 is treated as the centroid: 554.5008
# The most representative graph in the corpus is graph 3 with an average score of 335.9017, with ablation level 2

#--------------------------------------------------------------------------------------------------------------------------------------------
# RESULTS FOR DELTACON0 DIST, ABLATION 2 LEVEL
# BACH
# Score when graph 0 is treated as the centroid: 151.1151
# Score when graph 1 is treated as the centroid: 12.1556
# Score when graph 2 is treated as the centroid: 11.6735
# Score when graph 3 is treated as the centroid: 12.1556
# Score when graph 4 is treated as the centroid: 8.2740
# Score when graph 5 is treated as the centroid: 19.4409
# Score when graph 6 is treated as the centroid: 12.1556
# The most representative graph in the corpus is graph 4 with an average score of 8.2740, with ablation level 1

# BEETHOVEN
# Score when graph 0 is treated as the centroid: 75.6783
# Score when graph 1 is treated as the centroid: 4.1494
# Score when graph 2 is treated as the centroid: 4.3788
# Score when graph 3 is treated as the centroid: 4.3788
# Score when graph 4 is treated as the centroid: 2.3943
# Score when graph 5 is treated as the centroid: 2.3943
# Score when graph 6 is treated as the centroid: 2.3943
# Score when graph 7 is treated as the centroid: 4.1494
# Score when graph 8 is treated as the centroid: 2.3943
# Score when graph 9 is treated as the centroid: 4.1494
# The most representative graph in the corpus is graph 4 with an average score of 2.3943, with ablation level 1

# HAYDN
# Score when graph 0 is treated as the centroid: 90.4262
# Score when graph 1 is treated as the centroid: 1.0993
# Score when graph 2 is treated as the centroid: 1.0993
# Score when graph 3 is treated as the centroid: 1.0993
# Score when graph 4 is treated as the centroid: 3.2980
# The most representative graph in the corpus is graph 1 with an average score of 1.0993, with ablation level 1

# MOZART
# Score when graph 0 is treated as the centroid: 41.2591
# Score when graph 1 is treated as the centroid: 4.6023
# Score when graph 2 is treated as the centroid: 10.9639
# Score when graph 3 is treated as the centroid: 4.6023
# Score when graph 4 is treated as the centroid: 4.6023
# Score when graph 5 is treated as the centroid: 7.2219
# The most representative graph in the corpus is graph 1 with an average score of 4.6023, with ablation level 1