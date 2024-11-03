import os, sys, pickle, json, re
import numpy as np
import netcomp
import spectral_experiment
import networkx as nx

# DIRECTORY = "/home/ubuntu/project"
# DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
DIRECTORY = "/home/ilshapiro/project"
TIME_PARAM = '50s'

sys.path.append(f"{DIRECTORY}/centroid")
import simanneal_centroid_helpers

def load_graph(file_path):
  with open(file_path, 'rb') as f:
    graph = pickle.load(f)
  return graph

def adjacency_matrix_to_nx(adjacency_matrix):
  return nx.from_numpy_array(adjacency_matrix, create_using=nx.Graph)

if __name__ == "__main__":
    centroid_path = f"{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}"
    training_pieces_path = f"{DIRECTORY}/experiments/centroid/clusters/composer_centroid_input_graphs_{TIME_PARAM}.txt"

    # Load the centroid and training pieces
    composer_centroids_dict = spectral_experiment.load_centroids()
    composer_centroids_dict = {k: composer_centroids_dict[k] for k in sorted(composer_centroids_dict)}

    with open(training_pieces_path, 'r') as file:
        composer_training_pieces_paths = json.load(file)
        composer_training_pieces_dict = {
            composer: [load_graph(re.sub(r'^.*?/project', DIRECTORY, file_path)) 
                       for file_path in filepaths]
            for composer, filepaths in composer_training_pieces_paths.items()
        }

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
            score = np.std(distances) * np.std(distances)
            scores.append(score)

            print(f"Score when graph {i} is treated as the centroid: {score:.4f}")

        # Identify the graph with the highest score as the best centroid
        best_score = min(scores)
        best_graph_index = scores.index(best_score)

        print(f"The most representative graph in the corpus is graph {best_graph_index} with an average score of {best_score:.4f}.")

# RESULTS FOR DELTACON0 DIST 
# graph 0 is the candidate centroid in each group. we want the lowest score (mean * std) of the distances
# Score when graph 0 is treated as the centroid: 5758.7216
# Score when graph 1 is treated as the centroid: 15103.2771
# Score when graph 2 is treated as the centroid: 15377.8457
# Score when graph 3 is treated as the centroid: 18000.2300
# Score when graph 4 is treated as the centroid: 14918.1085
# Score when graph 5 is treated as the centroid: 23056.7677
# Score when graph 6 is treated as the centroid: 17413.2004
# The most representative graph in the corpus is graph 0 with an average score of 5758.7216.
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
# Score when graph 0 is treated as the centroid: 3001.7722
# Score when graph 1 is treated as the centroid: 12434.9219
# Score when graph 2 is treated as the centroid: 12621.0919
# Score when graph 3 is treated as the centroid: 11697.2503
# Score when graph 4 is treated as the centroid: 13467.9854
# The most representative graph in the corpus is graph 0 with an average score of 3001.7722.
# Score when graph 0 is treated as the centroid: 4121.1771
# Score when graph 1 is treated as the centroid: 14803.7125
# Score when graph 2 is treated as the centroid: 15808.8600
# Score when graph 3 is treated as the centroid: 13480.9526
# Score when graph 4 is treated as the centroid: 11228.4564
# Score when graph 5 is treated as the centroid: 14857.0757
# The most representative graph in the corpus is graph 0 with an average score of 4121.1771.