import os, sys, pickle, json, re
import numpy as np
import netcomp
import spectral_experiment
import networkx as nx

DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
TIME_PARAM = '50s'
NUM_GPUS = 8
sys.path.append(DIRECTORY)
sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/structural_distance/structural_distance_experiment")

import build_graph
import structural_distance_gen_clusters as st_gen_clusters
import simanneal_centroid_run, simanneal_centroid_helpers, simanneal_centroid

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
            score = np.mean(distances) * np.std(distances)
            scores.append(score)

            print(f"Score when graph {i} is treated as the centroid: {score:.4f}")

        # Identify the graph with the highest score as the best centroid
        best_score = min(scores)
        best_graph_index = scores.index(best_score)

        print(f"The most representative graph in the corpus is graph {best_graph_index} with an average score of {best_score:.4f}.")
