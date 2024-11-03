import os, sys, pickle, json, re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from karateclub import Graph2Vec
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
        
        # Pad adjacency matrices to uniform size
        listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([centroid] + training_pieces)
        A_g = listA_G[0]  # The centroid graph
        listA_G = listA_G[1:]  # Separate input graphs from the centroid

        graph2vec_model = Graph2Vec(dimensions=128, wl_iterations=5)  # Customize embedding dimensions as needed
        graph2vec_model.fit([adjacency_matrix_to_nx(A_g)] + [adjacency_matrix_to_nx(A_G) for A_G in listA_G])
        graph_embeddings = graph2vec_model.get_embedding()

        scores = []

        # Iterate over each graph in the corpus, treating each as the candidate centroid
        for i, test_embedding in enumerate(graph_embeddings):
            distances = [
                cosine_distances([test_embedding], [embedding])[0][0]
                for j, embedding in enumerate(graph_embeddings) if j > 0 # exclude the original centroid
            ]
            score = np.mean(distances) * np.std(distances)
            scores.append(score)

            # print(f"Score when graph {i} is treated as the centroid: {score:.4f}")

        # Identify the graph with the highest score as the best centroid
        best_score = min(scores)
        best_graph_index = scores.index(best_score)

        print(f"The most representative graph in the corpus is graph {best_graph_index} with an average score of {best_score:.4f}.")
