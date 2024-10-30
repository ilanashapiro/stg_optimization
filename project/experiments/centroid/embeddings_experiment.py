import os, sys, pickle, json, re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from karateclub import Graph2Vec
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
        
        # Pad adjacency matrices to uniform size
        listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([centroid] + training_pieces)
        A_g = listA_G[0]  # The centroid graph
        listA_G = listA_G[1:]  # Separate input graphs from the centroid

        # Convert graphs to embeddings using Graph2Vec
        graph2vec_model = Graph2Vec(dimensions=128, wl_iterations=5)  # Customize embedding dimensions as needed
        graph2vec_model.fit([adjacency_matrix_to_nx(A_g)] + [adjacency_matrix_to_nx(A_G) for A_G in listA_G])
        graph_embeddings = graph2vec_model.get_embedding()

        # Store average cosine similarities for each graph treated as the candidate centroid
        average_similarities = []

        # Iterate over each graph in the corpus, treating each as the candidate centroid
        for i, test_embedding in enumerate(graph_embeddings):
            similarities = [
                cosine_similarity([test_embedding], [embedding])[0][0]
                for j, embedding in enumerate(graph_embeddings) if j > 0 # exclude the original centroid
            ]
            avg_similarity = np.mean(similarities)
            average_similarities.append(avg_similarity)

            # Print or log each average similarity if desired
            # print(f"Average similarity when graph {i} is treated as the centroid: {avg_similarity:.4f}")

        # Identify the graph with the highest average similarity as the best centroid
        best_similarity = max(average_similarities)
        best_graph_index = average_similarities.index(best_similarity)

        print(f"The most representative graph in the corpus is graph {best_graph_index} with an average cosine similarity of {best_similarity:.4f}.")
