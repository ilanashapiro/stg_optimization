import os, sys, pickle, json, re
import math
import numpy as np
import networkx as nx

# DIRECTORY = "/home/ubuntu/project"
# DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
DIRECTORY = "/home/ilshapiro/project"
TIME_PARAM = '50s'

sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/centroid/gSpan/gspan_mining")
from gSpan.gspan_mining.gspan import gSpan
import simanneal_centroid_helpers

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

def save_line_graphs(composer_training_pieces_dict, output_dir="line_graphs"):
		for composer, graphs in composer_training_pieces_dict.items():
				# Create directory for the composer
				composer_dir = os.path.join(output_dir, composer)
				os.makedirs(composer_dir, exist_ok=True)

				# Iterate through graphs and save each line graph
				for i, line_graph in enumerate(graphs):
						# Generate output filename
						original_filepath = composer_training_pieces_paths[composer][i]
						original_filename = os.path.splitext(os.path.basename(original_filepath))[0]
						output_filepath = os.path.join(composer_dir, f"{original_filename}.txt")
						
						# Save the line graph as an edge list
						nx.write_edgelist(line_graph, output_filepath, data=False)
						print(f"Saved line graph to {output_filepath}")

def convert_to_gspan_format(graphs, file_name):
	with open(file_name, "w") as f:
		for graph_id, graph in enumerate(graphs):
			# Write graph header
			f.write(f"t # {graph_id}\n")
			
			# Write nodes (vertices)
			for node in graph.nodes:
				# node_label = graph.nodes[node].get("label", 0)  # Default label = 0 if not provided
				node_label = 0
				f.write(f"v {node} {node_label}\n")
			
			# Write edges
			for u, v, data in graph.edges(data=True):
				# edge_label = data.get("label", 0)  # Default label = 0 if not provided
				edge_label = 0
				f.write(f"e {u} {v} {edge_label}\n")
		
		# End of dataset
		f.write("t # -1\n")

def run_gspan(database_file_name, min_support):
	gs = gSpan(
				database_file_name=database_file_name,
				min_support=min_support
		)

	gs.run()
	gs.time_stats()
	return gs

if __name__ == "__main__":
	centroid_path = f"{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}"
	training_pieces_path = f"{DIRECTORY}/experiments/centroid/clusters/composer_centroid_input_graphs_{TIME_PARAM}.txt"
	
	composer_centroids_dict = load_centroids() 
	composer_centroids_dict = {k: composer_centroids_dict[k] for k in sorted(composer_centroids_dict)} # to ensure deterministic order

	with open(training_pieces_path, 'r') as file:
		composer_training_pieces_paths = json.load(file)
		def load_graph(file_path):
			with open(file_path, 'rb') as f:
				G = pickle.load(f)
			return G#nx.line_graph(G, create_using=nx.DiGraph)
		composer_training_pieces_dict = {}
		for composer, filepaths in composer_training_pieces_paths.items():
			graphs = [load_graph(re.sub(r'^.*?/project', DIRECTORY, file_path)) for file_path in filepaths]
			composer_training_pieces_dict[composer] = graphs
			convert_to_gspan_format(graphs, f"output_gspan_graphs_{composer}.data")
		# save_line_graphs(composer_training_pieces_dict)
		for composer, graphs in composer_training_pieces_dict.items():
			run_gspan(f"output_gspan_graphs_{composer}.data", math.ceil(len(graphs)/2))
			sys.exit(0)
		
	