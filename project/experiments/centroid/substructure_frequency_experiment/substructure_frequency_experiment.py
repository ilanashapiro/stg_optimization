import os, sys, pickle, json, re
import numpy as np
import torch
import networkx as nx
import rustworkx as rx
from rustworkx.visualization import mpl_draw
from collections import defaultdict
import matplotlib.pyplot as plt
import hashlib

# DIRECTORY = "/home/ubuntu/project"
DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
# DIRECTORY = "/home/ilshapiro/project"

sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/spminer/subgraph_mining")
import simanneal_centroid_helpers

ANALYZE_DERIVED_CENTROID = False

'''
This file runs the Section 6.2 experiment in the paper about examining the 5-node subgraphs common to all the STGs in each composer corpus
that we precisely enumerate with rustworkx
Then it evaluates how many of these common 5-node subgraphs appear in the centroid (i.e. making it "musically representative")
'''

def load_corpora():
	corpora_path = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/corpora/composer_centroid_input_graphs.txt"
	with open(corpora_path, 'r') as file:
		composer_corpus_pieces_paths = json.load(file)
		
	graphs = []
	composer_corpus_pieces_dict = {}
	for composer, filepaths in composer_corpus_pieces_paths.items():
		graphs = [load_pickle(re.sub(r'^.*?/project', DIRECTORY, file_path)).to_undirected() for file_path in filepaths]

		for graph in graphs:
			for node in graph.nodes:
				if node.startswith("Pr"):
					graph.nodes[node]['label'] = hash(node)
				else:
					graph.nodes[node]['label'] = hash(node[0])
			
			for u, v, data in graph.edges(data=True):
				source_label = graph.nodes[u]['label']
				sink_label = graph.nodes[v]['label']
				data['label'] = hash(f"{source_label}^{sink_label}") # data is a mutable dictionary reference to the actual edge attributes

		composer_corpus_pieces_dict[composer] = graphs
	
	return composer_corpus_pieces_dict

def load_pickle(pkl_path):
	with open(pkl_path, 'rb') as f:
		obj = pickle.load(f)
	return obj

def load_centroids():
	centroids_dict = {}
	for composer in ["bach", "beethoven", "haydn", "mozart", "chopin", "alkan"]:
		final_centroid_dir = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/final_centroids/{composer}"

		final_centroid_path = os.path.join(final_centroid_dir, "final_centroid.txt")
		final_centroid = np.loadtxt(final_centroid_path)
		print(f'Loaded: {final_centroid_path}')

		final_centroid_idx_node_mapping_path = os.path.join(final_centroid_dir, "final_idx_node_mapping.txt")
		with open(final_centroid_idx_node_mapping_path, 'r') as file:
			idx_node_mapping = json.load(file)
			idx_node_mapping = {int(k): v for k, v in idx_node_mapping.items()}
		print(f'Loaded: {final_centroid_idx_node_mapping_path}')

		# we just use the entire node metadata dict from approx_centroid_dir
		approx_centroid_dir = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/approx_centroids/{composer}"
		approx_centroid_node_metadata_dict_path = os.path.join(approx_centroid_dir, "node_metadata_dict.txt")
		with open(approx_centroid_node_metadata_dict_path, 'r') as file:
			node_metadata_dict = json.load(file)
		print(f'Loaded: {approx_centroid_node_metadata_dict_path}')
		
		centroids_dict[composer] = simanneal_centroid_helpers.adj_matrix_to_graph(final_centroid, idx_node_mapping, node_metadata_dict)
	
	return centroids_dict

def is_subgraph_of(subgraph, centroid):
	node_match = lambda n1, n2: n1['label'] == n2['label']
	edge_match = lambda e1, e2: set(e1['label'].split('^')) == set(e2['label'].split('^'))

	return rx.is_subgraph_isomorphic(centroid, subgraph,
																	node_matcher=node_match, edge_matcher=edge_match)

# Analyze subgraph ratio for the given centroid
def analyze_targets(centroid, frequent_subgraphs):
	num_subgraphs = 0
	for k, graph in enumerate(frequent_subgraphs):
		is_subgraph = is_subgraph_of(graph, centroid)
		if is_subgraph:
			num_subgraphs += 1
		# print(f"Graph {k}:" is_subgraph)

		# if not is_subgraph:
		# 	print("GED:", optimized_ged_to_subgraph(graph, centroid))
		# print()
	
	return num_subgraphs / len(frequent_subgraphs)

# so it's consistent with the augmented graph and instance labels only are about the layer type
def fix_labeling(graph):
	graph = graph.to_undirected()
	for node in graph.nodes:
		if node.startswith("Pr"):
			graph.nodes[node]['label'] = node
		else:
			graph.nodes[node]['label'] = node[0]
	for u, v, data in graph.edges(data=True):
		source_label = graph.nodes[u]['label']
		sink_label = graph.nodes[v]['label']
		data['label'] = f"{source_label}^{sink_label}"
	return graph

def hash_subgraph(subgraph):
	# use subgraph vertices for structural info as well as the edge label in the hash. together this determines equality (note assumes already isomorphic)
	edge_hash_info = frozenset(
		(min(u, v), max(u, v), "^".join(sorted(edge_data['label'].split('^'))))
		for u, v, edge_data in subgraph.weighted_edge_list()
	)

	# Efficient binary encoding of edge labels
	edge_hash_info_bytes = b"".join(
		(u.to_bytes(4, "little") + v.to_bytes(4, "little") + label.encode())  # Convert label to bytes
		for u, v, label in sorted(edge_hash_info)
	)

	return hashlib.sha256(edge_hash_info_bytes).hexdigest()


def get_frequent_k_subgraphs(graph_list, k, support_threshold=0.7):
	"""
	Finds subgraphs of size k that appear in at least `support_threshold` proportion of graphs.
	
	Args:
			graph_list (list of rx.PyGraph): List of input graphs.
			k (int): The size of subgraphs to extract.
			support_threshold (float): Minimum proportion of graphs a subgraph must appear in.
	
	Returns:
			list of rx.PyGraph: The frequent subgraphs of size k.
	"""
	subgraph_occurrences = defaultdict(set) # Track which graphs a subgraph appears in
	subgraph_instances = {} # Maps hashes to PyGraph subgraphs

	num_graphs = len(graph_list)
	min_support = int(support_threshold * num_graphs) # Convert threshold to count

	for graph_idx, graph in enumerate(graph_list):
		graph = rx.networkx_converter(fix_labeling(graph).to_undirected(), keep_attributes=True)
		k_subgraphs = rx.connected_subgraphs(graph, k)
		print(len(k_subgraphs))
		seen_hashes = set() # Track seen subgraphs in this graph instance

		for node_indices in k_subgraphs:
			subgraph = graph.subgraph(node_indices) # Extract once, not twice
			hash_key = hash_subgraph(subgraph) # Pass subgraph directly

			if hash_key not in seen_hashes:
				seen_hashes.add(hash_key)
				subgraph_occurrences[hash_key].add(graph_idx)  # Track occurrences
				subgraph_instances[hash_key] = subgraph  # Store actual subgraph

	print(f"Total unique hashes found: {len(subgraph_instances.keys())}")
	print(f"Subgraphs meeting support: {sum(len(v) >= min_support for v in subgraph_occurrences.values())}")
	
	# Filter subgraphs that appear in at least `min_support` graphs
	return [
		subgraph_instances[hash_key]
		for hash_key, graph_indices in subgraph_occurrences.items()
		if len(graph_indices) >= min_support
	]

## HELPER FUNCTION NOT USING NEED TO TEST
def rustworkx_to_networkx(rx_graph):
	nx_graph = nx.Graph()
	for node, node_info in zip(rx_graph.node_indices(), rx_graph.nodes()):
		nx_graph.add_node(node, **node_info)
	for edge, edge_info in zip(rx_graph.edge_indices(), rx_graph.edges()):
		u, v = edge
		nx_graph.add_edge(u, v, **edge_info)
	return nx_graph

def visualize_rustworkx_graphs(centroid, graph_list):
	for i, graph in enumerate(graph_list):
		if is_subgraph_of(graph, centroid):
			mpl_draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10, labels=lambda node: node['label'])
			plt.title(f"Graph {i+1}/{len(graph_list)}")
			plt.show()
	
def find_k_subgraphs_with_support(k, support, save_dir, corpus):
	frequent_subgraphs = get_frequent_k_subgraphs(corpus, k, support_threshold=support)
	os.makedirs(save_dir, exist_ok=True)
	with open(filename, "wb") as f:
		pickle.dump(frequent_subgraphs, f)
		print("SAVED", filename)

def analyze_k_frequent_subgraphs_with_support(subgraphs_filename, centroid):
	frequent_subgraphs = load_pickle(subgraphs_filename)
	centroid = rx.networkx_converter(fix_labeling(centroid).to_undirected(), keep_attributes=True)
	# visualize_rustworkx_graphs(centroid, frequent_subgraphs)
	print("NUM SUBGRAPHS", len(frequent_subgraphs))
	print(analyze_targets(centroid, frequent_subgraphs))

if __name__ == "__main__":
	centroid_path = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/final_centroids/final_centroid"
	
	composer_centroids_dict = load_centroids()
	composer_centroids_dict = {k: composer_centroids_dict[k] for k in sorted(composer_centroids_dict)} # to ensure deterministic order

	composer_corpora_dict = load_corpora()

	args = []
	for i, composer in enumerate(composer_centroids_dict.keys()):
		if ANALYZE_DERIVED_CENTROID:
			device = torch.device(f'cuda:{i}' if torch.cuda.is_available() else 'cpu')
			STG_augmented_list = [composer_centroids_dict[composer]] + composer_corpora_dict[composer]
			listA_G, idx_node_mapping, node_metadata_dict = simanneal_centroid_helpers.pad_adj_matrices(STG_augmented_list)
			listA_G_tensors = [torch.tensor(matrix, device=device, dtype=torch.float64) for matrix in listA_G]
			final_centroid, listA_G_tensors = listA_G_tensors[0], listA_G_tensors[1:]
			alignments, loss = simanneal_centroid.get_alignments_to_centroid(final_centroid, listA_G_tensors, idx_node_mapping, node_metadata_dict, device)
			print(f"FINAL CENTROID LOSS FOR COMPOSER CORPUS {composer}: {loss}")
			continue

		k = 5
		support = 1.0
		save_dir = f'{composer}_{k}-subgraphs'
		filename = f"{save_dir}/common_subgraphs_{k}_support_{support}.pkl"
		
		# frequent_subgraphs = find_k_subgraphs_with_support(k, support, save_dir, composer_corpora_dict[composer])
		analyze_k_frequent_subgraphs_with_support(filename, composer_centroids_dict[composer])
		


# support = 1, k = 5. num subgraphs then percent for each composer corpus
# 1804
# 0.766629711751663
# 1327
# 0.3654860587792012
# 132
# 0.38636363636363635
# 2111
# 0.7006158218853624
# 2504
# 0.6389776357827476
# 1900
# 0.7068421052631579


# support = 1, k = 4
# 0.7688564476885644
# 0.5100864553314121
# 0.4642857142857143
# 0.7735849056603774
# 0.6996047430830039
# 0.7672209026128266