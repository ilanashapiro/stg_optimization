import os, sys, pickle, json, re
import math
import numpy as np
import networkx as nx
# import rustworkx as rx
from multiprocessing import Pool
from networkx.algorithms import isomorphism
from collections import defaultdict
import argparse

DIRECTORY = "/home/ubuntu/project"
# DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
# DIRECTORY = "/home/ilshapiro/project"
TIME_PARAM = '50s'

sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/centroid/spminer/subgraph_mining")
import simanneal_centroid_helpers, decoder

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

def parse_graph_file(file_path):
	graphs = []
	current_graph = None
	support = None

	with open(file_path, "r") as f:
		for line in f:
			line = line.strip()

			if line.startswith("t #"):
				if current_graph is not None and nx.is_connected(current_graph.to_undirected()): # only append connected graphs or we get redundancy
					graphs.append((current_graph, support))
				current_graph = nx.Graph()  # Change to nx.DiGraph() for directed graphs
				support = None  # Reset support

			elif line.startswith("v"):
				# Add a vertex to the graph
				parts = line.split()
				vertex_id = int(parts[1])
				label = parts[2]
				current_graph.add_node(vertex_id, label=label)

			elif line.startswith("e"):
				# Add an edge to the graph
				parts = line.split()
				src = int(parts[1])
				dest = int(parts[2])
				label = parts[3]
				current_graph.add_edge(src, dest, label=label)

			elif line.startswith("Support:"):
				# Record the support value
				support = int(line.split()[1])

		# Add the last graph to the list
		if current_graph is not None and nx.is_connected(current_graph.to_undirected()):
			graphs.append((current_graph, support))
	return graphs

def is_subgraph_of(subgraph, centroid):
	centroid, subgraph = centroid.to_undirected(), subgraph.to_undirected()
	node_match = lambda n1, n2: n1['label'] == n2['label']
	edge_match = lambda e1, e2: set(e1['label'].split('^')) == set(e2['label'].split('^'))
	
	matcher = isomorphism.GraphMatcher(centroid, subgraph,
																			node_match=node_match, edge_match=edge_match)
	return matcher.subgraph_is_isomorphic()

def relabel_pattern_protos(graph):
	# Step 1: Extract nodes with PrPattern_num:{n} and parse `n` values
	pattern_nodes = [
			(node, int(data["label"].split(":")[1])) 
			for node, data in graph.nodes(data=True) 
			if data["label"].startswith("PrPattern_num:")
	]

	# Step 2: Sort nodes by the `n` values
	pattern_nodes.sort(key=lambda x: x[1])  # Sort by the numeric part of the label

	# Step 3: Map sorted `n` values to consecutive integers
	new_mapping = {node: f"PrPattern_num:{i+1}" for i, (node, _) in enumerate(pattern_nodes)}

	# Step 4: Relabel nodes in the graph
	for node, new_label in new_mapping.items():
			graph.nodes[node]["label"] = new_label
	
	return graph

def compute_ged_to_subgraph(g, G):
		min_ged = float('inf')
		best_subgraph = None

		# Generate candidate subgraphs of G (bounded by g's size)
		subgraphs = [G.subgraph(nodes) for nodes in nx.enumerate_all_cliques(G) if len(nodes) >= len(g.nodes)]
		
		for sub in subgraphs:
				# Compute GED between g and current subgraph
				ged = nx.graph_edit_distance(g, sub)  # Use an approximate method if necessary
				if ged < min_ged:
						min_ged = ged
						best_subgraph = sub

		return min_ged, best_subgraph

import networkx as nx
from networkx.algorithms import approximation

def optimized_ged_to_subgraph(g, G, max_candidates=100):
		best_ged = float('inf')
		best_subgraph = None

		# Heuristic: Find candidate subgraphs using neighborhood sampling
		candidates = []
		for node in G.nodes:
				subgraph = nx.ego_graph(G, node, radius=len(g.nodes) // 2)
				if len(subgraph) >= len(g):
						candidates.append(subgraph)
				if len(candidates) >= max_candidates:
						break

		# Compute approximate GED for each candidate
		for sub in candidates:
				ged = nx.graph_edit_distance(g, sub, timeout=2.0)  # Timeout for approximate GED
				if ged is not None and ged < best_ged:
						best_ged = ged
						best_subgraph = sub

		return best_ged, best_subgraph

def search(composer, centroid):
	min_pattern_size, max_pattern_size, out_batch_size = 1, 5, 7
	search_strategy = "mcts"
	results_fp = f"{DIRECTORY}/experiments/centroid/spminer/results/{composer}/{search_strategy}-out-patterns.p"
	plots_dir = f"{DIRECTORY}/experiments/centroid/spminer/results/{composer}/plots"
        
	args = argparse.Namespace(
			dataset='stg',  # Choose a dataset (this simulates the command line argument)
			node_anchored=True,  # Set flags as needed
			min_pattern_size=min_pattern_size,
			max_pattern_size=max_pattern_size,
			out_batch_size=out_batch_size,
			out_path=results_fp,
			plots_dir=plots_dir,
			search_strategy=search_strategy,
			composer=composer
	)
	# decoder.main(args)
	# sys.exit(0)

	# scores = []
	curr_score = -1
	for i in range(20):
		decoder.main(args)
		with open(results_fp, "rb") as f:
			targets = pickle.load(f) # THIS FILE IS GENERATED BY decoder.main(args)

		for j, graph in enumerate(targets):
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
			targets[j] = graph

		# Group graphs by number of nodes
		node_count_groups = defaultdict(list)
		for graph in targets:
				node_count = graph.number_of_nodes()
				node_count_groups[node_count].append(graph)

		# Analyze subgraph ratio for each group
		results_by_group = {}
		for node_count, graphs in node_count_groups.items():
			print(f"Analyzing group with {node_count} nodes ({len(graphs)} graphs)...")
			
			num_subgraphs = 0
			for k, graph in enumerate(graphs):
				print(f"Graph {k}:")
				# print("Nodes:")
				# for node, data in graph.nodes(data=True):
				# 		print(f"  Node: {node}, Label: {data.get('label', 'No Label')}")

				# # Print edges with only the 'label' attribute
				# print("Edges:")
				# for u, v, data in graph.edges(data=True):
				# 		print(f"  Edge: ({u}, {v}), Label: {data.get('label', 'No Label')}")

				is_subgraph = is_subgraph_of(graph, centroid)
				if is_subgraph:
					num_subgraphs += 1
				print(is_subgraph)
				# if not is_subgraph:
				# 	print("GED:", optimized_ged_to_subgraph(graph, centroid))
				# print()
			
			# Calculate and store the result for this group
			ratio = num_subgraphs / len(graphs)
			results_by_group[node_count] = ratio
			# print(f"RESULT for {node_count}-node group:", ratio)
			# print()
		# continue

		new_score = sum(len(graphs) * results_by_group[node_count] for node_count, graphs in node_count_groups.items()) / len(targets)
		# scores.append(new_score)
		print(f"RESULT for {composer}: {new_score}")
		# print(f"CURRENT OVERALL AVERAGE for {composer}: {np.mean(scores)}")
		
		if new_score > curr_score and new_score < 1 and os.path.exists(results_fp):
			curr_score = new_score
			print("1",results_fp)
			os.rename(results_fp, results_fp[:-2] + '_RESULT.p')
			print("",results_fp)
			print(f"Found better subgraph set and saved")

if __name__ == "__main__":
	centroid_path = f"{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}"
	training_pieces_path = f"{DIRECTORY}/experiments/centroid/clusters/composer_centroid_input_graphs_{TIME_PARAM}.txt"

	composer_centroids_dict = load_centroids()
	composer_centroids_dict = {k: composer_centroids_dict[k] for k in sorted(composer_centroids_dict)} # to ensure deterministic order

	def load_graph(file_path):
			with open(file_path, 'rb') as f:
				return pickle.load(f).to_undirected()

	with open(training_pieces_path, 'r') as file:
		composer_training_pieces_paths = json.load(file)
		composer_training_pieces_dict = {}
		for composer, filepaths in composer_training_pieces_paths.items():
			if composer in ["mozart"]:# ["bach", "beethoven," "haydn", ]:
				graphs = [load_graph(re.sub(r'^.*?/project', DIRECTORY, file_path)) for file_path in filepaths]
				composer_training_pieces_dict[composer] = graphs
				centroid = composer_centroids_dict[composer]

				for node in centroid.nodes:
					if node.startswith("Pr"):
						centroid.nodes[node]['label'] = node
					else:
						centroid.nodes[node]['label'] = node[0]
				for u, v, data in centroid.edges(data=True):
					source_label = centroid.nodes[u]['label']
					sink_label = centroid.nodes[v]['label']
					data['label'] = f"{source_label}^{sink_label}"
				centroid = relabel_pattern_protos(centroid)
				
				# print(f"Centroid:")
				# print("Nodes:")
				# for node, data in centroid.nodes(data=True):
				# 		print(f"  Node: {node}, Label: {data.get('label', 'No Label')}")

				# # Print edges with only the 'label' attribute
				# print("Edges:")
				# for u, v, data in centroid.edges(data=True):
				# 		print(f"  Edge: ({u}, {v}), Label: {data.get('label', 'No Label')}")

				# search(composer, centroid)
				output_file = f"{DIRECTORY}/experiments/centroid/spminer/results/{composer}/out-patterns_{composer}.txt"
				os.makedirs(os.path.dirname(output_file), exist_ok=True)
				with open(output_file, "w", buffering=1) as f:
					try:
						original_stdout = sys.stdout  # Save a reference to the original stdout
						sys.stdout = f  # Redirect stdout to the file
						search(composer, centroid)
					finally:
						sys.stdout = original_stdout 

				



# PLAN
# get rid of PCA -- maybe just do table of euclidean distances between full dim spectral embeddings
# substructure frequency plus qualitative analysis -- find the best/intuitive subgraphs to talk about, try to tell a story
# be careful that substructure frequency isn't music specific, maybe qualitative anlaysis can help bridge this
# then search for downstream applications and just see if i can get any of those in in time
# and consider organizing the structure of the experiments section that pushes people in my favor: 
# intrinsic measure of success of centroid finding, then extrinsic measure of music applications for downstream music tasks, then inspection of centroid
# idk -- but VERY SPECIFICALLY RELATE to the intro, what im tryiing to contribute, music perception etc -- this gives us a useful summary of music inn the following way!
# our centroids are accurate in the following way and useful for this application