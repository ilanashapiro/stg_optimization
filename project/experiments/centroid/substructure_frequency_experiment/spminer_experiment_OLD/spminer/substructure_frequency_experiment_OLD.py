import os, sys, pickle, json, re, shutil, glob
import math

import numpy as np
import networkx as nx
import torch.multiprocessing as tmp
import multiprocessing as mp
from networkx.algorithms import isomorphism
from collections import defaultdict
from itertools import combinations
import argparse
# DIRECTORY = "/home/ubuntu/project"
DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
# DIRECTORY = "/home/ilshapiro/project"

sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/spminer/subgraph_mining")
import simanneal_centroid_helpers#, centroid_simanneal_gen, decoder

def load_corpora():
	corpora_path = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/corpora/composer_centroid_input_graphs.txt"
	with open(corpora_path, 'r') as file:
		composer_corpus_pieces_paths = json.load(file)
	def load_graph(file_path):
		with open(file_path, 'rb') as f:
			return pickle.load(f).to_undirected()
	graphs = []
	composer_corpus_pieces_dict = {}
	for composer, filepaths in composer_corpus_pieces_paths.items():
		graphs = [load_graph(re.sub(r'^.*?/project', DIRECTORY, file_path)) for file_path in filepaths]

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

def load_STG(stg_path):
	with open(stg_path, 'rb') as f:
		graph = pickle.load(f)
	return graph

def load_naive_centroids():
	corpora_path = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/corpora/composer_centroid_input_graphs.txt"
	with open(corpora_path, 'r') as file:
		composer_corpus_pieces = json.load(file)

	naive_centroids_dict = {}
	for composer in ["bach", "beethoven", "haydn", "mozart", "chopin", "alkan"]:
		alignments_dir = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/initial_alignments/{composer}"
		initial_centroid_filepath = glob.glob(os.path.join(alignments_dir, '*initial_centroid*'))[0]
		initial_centroid_filename = os.path.basename(initial_centroid_filepath)[:-4].replace('initial_centroid_', '')
		
		naive_centroid_filepath = next((f for f in composer_corpus_pieces[composer] if initial_centroid_filename in f), None)
		naive_centroid = load_STG(naive_centroid_filepath)
		naive_centroids_dict[composer] = naive_centroid
	
	return naive_centroids_dict


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
	# centroid, subgraph = centroid.to_undirected(), subgraph.to_undirected()
	node_match = lambda n1, n2: n1['label'] == n2['label']
	edge_match = lambda e1, e2: set(e1['label'].split('^')) == set(e2['label'].split('^'))

	return rx.is_subgraph_isomorphic(centroid, subgraph,
																			node_matcher=node_match, edge_matcher=edge_match)
	
	# matcher = isomorphism.GraphMatcher(centroid, subgraph,
	# 																		node_match=node_match, edge_match=edge_match)
	# return matcher.subgraph_is_isomorphic()

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

# Analyze subgraph ratio for the given centroid
def analyze_targets(centroid, frequent_subgraphs):
	num_subgraphs = 0
	for k, graph in enumerate(frequent_subgraphs):
		# print(f"Graph {k}:")

		# print("Nodes:")
		# for node, data in graph.nodes(data=True):
		# 		print(f"  Node: {node}, Label: {data.get('label', 'No Label')}")

		# # Print edges with only the 'label' attribute
		# print("Edges:")
		# for u, v, data in graph.edges(data=True):
		# 		print(f"  Edge: ({u}, {v}), Label: {data.get('label', 'No Label')}")

		# print(graph.edges(), "\n\n----", centroid.edges())
		# print(graph.nodes(), "\n\n----", centroid.nodes())
		# print()
		is_subgraph = is_subgraph_of(graph, centroid)
		if is_subgraph:
			num_subgraphs += 1
		# print(is_subgraph)

		# if not is_subgraph:
		# 	print("GED:", optimized_ged_to_subgraph(graph, centroid))
		# print()
	
	return num_subgraphs / len(frequent_subgraphs)

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

def search(composer, centroid, gpu_id=0):
	min_pattern_size, max_pattern_size, out_batch_size = 2, 7, 7
	search_strategy = "mcts"
	results_dir = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/spminer/results_7/{composer}"
	results_fp = results_dir + f"/{search_strategy}-out-patterns.p"
	plots_dir = results_dir + f"/plots"
				
	args = argparse.Namespace(
			dataset='stg',  # Choose a dataset (this simulates the command line argument)
			node_anchored=True,  # Set flags as needed
			min_pattern_size=min_pattern_size,
			max_pattern_size=max_pattern_size,
			out_batch_size=out_batch_size,
			out_path=results_fp,
			plots_dir=plots_dir,
			search_strategy=search_strategy,
			composer=composer,
			gpu_id=gpu_id
	)
	
	# scores = []
	curr_score = -1
	for i in range(25):
		decoder.main(args)
		targets = load_STG(results_fp) # THIS FILE IS GENERATED BY decoder.main(args)
		
		# print("TARGETS:", targets)

		for j, graph in enumerate(targets):
			graph = fix_labeling(graph)
		targets[j] = graph

		# Group graphs by number of nodes
		node_count_groups = defaultdict(list)
		for graph in targets:
				node_count = graph.number_of_nodes()
				node_count_groups[node_count].append(graph)

		results_by_group = {}
		for node_count, graphs in node_count_groups.items():
			print(f"Analyzing group with {node_count} nodes ({len(graphs)} graphs)...")
			results_by_group[node_count] = analyze_targets(centroid, graphs)
			# print(f"RESULT for {node_count}-node group:", ratio)
			# print()

		new_score = sum(len(graphs) * results_by_group[node_count] for node_count, graphs in node_count_groups.items()) / len(targets)
		# scores.append(new_score)
		print(f"RESULT for {composer}: {new_score}")
		# print(f"CURRENT OVERALL AVERAGE for {composer}: {np.mean(scores)}")
		
		if new_score > curr_score and new_score < 1 and os.path.exists(results_fp) and os.path.exists(plots_dir):
			curr_score = new_score
			os.rename(results_fp, results_fp[:-2] + '_RESULT.p')

			results_plot_dir = plots_dir + '_RESULT'
			if os.path.exists(results_plot_dir): # if we try to rename an existing dir it crashes
				shutil.rmtree(results_plot_dir)
			os.rename(plots_dir, results_plot_dir)
			print(f"Saved subgraph set")
		 
		print("FINISHED ITERATION", i)

def get_layer_id(node):
	for layer_id in ['S', 'P', 'K', 'C', 'M']:
		if node.startswith(layer_id):
			return layer_id
		
def is_instance(node_id):
	return not is_proto(node_id)

def is_proto(node_id):
	return node_id.startswith('Pr')

def partition_prototype_features(idx_node_mapping, node_metadata_dict):
	prototype_features_dict = {}
	for node_id in idx_node_mapping.values():
		if is_proto(node_id):
			feature_name = node_metadata_dict[node_id]['feature_name']
			prototype_features_dict.setdefault(feature_name, []).append(node_id)
	return prototype_features_dict

def proto_to_layer_id(proto_node_id):
	if 'filler' in proto_node_id:
		return proto_node_id[2] # Pr[layer_id]filler
	mapping = {
		"section_num": "S",
		"pattern_num": "P",
		"key": "K",
		"degree": "C",
		"interval": "M"
	}
	for k,v in mapping.items():
		if k.lower() in proto_node_id.lower():
			return v
	return None

## OLD HELPER FUNCTION NOT USING
def partition_stg_nodes(g):
	# key: layer id
	# value: list of node IDs
	node_levels_partitions = {} 
	for node_id in g.nodes():
		if is_instance(node_id):
			node_levels_partitions.setdefault(get_layer_id(node_id), []).append(node_id)
		else:
			layer_id = proto_to_layer_id(node_id) # will be none if proto is "quality." doesn't check for "quality" which is either key or chord layer
			if layer_id: 
				node_levels_partitions.setdefault(layer_id, []).append(node_id)
			elif "quality" in node_id.lower():
				node_levels_partitions.setdefault('C', []).append(node_id)
				node_levels_partitions.setdefault('K', []).append(node_id)
			else:
				print("ERROR ADDING", node_id, " TO PARTITIONS")
	return node_levels_partitions


## OLD HELPER FUNCTION NOT USING
def partition_stg(g):
	stg_partition_subgraphs = []
	node_levels_partitions = partition_stg_nodes(g)
	levels = ['S', 'P', 'K', 'C', 'M']
	for layer1, layer2 in zip(levels, levels[1:]):
		nodes_partition1 = set(node_levels_partitions.get(layer1, []))
		nodes_partition2 = set(node_levels_partitions.get(layer2, []))
		
		# Combine the two sets of nodes to get all relevant nodes for the subgraph
		relevant_nodes = nodes_partition1.union(nodes_partition2)
		
		# Extract edges between the relevant nodes (including edges within and between partitions)
		edges_in_subgraph = [(u, v) for u, v in g.edges(relevant_nodes) if u in relevant_nodes and v in relevant_nodes]
		stg_partition_subgraphs.append(g.edge_subgraph(edges_in_subgraph).copy())
	return stg_partition_subgraphs

def process_composer(composer, composer_centroids_dict, gpu_id, stdout_to_file=True):
	try:
		centroid = composer_centroids_dict[composer]

		# Label nodes and edges
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

		if stdout_to_file:
			stdout_file = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/spminer/results_7/{composer}/out-patterns_{composer}.txt"
			os.makedirs(os.path.dirname(stdout_file), exist_ok=True)
			
			with open(stdout_file, "w", buffering=1) as f:
				original_stdout = sys.stdout
				sys.stdout = f
				try:
					search(composer, centroid, gpu_id)
				finally:
					sys.stdout = original_stdout  # Restore original stdout
		else:
			search(composer, centroid, gpu_id)

		return f"Finished processing {composer}"
	
	except Exception as e:
		return f"Error processing {composer}: {str(e)}"

def visualize_rustworkx_graphs(centroid, graph_list):
		for i, graph in enumerate(graph_list):
				if is_subgraph_of(graph, centroid):
					mpl_draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10, labels=lambda node: node['label'])
					plt.title(f"Graph {i+1}/{len(graph_list)}")
					plt.show()
				# input("Press Enter to continue...")  # Wait for user input before proceeding
	
if __name__ == "__main__":
	centroid_path = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/final_centroids/final_centroid"
	
	composer_centroids_dict = load_centroids()
	composer_centroids_dict = {k: composer_centroids_dict[k] for k in sorted(composer_centroids_dict)} # to ensure deterministic order

	naive_centroids_dict = load_naive_centroids()
	naive_centroids_dict = {k: naive_centroids_dict[k] for k in sorted(naive_centroids_dict)} # to ensure deterministic order

	composer_corpora_dict = load_corpora()

	args = []
	for i, composer in enumerate(composer_centroids_dict.keys()):
		# process_composer(composer, naive_centroids_dict, gpu_id=1, stdout_to_file=True)
		args.append((composer, composer_centroids_dict, i))

	ctx = tmp.get_context("spawn")  # Get a new multiprocessing context
	with ctx.Pool() as pool:
		pool.starmap(process_composer, args)
				

# PLAN
# get rid of PCA -- maybe just do table of euclidean distances between full dim spectral embeddings
# substructure frequency plus qualitative analysis -- find the best/intuitive subgraphs to talk about, try to tell a story
# be careful that substructure frequency isn't music specific, maybe qualitative anlaysis can help bridge this
# then search for downstream applications and just see if i can get any of those in in time
# and consider organizing the structure of the experiments section that pushes people in my favor: 
# intrinsic measure of success of centroid finding, then extrinsic measure of music applications for downstream music tasks, then inspection of centroid
# idk -- but VERY SPECIFICALLY RELATE to the intro, what im tryiing to contribute, music perception etc -- this gives us a useful summary of music inn the following way!
# our centroids are accurate in the following way and useful for this application