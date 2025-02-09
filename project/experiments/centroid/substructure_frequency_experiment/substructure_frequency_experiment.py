import os, sys, pickle, json, re
import math
import numpy as np
import networkx as nx
# import rustworkx as rx
import torch.multiprocessing as mp
from networkx.algorithms import isomorphism
from collections import defaultdict
import argparse

DIRECTORY = "/home/ubuntu/project"
# DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
# DIRECTORY = "/home/ilshapiro/project"

sys.path.append(f"{DIRECTORY}/centroid")
sys.path.append(f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/spminer/subgraph_mining")
import simanneal_centroid_helpers, decoder

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

def search(composer, centroid, gpu_id=0):
	min_pattern_size, max_pattern_size, out_batch_size = 2, 5, 7
	search_strategy = "mcts"
	results_fp = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/spminer/results/{composer}/{search_strategy}-out-patterns.p"
	plots_dir = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/spminer/results/{composer}/plots"
				
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
		with open(results_fp, "rb") as f:
			targets = pickle.load(f) # THIS FILE IS GENERATED BY decoder.main(args)
		
		print("TARGETS:", targets)
		
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
			os.rename(results_fp, results_fp[:-2] + '_RESULT.p')
			os.rename(plots_dir, plots_dir[:-2] + '_RESULT.p')
			print(f"Saved subgraph set")

def process_composer(composer, composer_centroids_dict, gpu_id):
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

		# Define output path
		output_file = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/spminer/results/{composer}/out-patterns_{composer}.txt"
		os.makedirs(os.path.dirname(output_file), exist_ok=True)

		# Redirect stdout and run search
		with open(output_file, "w", buffering=1) as f:
			original_stdout = sys.stdout
			sys.stdout = f
			try:
				search(composer, centroid, gpu_id)
			finally:
				sys.stdout = original_stdout  # Restore original stdout

		return f"Finished processing {composer}"
	
	except Exception as e:
		return f"Error processing {composer}: {str(e)}"
	
if __name__ == "__main__":
	centroid_path = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/final_centroids/final_centroid"
	training_pieces_path = f"{DIRECTORY}/experiments/centroid/substructure_frequency_experiment/corpora/composer_centroid_input_graphs.txt"

	composer_centroids_dict = load_centroids()
	composer_centroids_dict = {k: composer_centroids_dict[k] for k in sorted(composer_centroids_dict)} # to ensure deterministic order

	with open(training_pieces_path, 'r') as file:
		composer_training_pieces_paths = json.load(file)

		args = []
		for i, (composer, filepaths) in enumerate(composer_training_pieces_paths.items()):
			# if composer in ["bach", "beethoven", "haydn", "mozart"]:
				# process_composer(composer, composer_centroids_dict, i)
			args.append((composer, composer_centroids_dict, i))
	
		ctx = mp.get_context("spawn")  # Get a new multiprocessing context
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