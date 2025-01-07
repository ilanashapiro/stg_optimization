import os, sys, pickle, json, re
import math
import numpy as np
import networkx as nx
from multiprocessing import Pool
from networkx.algorithms import isomorphism

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
				node_label = node
				if not node.startswith('Pr'): 
					node_label = node[0] # remove compressed info from instance nodes
				f.write(f"v {node} {node_label}\n")
			
			# Write edges
			for u, v, data in graph.edges(data=True):
				# edge_label = data.get("label", 0)  # Default label = 0 if not provided
				u_label = u
				if not u.startswith('Pr'): 
					u_label = u[0] # remove compressed info from instance nodes
				v_label = v
				if not v.startswith('Pr'): 
					v_label = v[0] # remove compressed info from instance nodes
				edge_label = f'{u_label}^{v_label}'
				f.write(f"e {u} {v} {edge_label}\n")
		
		# End of dataset
		f.write("t # -1\n")

def parse_graph_file(file_path):
	graphs = []
	current_graph = None
	support = None

	with open(file_path, "r") as f:
		for line in f:
			line = line.strip()

			if line.startswith("t #"):
				# Start a new graph
				if current_graph is not None:
					# Append the completed graph to the list
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
		if current_graph is not None:
			graphs.append((current_graph, support))

	return graphs

def is_subgraph_of(centroid, subgraph):
	node_match = lambda n1, n2: n1['label'] == n2['label']
	edge_match = lambda e1, e2: e1['label'] == e2['label']
	
	matcher = isomorphism.GraphMatcher(centroid, subgraph,
																			node_match=node_match, edge_match=edge_match)
	return matcher.subgraph_is_isomorphic()

def parse_graph_file(file_path):
	graphs = []
	current_graph = None
	support = None

	with open(file_path, "r") as f:
		for line in f:
			line = line.strip()

			if line.startswith("t #"):
				# Start a new graph
				if current_graph is not None:
					# Append the completed graph to the list
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
		if current_graph is not None:
			graphs.append((current_graph, support))

	return graphs

def run_gspan(database_file_name, min_support):
	gs = gSpan(
				database_file_name=database_file_name,
				min_support=min_support,
				min_num_vertices=2 # we don't want trivial (i.e. 1-vertex) graphs
		)

	gs.run()
	gs.time_stats()
	return gs

def process_composer(args):
		composer, graphs = args
		if composer in ["bach", "beethoven", "haydn", "mozart"]:
				print("PROCESSING", composer)
				freq_subgraphs_dir = 'frequent_subgraphs_raw'
				if not os.path.exists(freq_subgraphs_dir):
					os.makedirs(freq_subgraphs_dir)
				output_file = f"{freq_subgraphs_dir}/{composer}_frequent_subgraphs_raw.txt"
				with open(output_file, "w", buffering=1) as f:
					try:
						original_stdout = sys.stdout  # Save a reference to the original stdout
						sys.stdout = f  # Redirect stdout to the file
						support = math.ceil(0.5*len(graphs))
						run_gspan(f"output_gspan_graphs_{composer}.data", support)
					finally:
						sys.stdout = original_stdout  # Restore original stdout

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
			# convert_to_gspan_format(graphs, f"output_gspan_graphs_{composer}.data")
		# save_line_graphs(composer_training_pieces_dict)

		# Prepare data for parallel processing
		# composer_data = list(composer_training_pieces_dict.items())
		# with Pool() as pool:
		# 		pool.map(process_composer, composer_data)

		if composer in ["bach", "beethoven", "haydn", "mozart"]:
			graphs_with_support = parse_graph_file(f'frequent_subgraphs_raw/{composer}_frequent_subgraphs_raw.txt')
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
			print(f"Nodes: {centroid.nodes(data=True)}")
			print(f"Edges: {centroid.edges(data=True)}")
			sys.exit(0)

			for i, (graph, support) in enumerate(graphs_with_support):
					print(f"Graph {i}:")
					print(f"Nodes: {graph.nodes(data=True)}")
					print(f"Edges: {graph.edges(data=True)}")
					print(f"Support: {support}")
					print()
			
		