import networkx as nx
import matplotlib.pyplot as plt
import math 
import re
import os
import glob
import json 
import parse_analyses

def create_graph(layers):
	G = nx.DiGraph()

	for layer in layers:
		for node in layer:
			G.add_node(node['id'], start=node['start'], end=node['end'], label=node['label'])

	for i in range(len(layers) - 1):
		for node_a in layers[i]:
			for node_b in layers[i + 1]:
				start_a, end_a = node_a['start'], node_a['end']
				start_b, end_b = node_b['start'], node_b['end']
				# node has edge to parent if its interval overlaps with that parent's interval
				if (start_a <= start_b < end_a) or (start_a < end_b <= end_a):
					G.add_edge(node_a['id'], node_b['id'], label=f"({node_a['label']},{node_b['label']})")

	return G

def vertical_sort_key(layer):
		id = layer[0]['id'] # get the id of the first node in the layer
		if id.startswith('S'): # Prioritize segmentation, and sort by subsegmentation level
			level = int(id.split('L')[1].split('N')[0])
			return (0, level)
		elif id.startswith('P'): # Prioritize pattern/motifs next. , 0 as a placeholder for level since it's irrelevant for motif
			return (1, 0)
		elif id.startswith('FHK'): # Next prioritize functional harmony keys
			return (2, 0)
		elif id.startswith('FHC'): # Next prioritize functional harmony chords
			return (2, 1)
		elif id.startswith('M'): # Finally prioritize melody
			return (3, 0)
		raise Exception("Invalid node encountered in sort", id)

def get_unsorted_layers_from_graph_by_index(G):
	# Regular expressions to match the ID/label formats
	segments_pattern = re.compile(r"^S(\d+)L(\d+)N(\d+)$")
	motives_pattern = re.compile(r"^P(\d+)O(\d+)N(\d+)$")
	fh_keys_pattern = re.compile(r"^FHK(([A-G]|[a-g][+-]?)+)N(\d+)$")
	fh_chords_pattern = re.compile(r"^FHC([1-7\+\-/]+),([1-7\+\-/]+)Q(M|m|d|M7|m7|D7|d7|a|a6|h7)N(\d+)$")
	melody_pattern = re.compile(r"^M(-?\d+)N(\d+)$")

	partition_segments = []
	partition_motives = []
	partition_fh_keys = []
	partition_fh_chords = []
	partition_melody = []
	
	for node, data in G.nodes(data=True):
		if segments_pattern.match(node):
			result = segments_pattern.search(node)
			if result:
				n = result.group(3)
				partition_segments.append({'id': node, 'label': data['label'], 'index': int(n)})
		elif motives_pattern.match(node):
			result = motives_pattern.search(node)
			if result:
				n = result.group(3)
				partition_motives.append({'id': node, 'label': data['label'], 'index': int(n)})
		elif fh_keys_pattern.match(node):
			result = fh_keys_pattern.search(node)
			if result:
				n = result.group(3)
				partition_fh_keys.append({'id': node, 'label': data['label'], 'index': int(n)})
		elif fh_chords_pattern.match(node):
			result = fh_chords_pattern.search(node)
			if result:
				n = result.group(4)
				partition_fh_chords.append({'id': node, 'label': data['label'], 'index': int(n)})
		elif melody_pattern.match(node):
			result = fh_chords_pattern.search(node)
			if result:
				n = result.group(2)
				partition_melody.append({'id': node, 'label': data['label'], 'index': int(n)})
		else:
			raise Exception("Node cannot be classified", node)

	# For the partition_segments list, further partition by the L{n2} substring
	partition_segments_grouped = {}

	for item in partition_segments:
		# Extract the L{n2} part using regular expression
		match = re.search(r'L(\d+)', item['label'])
		if match:
			l_value = match.group(1)
			if l_value not in partition_segments_grouped:
				partition_segments_grouped[l_value] = []
			partition_segments_grouped[l_value].append(item)

	layers = list(partition_segments_grouped.values()) # Convert the grouped dictionary into a list of nested lists
	layers.append(partition_motives)
	layers.append(partition_fh_keys)
	layers.append(partition_fh_chords)
	layers.append(partition_melody)
	layers = [lst for lst in layers if lst]

	return sorted(layers, key=vertical_sort_key)

# this doesn't work properly need to debug
def compress_graph(G):
	instance_labels = {}
	motif_occurrence_counts = {}

	def find_top_level_instance_nodes(G):
		potential_top_levels = set()
		for from_node, to_node in G.edges():
			if 'Pr' in from_node:
				potential_top_levels.add(to_node)
		return potential_top_levels
	
	def get_proto_parents(node_id):
		return [from_node for from_node, _ in G.in_edges(node_id) if 'Pr' in from_node]

	# Recursively assign new labels and find levels
	def assign_labels_and_levels(node, level, index, parent_proto):
		if 'PrS' in parent_proto:
			n = parent_proto.split('PrS')[1]
			new_label = f'S{n}L{level}N{index}'
		elif 'PrP' in parent_proto:
			p_n = parent_proto.split('PrP')[1]
			if p_n not in motif_occurrence_counts:
				motif_occurrence_counts[p_n] = 1
			else:
				motif_occurrence_counts[p_n] += 1
			o_n = motif_occurrence_counts[p_n]
			new_label = f'P{p_n}O{o_n}N{index}'
		else:
			raise ValueError("Unknown prototype parent type")

		instance_labels[node] = new_label
		children = [to_node for _, to_node in G.edges(node)]
		children = sorted(children, key=lambda x: int(x.split('N')[-1]))  # Sort by N value

		# Recursive call for each child
		for i, child in enumerate(children):
			proto_parents = get_proto_parents(child)
			if not proto_parents:
				raise ValueError("No prototype parent found for child")
			assign_labels_and_levels(child, level+1, i, proto_parents[0])

	top_level_instance_nodes = find_top_level_instance_nodes(G)

	# For each root node, determine its level (0) and assign labels
	for index, root_node in enumerate(sorted(top_level_instance_nodes, key=lambda node_label: int(node_label.split('N')[-1]))):
		proto_parents = get_proto_parents(root_node)
		if proto_parents:
			assign_labels_and_levels(root_node, 0, index, proto_parents[0])
		else:
			raise ValueError("Root node without a prototype parent found")

	return instance_labels

# augment with prototype nodes and intra-level layers
def augment_graph(G):
	layers = get_unsorted_layers_from_graph_by_index(G)
	
	# Add prototype nodes and edges to instances
	for layer in layers:
		for node_info in layer:
			if 'L' in node_info['id']:
				proto_node_id = 'PrS' + node_info['id'].split('S')[1].split('L')[0]
			else:
				proto_node_id = 'PrP' + node_info['id'].split('P')[1].split('O')[0]

			if proto_node_id not in G:
				G.add_node(proto_node_id, label=proto_node_id)

			if not G.has_edge(proto_node_id, node_info['id']):
				G.add_edge(proto_node_id, node_info['id'])

	# Add intra-level edges based on index
	for layer in layers:
		# Sort nodes within each layer by their index to ensure proper sequential connections
		sorted_layer_nodes = sorted(layer, key=lambda x: x['index'])
		
		for i in range(len(sorted_layer_nodes)-1):
			current_node_id = sorted_layer_nodes[i]['id']
			next_node_id = sorted_layer_nodes[i+1]['id']
			if not G.has_edge(current_node_id, next_node_id):
				G.add_edge(current_node_id, next_node_id)

def visualize(graph_list, layers_list, labels_dicts = None):
	n = len(graph_list)
	
	# Determine grid size (rows x cols) for subplots
	cols = int(math.ceil(math.sqrt(n)))
	rows = int(math.ceil(n / cols))
	
	# Create a figure with subplots arranged in the calculated grid
	_, axes = plt.subplots(rows, cols, figsize=(10 * cols, 14 * rows))
	
	# Flatten axes array for easy iteration if it's 2D (which happens with multiple rows and columns)
	axes_flat = axes.flatten() if n > 1 else [axes]
	
	for idx, G in enumerate(graph_list):
		layers = layers_list[idx]
		labels_dict = labels_dicts[idx] if labels_dicts else None
		pos = {}  # Positions dictionary: node -> (x, y)
		layer_spacing_factor = 1
		layer_height = 1.0 / (layer_spacing_factor * (len(layers) + 1))
		for i, layer in enumerate(layers):
			y = 1 - (i + 1) * layer_height
			layer = sorted(layer, key=lambda node: node['index'])
					
			x_step = 1.0 / (len(layer) + 1)
			for j, node in enumerate(layer):
				x = (j + 1) * x_step
				pos[node['id']] = (x, y)
		
		ax = axes_flat[idx]
		colors = ["#98FDFF", "#FFB4E4", "#FFDDC1", "#C6E2FF", "#D3FFCE"]
		for layer in layers:
			level = vertical_sort_key(layer)
			color = colors[level[0] % len(colors)]
			nx.draw_networkx_nodes(
					G, pos,
					nodelist=[node['id'] for node in layer],
					node_color=color,
					node_size=1000,
					ax=ax,
					edgecolors='black',
					linewidths=0.5
			)

		# # Draw all nodes except the last layer with the default color
		# non_last_layer_nodes = [node for layer in layers[:-1] for node in layer]
		# nx.draw_networkx_nodes(G, pos, nodelist=[node['id'] for node in non_last_layer_nodes], node_color="#98FDFF", node_size=1000, ax=ax, edgecolors='black', linewidths=0.5)
		
		# # Draw the last layer nodes with yellow color
		# last_layer_nodes = [node['id'] for node in layers[-1]]
		# nx.draw_networkx_nodes(G, pos, nodelist=last_layer_nodes, node_color="#FFB4E4", node_size=1000, ax=ax, edgecolors='black', linewidths=0.5)
		
		# Draw edges and labels for all nodes
		nx.draw_networkx_edges(G, pos, edge_color="black", arrows=True, ax=ax, arrowstyle="-|>,head_length=0.7,head_width=0.5", node_size=1000)
		nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8, ax=ax)
		ax.set_title(f"Graph {idx + 1}")
	
	# Hide any unused subplots in the grid
	for ax in axes_flat[n:]:
			ax.axis('off')
	
	plt.tight_layout()
	plt.show()

def visualize_p(graph_list, layers_list, labels_dicts=None):
	n = len(graph_list)
	
	# Determine grid size (rows x cols) for subplots
	cols = int(math.ceil(math.sqrt(n)))
	rows = int(math.ceil(n / cols))
	
	# Create a figure with subplots arranged in the calculated grid
	_, axes = plt.subplots(rows, cols, figsize=(8 * cols, 12 * rows))
	
	# Flatten axes array for easy iteration if it's 2D (which happens with multiple rows and columns)
	axes_flat = axes.flatten() if n > 1 else [axes]
	
	for idx, G in enumerate(graph_list):
		layers = layers_list[idx]
		labels_dict = labels_dicts[idx] if labels_dicts else None
		pos = {}  # Positions dictionary: node -> (x, y)
		prototype_nodes = []
		
		# Prototype node positioning
		prototype_list = [node for node in G.nodes() if not bool(re.search(r'N\d+$', node))]
		
		# Custom order: "S" prototypes first, then "P", both sorted numerically within their groups
		def proto_sort(proto):
			order = {'PrS': 0, 'PrP': 1}  # Define custom order for the first characters
			return (order[proto[:3]], int(proto[3:]))  # Sort by custom order and then numerically
		prototype_list_sorted = sorted(prototype_list, key=proto_sort)
		
		# Spacing out prototype nodes vertically
		proto_y_step = 1.0 / (len(prototype_list_sorted) + 1)
		for index, prototype in enumerate(prototype_list_sorted):
			y = 1 - (index + 1) * proto_y_step  # Adjust y-coordinate
			if y == 0.5:
				y = y + 0.05
			pos[prototype] = (0.05, y)  # Slightly to the right to avoid touching the plot border
			prototype_nodes.append(prototype)

		layer_height = 1.0 / (len(layers) + 1)
		for i, layer in enumerate(layers):
			layer = sorted(layer, key=lambda node: node['index'])
			y = 1 - (i + 1) * layer_height
			x_step = 1.0 / (len(layer) + 1)
			for j, node in enumerate(layer):
				x = (j + 1) * x_step + 0.1  # Adjust x to the right to accommodate prototypes
				pos[node['id']] = (x, y)
		
		ax = axes_flat[idx]
		
		# segmentation nodes one color
		non_last_layer_nodes = [node for layer in layers[:-1] for node in layer]
		nx.draw_networkx_nodes(G, pos, nodelist=[node['id'] for node in non_last_layer_nodes], node_color="#98FDFF", node_size=1000, ax=ax, edgecolors='black', linewidths=0.5)
		
		# motif nodes new color
		last_layer_nodes = [node['id'] for node in layers[-1]]
		nx.draw_networkx_nodes(G, pos, nodelist=last_layer_nodes, node_color="#FFB4E4", node_size=1000, ax=ax, edgecolors='black', linewidths=0.5)
		nx.draw_networkx_nodes(G, pos, nodelist=prototype_nodes, node_color="#F8FF7D", node_size=1000, ax=ax, edgecolors='black', linewidths=0.5)

		all_edges = set(G.edges())
		intra_level_edges = []
		inter_level_edges = []
		proto_edges = []

		def extract_level(node_id):
			match = re.search(r'S\d+L(\d+)N\d+', node_id)
			return match.group(1) if match else None

		for u, v in all_edges:
			if u in prototype_nodes or v in prototype_nodes:
				proto_edges.append((u, v))
			elif extract_level(u) == extract_level(v):
				intra_level_edges.append((u, v))
			else:
				inter_level_edges.append((u, v))

		nx.draw_networkx_edges(G, pos, edgelist=proto_edges, ax=ax, edge_color="red", arrows=True, arrowstyle="-|>,head_length=0.7,head_width=0.5", node_size=1000)
		nx.draw_networkx_edges(G, pos, edgelist=intra_level_edges, ax=ax, edge_color="#09EF01", arrows=True, arrowstyle="-|>,head_length=0.7,head_width=0.5", node_size=1000)
		nx.draw_networkx_edges(G, pos, edgelist=inter_level_edges, ax=ax, edge_color="black", arrows=True, arrowstyle="-|>,head_length=0.7,head_width=0.5", node_size=1000)
		nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8, ax=ax)
		ax.set_title(f"Graph {idx + 1}")
	
	for ax in axes_flat[n:]:
		ax.axis('off')
	
	plt.tight_layout()
	plt.show()

# segment_identifier should be _{boundary algorithm name}_{labeling algorithm name}_segments.txt
def generate_augmented_graphs_from_dir(dirname, segment_identifier = "segments.txt", motif_identifier = "motives.txt"):
	augmented_graphs = [] 
	for f1 in os.listdir(dirname):
		f1_path = os.path.join(dirname, f1)
		if os.path.isdir(f1_path):
			for f2 in os.listdir(f1_path):
				f2_path = os.path.join(f1_path, f2)
				if os.path.isdir(f2_path):
					graph_filename = os.path.join(f2_path, "augmented_graph.edgelist")
					layers_filename = os.path.join(f2_path, "layers_with_index.json")

					# Load existing graph and layers if they exist
					if os.path.exists(graph_filename) and os.path.exists(layers_filename):
						G = nx.read_edgelist(graph_filename)
						with open(layers_filename, 'r') as file:
							layers_with_index = json.load(file)
						augmented_graphs.append((G, layers_with_index))
						print(f"Loaded graph and layers from {f2_path}")
						continue

					# Else, proceed to parse and generate graph
					segments_file = next(glob.iglob(f2_path + f'/*{segment_identifier}.txt'), None)
					motives_file = next(glob.iglob(f2_path + f'/*{motif_identifier}.txt'), None)

					if segments_file and motives_file:
						layers = parse_analyses.parse_form_file(segments_file)
						layers.append(parse_analyses.parse_motives_file(motives_file))

						G = create_graph(layers)
						augment_graph(G)

						layers_with_index = get_unsorted_layers_from_graph_by_index(G)
						augmented_graphs.append((G, layers_with_index))

						nx.write_edgelist(G, graph_filename)
						with open(layers_filename, 'w') as file:
							json.dump(layers_with_index, file)
						print(f"Graph and layers saved in {f2_path}")
					else:
						print(f"{f2_path} does not have complete analysis files")

	return augmented_graphs

def generate_graph(segments_filepath, motives_filepath, harmony_filepath, melody_filepath):
	layers = parse_analyses.parse_form_file(segments_filepath)
	layers.append(parse_analyses.parse_motives_file(motives_filepath))
	layers.extend(parse_analyses.parse_harmony_file(harmony_filepath))
	# layers.append(parse_analyses.parse_melody_file(melody_filepath))
	G = create_graph(layers)
	layers_with_index = get_unsorted_layers_from_graph_by_index(G)
	return (G, layers_with_index)

if __name__ == "__main__":
	base_path = '/Users/ilanashapiro/Documents/constraints_project/project/datasets/chopin/classical_piano_midi_db/chpn-p7/chpn-p7'
	segments_file = base_path + '_scluster_scluster_segments.txt'
	motives_file = base_path + '_motives1.txt'
	harmony_file = base_path + '_functional_harmony.txt'
	melody_file = base_path + '_vamp_mtg-melodia_melodia_melody_intervals.csv'
	G, layers = generate_graph(segments_file, motives_file, harmony_file, melody_file)
	visualize([G], [layers])
	# augment_graph(G)
	# visualize_p([G], [layers])