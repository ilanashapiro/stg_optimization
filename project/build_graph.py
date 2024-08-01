from tracemalloc import start
import networkx as nx
import matplotlib.pyplot as plt
import math 
import re
import os
import glob
import json 
import parse_analyses
import pandas as pd
from analyses import format_conversions as fc
import mido
import sys, pickle
from multiprocessing import Pool
from collections import defaultdict, deque

def get_layer_id(node):
	for layer_id in ['S', 'P', 'K', 'C', 'M']:
		if node.startswith(layer_id):
			return layer_id
	raise Exception("Invalid node", node)

def create_graph(piece_start_time, piece_end_time, layers):
	G = nx.DiGraph()
	
	for layer in layers:
		sorted_nodes = sorted(layer, key=lambda x: x['start']) # Sort nodes in the current layer by their start time
		
		for idx, node in enumerate(sorted_nodes):
			# account for rounding errors
			tolerance = 0.001
			if abs(node['start'] - piece_start_time) <= tolerance:
				node['start'] = piece_start_time
			if abs(node['end'] - piece_end_time) <= tolerance:
					node['end'] = piece_end_time
			
			# account for audio/symbolic sync issues for segmentation timestamps
			if get_layer_id(node['id']) == 'S':
				if idx == 0 and node['start'] > piece_start_time:
					node['start'] = piece_start_time
				elif idx == len(sorted_nodes) - 1 and node['end'] < piece_end_time:
					node['end'] = piece_end_time
			
			# Add all real nodes to the graph
			if node['index'] < 1:
				raise Exception("Node index < 1:", node)
			if node['start'] >= piece_start_time and node['end'] <= piece_end_time:
				G.add_node(node['id'], start=node['start'], end=node['end'], label=node['label'], index=node['index'], features_dict=node['features_dict'])
				
		# Add all filler nodes
		for i in range(len(sorted_nodes) - 1):
			node1 = sorted_nodes[i]
			node2 = sorted_nodes[i + 1]
			
			# There's a gap between node1 and node2
			if node1['end'] < node2['start'] and node2['start'] <= piece_end_time and node1['end'] >= piece_start_time: 
				filler_node_index = node1['index'] + 0.5
				filler_node_id = f"{get_layer_id(node1['id'])}fillerN{filler_node_index}" # Hardcoding P for now since motif/pattern layer is the only one that requires fillers
				filler_node_label = filler_node_id.split('N')[0]
				G.add_node(filler_node_id, start=node1['end'], end=node2['start'], label=filler_node_label, index=filler_node_index, features_dict={})
				filler_node = {'id': filler_node_id, 'start': node1['end'], 'end': node2['start'], 'label': filler_node_label, 'index': filler_node_index, 'features_dict': {}}
				layer.append(filler_node)
				
		# if the first node in the layer starts after the piece start time, we have a gap at the beginning
		first_node = sorted_nodes[0]
		if first_node['start'] > piece_start_time:
			filler_node_id = f"{get_layer_id(first_node['id'])}fillerN{0.5}"
			filler_node_label = filler_node_id.split('N')[0]
			G.add_node(filler_node_id, start=piece_start_time, end=first_node['start'], label=filler_node_label, index=0.5, features_dict={})
			filler_node = {'id': filler_node_id, 'start': piece_start_time, 'end': first_node['start'], 'label': filler_node_label, 'index': 0.5, 'features_dict': {}}
			layer.append(filler_node)

		# if the layer node in the layer starts after the piece end time, we have a gap at the end
		last_node = sorted_nodes[-1]
		if last_node['end'] < piece_end_time:
			filler_node_index = last_node['index'] + 0.5
			filler_node_id = f"{get_layer_id(last_node['id'])}fillerN{filler_node_index}"
			filler_node_label = filler_node_id.split('N')[0]
			G.add_node(filler_node_id, start=last_node['end'], end=piece_end_time, label=filler_node_label, index=filler_node_index, features_dict={})
			filler_node = {'id': filler_node_id, 'start': last_node['end'], 'end': piece_end_time, 'label': filler_node_label, 'index': filler_node_index, 'features_dict': {}}
			layer.append(filler_node)
		layer.sort(key=lambda x: x['start'])
			
	for i in range(len(layers) - 1):
		for node_a in layers[i]:
			for node_b in layers[i + 1]:
				start_a, end_a = node_a['start'], node_a['end']
				start_b, end_b = node_b['start'], node_b['end']
				# node has edge to parent if its interval overlaps with that parent's interval
				if (start_a <= start_b < end_a) or (start_a < end_b <= end_a):
					G.add_edge(node_a['id'], node_b['id'], label=f"({node_a['label']},{node_b['label']})")

	# Remove unused filler nodes
	unused_filler_nodes = [node for node in G.nodes() if "filler" in node and G.out_degree(node) == 0]
	G.remove_nodes_from(unused_filler_nodes)
	return G

def vertical_sort_key(layer):
	node_id = layer[0]['id'] # get the id of the first node in the layer
	return get_layer_rank(node_id)

def get_layer_rank(node):
	if node.startswith('S'): # Prioritize segmentation, and sort by subsegmentation level
		level = int(node.split('L')[1].split('N')[0]) - 1 # zero-indexing
		return (0, level)
	elif node.startswith('P'): # Prioritize pattern/motifs next. , 0 as a placeholder for level since it's irrelevant for motif
		return (1, 0)
	elif node.startswith('K'): # Next prioritize functional harmony keys
		return (2, 0)
	elif node.startswith('C'): # Next prioritize functional harmony chords
		return (3, 0)
	elif node.startswith('M'): # Finally prioritize melody
		return (4, 0)
	raise Exception("Invalid node encountered in sort", node)

def get_unsorted_layers_from_graph_by_index(G):
	partition_segments = []
	partition_motives = []
	partition_keys = []
	partition_chords = []
	partition_melody = []

	partition_map = {
		'S': partition_segments,
		'P': partition_motives,
		'K': partition_keys,
		'C': partition_chords,
		'M': partition_melody,
	}
	
	for node, data in G.nodes(data=True):
		if node.startswith('Pr'): # don't include protos in the partition
			continue
		index = int(data['index'])
		label = data['label']
		features_dict = data['features_dict']
		
		for prefix, partition in partition_map.items():
			if node.startswith(prefix):
				partition.append({'id': node, 'label': label, 'features_dict': features_dict, 'index': index})
				break

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
	layers.append(partition_keys)
	layers.append(partition_chords)
	layers.append(partition_melody)
	layers = [lst for lst in layers if lst]

	return sorted(layers, key=vertical_sort_key)

# augment with prototype nodes and intra-level layers
def augment_graph(G):
	layers = get_unsorted_layers_from_graph_by_index(G)
	
	# Add prototype nodes and edges to instances
	for layer in layers:
		for node in layer:
			instance_node_id = node['id']
			features_dict = node['features_dict']
			proto_nodes = []

			for feature_name, value in features_dict.items():
				proto_node_id = f"Pr{feature_name.capitalize()}:{value}"
				proto_nodes.append((proto_node_id, feature_name, get_layer_id(instance_node_id)))
			
			if not bool(features_dict):
				source_layer_kind = get_layer_id(instance_node_id)
				feature_name = get_layer_id(instance_node_id) + "filler"
				proto_nodes.append((f"Pr{feature_name}:{value}", feature_name, source_layer_kind))
			
			for (proto_node_id, feature_name, source_layer_kind) in proto_nodes:
				if proto_node_id not in G:
					G.add_node(proto_node_id, label=proto_node_id, layer_rank=get_layer_rank(instance_node_id), feature_name=feature_name, source_layer_kind=source_layer_kind)
				if not G.has_edge(proto_node_id, instance_node_id):
					G.add_edge(proto_node_id, instance_node_id)

	# Add intra-level edges based on index
	for layer in layers:
		# Sort nodes within each layer by their index to ensure proper sequential connections
		sorted_layer_nodes = sorted(layer, key=lambda x: x['index'])
		
		for i in range(len(sorted_layer_nodes)-1):
			current_node_id = sorted_layer_nodes[i]['id']
			next_node_id = sorted_layer_nodes[i+1]['id']
			if not G.has_edge(current_node_id, next_node_id):
				G.add_edge(current_node_id, next_node_id)

def visualize(graph_list, layers_list):
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
		labels_dict = {node: data.get('label', node) for node, data in G.nodes(data=True)} # Extract node labels from node attributes

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
		colors = ["#B797FF", "#fd7373", "#ffda69", "#99d060", "#99e4ff"]
		filler_color = '#FF0000'
		for layer in layers:
			level = vertical_sort_key(layer)
			color = colors[level[0] % len(colors)]
			for node in layer:
				node_color = filler_color if "filler" in node['id'] else color
				nx.draw_networkx_nodes(
						G, pos,
						nodelist=[node['id']],
						node_color=node_color,
						node_size=1000,
						ax=ax,
						edgecolors='black',
						linewidths=0.5
				)

		# Draw edges and labels for all nodes
		nx.draw_networkx_edges(G, pos, edge_color="black", arrows=True, ax=ax, arrowstyle="-|>,head_length=0.7,head_width=0.5", node_size=1000)
		nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8, ax=ax)
		ax.set_title(f"Graph {idx + 1}")
	
	# Hide any unused subplots in the grid
	for ax in axes_flat[n:]:
			ax.axis('off')
	
	plt.tight_layout()
	plt.show()

def visualize_p(graph_list, layers_list):
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
		labels_dict = {node: data.get('label', node) for node, data in G.nodes(data=True)} # Extract node labels from node attributes
		pos = {}  # Positions dictionary: node -> (x, y)
		prototype_nodes = []
		
		# Prototype node positioning
		prototype_list = [{'id': node, 'layer_rank': data['layer_rank']} for node, data in G.nodes(data=True) if node.startswith("Pr")]
		def proto_sort(proto_node):
			return (proto_node['layer_rank'], proto_node['id']) # primary and secondary sort
		prototype_list_sorted = [d['id'] for d in sorted(prototype_list, key=proto_sort)]

		# Spacing out prototype nodes vertically
		proto_y_step = 1.0 / (len(prototype_list_sorted) + 1)
		for index, prototype in enumerate(prototype_list_sorted):
			y = 1 - (index + 1) * proto_y_step  # Adjust y-coordinate
			pos[prototype] = (0.05, y)  # Slightly to the right to avoid touching the plot border
			prototype_nodes.append(prototype)
		
		ax = axes_flat[idx]
		all_edges = set(G.edges())
		intra_level_edges = []
		inter_level_edges = []
		proto_edges = []

		for u, v in all_edges:
			if u in prototype_nodes or v in prototype_nodes:
				proto_edges.append((u, v))
			elif get_layer_rank(u) == get_layer_rank(v):
				intra_level_edges.append((u, v))
			else:
				inter_level_edges.append((u, v))
		
		def topological_sort(nodes, edges):
			graph = defaultdict(list)
			in_degree = {node: 0 for node in nodes}
			for u, v in edges:
					graph[u].append(v)
					in_degree[v] += 1
			queue = deque([node for node in nodes if in_degree[node] == 0])
			sorted_nodes = []
			while queue:
					node = queue.popleft()
					sorted_nodes.append(node)
					for neighbor in graph[node]:
							in_degree[neighbor] -= 1
							if in_degree[neighbor] == 0:
									queue.append(neighbor)
			return sorted_nodes

		def get_layer_nodes_and_edges(layer, edges):
			"""Return nodes and intra-level edges for a specific layer."""
			layer_nodes = [node['id'] for node in layer]
			layer_edges = [(u, v) for u, v in edges if u in layer_nodes and v in layer_nodes]
			return layer_nodes, layer_edges

		layer_height = 1.0 / (len(layers) + 1)
		for i, layer in enumerate(layers):
			if nx.is_directed_acyclic_graph(G):
				layer_nodes, layer_edges = get_layer_nodes_and_edges(layer, intra_level_edges)
				sorted_node_ids = topological_sort(layer_nodes, layer_edges)
				layer = [next(node for node in layer if node['id'] == node_id) for node_id in sorted_node_ids]
			else:
				layer = sorted(layer, key=lambda node: node['index'])
			y = 1 - (i + 1) * layer_height
			x_step = 1.0 / (len(layer) + 1)
			for j, node in enumerate(layer):
				x = (j + 1) * x_step + 0.1  # Adjust x to the right to accommodate prototypes
				pos[node['id']] = (x, y)
				
		colors = ["#B797FF", "#fd7373", "#ffda69", "#99d060", "#99e4ff"]
		filler_color = '#808080'
		for layer in layers:
			level = vertical_sort_key(layer)
			color = colors[level[0] % len(colors)]
			for node in layer:
				node_color = filler_color if "filler" in node['id'] else color
				nx.draw_networkx_nodes(
						G, pos,
						nodelist=[node['id']],
						node_color=node_color,
						node_size=1000,
						ax=ax,
						edgecolors='black',
						linewidths=0.5
				)		
		
		nx.draw_networkx_nodes(G, pos, nodelist=prototype_nodes, node_color="#F8FF7D", node_size=1000, ax=ax, edgecolors='black', linewidths=0.5)
		nx.draw_networkx_edges(G, pos, edgelist=proto_edges, ax=ax, edge_color="red", arrows=True, arrowstyle="-|>,head_length=0.7,head_width=0.5", node_size=1000)
		nx.draw_networkx_edges(G, pos, edgelist=intra_level_edges, ax=ax, edge_color="#09EF01", arrows=True, arrowstyle="-|>,head_length=0.7,head_width=0.5", node_size=1000)
		nx.draw_networkx_edges(G, pos, edgelist=inter_level_edges, ax=ax, edge_color="black", arrows=True, arrowstyle="-|>,head_length=0.7,head_width=0.5", node_size=1000)
		nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8, ax=ax)
		ax.set_title(f"Graph {idx + 1}")

	for ax in axes_flat[n:]:
		ax.axis('off')
	
	plt.tight_layout()
	plt.show()

def generate_graph(piece_start_time, piece_end_time, segments_filepath, motives_filepath, harmony_filepath, melody_filepath):
	try:
		layers = parse_analyses.parse_segments_file(segments_filepath, piece_start_time, piece_end_time)
		keys_layer, chords_layer = parse_analyses.parse_harmony_file(piece_start_time, piece_end_time, harmony_filepath)
		# layers.append(keys_layer)
		# layers.append(parse_analyses.parse_motives_file(piece_start_time, piece_end_time, motives_filepath))
		# layers.append(chords_layer)
		layers.append(parse_analyses.parse_motives_file(piece_start_time, piece_end_time, motives_filepath))
		layers.extend(parse_analyses.parse_harmony_file(piece_start_time, piece_end_time, harmony_filepath))
		layers.append(parse_analyses.parse_melody_file(piece_start_time, piece_end_time, melody_filepath))
		G = create_graph(piece_start_time, piece_end_time, layers)

		for node in G.nodes: # hack for some graphs whose CSV files don't match MIDI for reasons i'm not sure of. they contain nodes like 'Sfiller' due to timing conversion problems
			if 'filler' in node and 'Pfiller' not in node:
				print(f"Error processing graph at {os.path.dirname(segments_filepath)}: MIDI-CSV conversion problem")
				return
			
		layers_with_index = get_unsorted_layers_from_graph_by_index(G)
		return (G, layers_with_index)
	except Exception as e:
		print(f"Error processing graph at {os.path.dirname(segments_filepath)}: {e}")

def process_graphs(midi_filepath):
	mid = mido.MidiFile(midi_filepath)
	tempo_changes = fc.preprocess_tempo_changes(mid)

	base_path = midi_filepath[:-4]
	mid_df = pd.read_csv(base_path + ".csv")

	# Convert durations to seconds and calculate end times
	mid_df['duration_seconds'] = mid_df['duration'].apply(
			lambda duration: fc.ticks_to_secs_with_tempo_changes(
					duration * mid.ticks_per_beat, tempo_changes, mid.ticks_per_beat)
	)
	mid_df['end_time'] = mid_df['onset_seconds'] + mid_df['duration_seconds']
	piece_end_time = mid_df['end_time'].max()
	piece_start_time = mid_df['onset_seconds'].min()

	# segments_file = base_path + '_scluster_scluster_segments.txt'
	segments_file = base_path + '_sf_fmc2d_segments.txt'
	motives_file = base_path + '_motives1.txt'
	harmony_file = base_path + '_functional_harmony.txt'
	melody_file = base_path + '_vamp_mtg-melodia_melodia_melody_contour.csv'
	graph_and_layers = generate_graph(piece_start_time, piece_end_time, segments_file, motives_file, harmony_file, melody_file)
	if graph_and_layers:
		G, layers = graph_and_layers
		# visualize([G], [layers])
		# augment_graph(G)
		visualize_p([G], [layers])
		hierarchical_status = 'hier' if '_scluster_scluster_segments.txt' in segments_file else 'flat'
		aug_graph_filepath = base_path + f"_augmented_graph_{hierarchical_status}.pickle"
		if not os.path.exists(aug_graph_filepath):
			with open(aug_graph_filepath, 'wb') as f:
				pickle.dump(G, f)
				print("Saved graph at", aug_graph_filepath)
		# else:
		# 	print(f"File {aug_graph_filepath} already exists, skipping save.")
	
if __name__ == "__main__":
	# def delete_files_with_substring(directory, substring):
	# 	for root, _, files in os.walk(directory):
	# 		for file in files:
	# 			if substring in file:
	# 				file_path = os.path.join(root, file)
	# 				print(f"Deleting {file_path}")
	# 				os.remove(file_path)

	# directory = '/Users/ilanashapiro/Documents/constraints_project/project/datasets'
	# substring = '_melody_contour'
	# delete_files_with_substring(directory, substring)
	# sys.exit(0)

	# directory = '/Users/ilanashapiro/Documents/constraints_project/project/datasets/chopin/classical_piano_midi_db/chpn-p7'
	# directory = '/Users/ilanashapiro/Documents/constraints_project/project/datasets/mozart/kunstderfuge/mozart-l_menuet_6_(nc)werths'
	directory = '/Users/ilanashapiro/Documents/constraints_project/project/datasets'
	# directory = directory + '/beethoven/kunstderfuge/biamonti_461_(c)orlandi'
	# directory = directory + '/chopin/classical_piano_midi_db/chpn-p7'
	directory = directory + '/beethoven/kunstderfuge/biamonti_461_(c)orlandi'

	tasks = []
	for dirpath, _, _ in os.walk(directory):
		motives_files = [file for file in glob.glob(os.path.join(dirpath, '*_motives1.txt')) if os.path.getsize(file) > 0]
		if motives_files:
			motives_file = motives_files[0] 
			midi_filepaths = glob.glob(os.path.join(dirpath, '*.mid'))
			if midi_filepaths:
				midi_file = midi_filepaths[0]
				tasks.append(midi_file)
			else:
				raise Exception("No midi file but motives", dirpath)
	
	with Pool() as pool:
		pool.map(process_graphs, tasks)
			