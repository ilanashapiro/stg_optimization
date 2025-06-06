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

DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project/datasets"
# DIRECTORY = '/home/ilshapiro/project/datasets'

# NOTE: does NOT work for secondary chords, we just use it for the example figure
def get_primary_functional_chord_label_from_features(degree, quality):
	maj_roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
	min_roman = [r.lower() for r in maj_roman]
	quality_map = {'M':'', 'm':'', 'a':'+', 'd':'o', 'M7':'M7', 'm7':'m7', 'D7':'7', 'd7':'o7', 'h7':'ø7', 'a6':'+6', 'a7':'aug7'}
	accidental_map = {'+':'#', '-':'b'}

	def parse_degree(degree):
		deg_num = int(degree[1]) if any(acc in degree for acc in accidental_map.keys()) else int(degree)
		deg_acc = accidental_map[degree[0]] if any(acc in degree for acc in accidental_map.keys()) else ""
		return (deg_num, deg_acc)

	deg_num, deg_acc = parse_degree(degree)

	if deg_num < 1 or deg_num > 7:
		raise Exception("Degree not in valid range", deg_num)
	if quality not in quality_map:
		raise Exception("Invalid quality", quality)

	if quality == "a6":
		chord_roman = "aug6"
	else:
		chord_symbol = maj_roman[deg_num - 1] if quality[0].isupper() else min_roman[deg_num - 1] # 0-indexing
		chord_roman = deg_acc + chord_symbol + quality_map[quality]

	return chord_roman

def get_layer_id(node):
	for layer_id in ['S', 'P', 'K', 'C', 'M']:
		if node.startswith(layer_id):
			return layer_id
	raise Exception("Invalid node", node)

def create_graph(piece_start_time, piece_end_time, layers, ablation_level=None):
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
				G.add_node(node['id'], start=node['start'], end=node['end'], label=node['label'], index=node['index'], features_dict=node['features_dict'], layer_rank=get_layer_rank(node['id']))
				
		# Add all filler nodes
		for i in range(len(sorted_nodes) - 1):
			node1 = sorted_nodes[i]
			node2 = sorted_nodes[i + 1]
			
			# There's a gap between node1 and node2
			if node1['end'] < node2['start'] and node2['start'] <= piece_end_time and node1['end'] >= piece_start_time: 
				filler_node_index = node1['index'] + 0.5
				filler_node_id = f"{get_layer_id(node1['id'])}fillerN{filler_node_index}" 
				filler_feature_name = filler_node_id.split('N')[0]
				filler_node = {
					'id': filler_node_id,
					'start': node1['end'],
					'end': node2['start'],
					'label': "filler",
					'index': filler_node_index,
					'features_dict': {filler_feature_name: filler_feature_name},
					'layer_rank': get_layer_rank(node1['id'])
				}
				G.add_node(filler_node_id, **filler_node)
				layer.append(filler_node)
				
		# if the first node in the layer starts after the piece start time, we have a gap at the beginning
		first_node = sorted_nodes[0]
		if first_node['start'] > piece_start_time:
			filler_node_id = f"{get_layer_id(first_node['id'])}fillerN{0.5}"
			filler_feature_name = filler_node_id.split('N')[0]
			filler_node = {
				'id': filler_node_id, 
				'start': piece_start_time, 
				'end': first_node['start'], 
				'label': "filler", 
				'index': 0.5, 
				'features_dict': {filler_feature_name: filler_feature_name}, 
				'layer_rank': get_layer_rank(first_node['id'])
			}
			G.add_node(filler_node_id, **filler_node)
			layer.append(filler_node)

		# if the layer node in the layer ends before the piece end time, we have a gap at the end
		last_node = sorted_nodes[-1]
		generate_example_stg = False
		if generate_example_stg: # this is for generating the example STGs for the slideshow figures, without fillers
			if last_node['end'] < piece_end_time and get_layer_id(last_node['id']) == 'P':
				filler_node_index = last_node['index'] + 1
				filler_node_id = f"P1O1N{filler_node_index}"

				filler_feature_name = filler_node_id.split('N')[0]
				filler_node = {
					'id': filler_node_id, 
					'start': last_node['end'], 
					'end': piece_end_time, 
					'label': 1,
					'index': filler_node_index, 
					'features_dict': {'pattern_num':1},
					'layer_rank': get_layer_rank(last_node['id'])
				}
				print("ADDED", filler_node_id, filler_node)
				G.add_node(filler_node_id, **filler_node)
				layer.append(filler_node)
		else: # this is for generating real STGs for actual purposes
			if last_node['end'] < piece_end_time:
				filler_node_index = last_node['index'] + 0.5
				filler_node_id = f"{get_layer_id(last_node['id'])}fillerN{filler_node_index}"
				filler_feature_name = filler_node_id.split('N')[0]
				filler_node = {
					'id': filler_node_id, 
					'start': last_node['end'], 
					'end': piece_end_time, 
					'label': "filler", 
					'index': filler_node_index, 
					'features_dict': {filler_feature_name: filler_feature_name}, 
					'layer_rank': get_layer_rank(last_node['id'])
				}
				G.add_node(filler_node_id, **filler_node)
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
	G.remove_nodes_from(unused_filler_nodes) # THIS IS CORRECT FOR GENERAL CASE

	# just for now for making figures for slides, ignore otherwise
	# for n in unused_filler_nodes:
	#   if n != 'PfillerN4.5':
	#     G.remove_nodes_from([n])

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

# this partitions INSANCE NODES ONLY
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
		match = re.search(r'L(\d+)', item['id'])
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

'''
This is the augmentation procedure from Section 4.1 in the paper
'''
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
				if "filler" in feature_name: # hack to fix the label for fillers to not have layer prefix
					proto_label = "filler:filler"
				else:
					proto_label = f"{feature_name}:{value}"
				proto_node_id = f"Pr{feature_name.capitalize()}:{value}"
				proto_nodes.append((proto_node_id, proto_label, feature_name, get_layer_id(instance_node_id)))
			
			if not bool(features_dict):
				raise Exception("Node", node, "doesn't have a features dict.")
			
			for (proto_node_id, proto_label, feature_name, source_layer_kind) in proto_nodes:
				if proto_node_id not in G:
					G.add_node(proto_node_id, label=proto_label, layer_rank=get_layer_rank(instance_node_id), feature_name=feature_name, source_layer_kind=source_layer_kind)
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

def visualize(graph_list, layers_list, augmented=True, compress_graph=False, ablation_level=5):
	n = len(graph_list)
	
	# Determine grid size (rows x cols) for subplots
	cols = int(math.ceil(math.sqrt(n)))
	rows = int(math.ceil(n / cols))
	
	# Create a figure with subplots arranged in the calculated grid
	_, axes = plt.subplots(rows, cols, figsize=(8 * cols, 12 * rows))
	
	# Flatten axes array for easy iteration if it's 2D (which happens with multiple rows and columns)
	axes_flat = axes.flatten() if n > 1 else [axes]
	
	for idx, G in enumerate(graph_list):
		layers = layers_list[idx][:ablation_level]
		ablation_nodes = {node['id'] for layer in layers for node in layer} 
		proto_nodes = {node for node in G.nodes if 'pr' in node.lower()}
		nodes_to_remove = set(G.nodes) - (ablation_nodes | proto_nodes)  
		G.remove_nodes_from(nodes_to_remove) # remove the nodes not in the ablation layers
		
		# remove the remaining protos from layers not in the ablation
		zero_arity_nodes = {node for node, degree in G.degree() if degree == 0}
		G.remove_nodes_from(zero_arity_nodes)

		for node in G.nodes():
			if not node.startswith("Pr"):
				if compress_graph:
					proto_parent_ids = []
					for u, v in G.edges():
						if u.startswith("Pr") and v == node:
							proto_parent_ids.append(u)
					features_info = {}
					for proto_parent in proto_parent_ids:
						features_info[G.nodes[proto_parent]['feature_name']] = G.nodes[proto_parent]['label'].split(":")[1]
					match get_layer_id(node):
						case 'S':
							G.nodes[node]["label"] = features_info['section_num']
						case 'P':
							G.nodes[node]["label"] = features_info['pattern_num'] if 'pattern_num' in features_info else 'filler'
						case 'K':
							G.nodes[node]["label"] = f"{features_info['relative_key_num']}{features_info['quality']}"
						case 'C':
							G.nodes[node]["label"] = get_primary_functional_chord_label_from_features(features_info['degree2'], features_info['quality'])
						case 'M':
							G.nodes[node]["label"] = f"{features_info['interval_sign']}{features_info['abs_interval']}"
						case _:
							raise ValueError("Node", node, "has invalid id")
				elif augmented:
					G.nodes[node]["label"] = node[0]
			else:
				if "filler" in G.nodes[node]["label"]:
					G.nodes[node]["label"] = "filler:filler"
				else:
					G.nodes[node]["label"] = node[2:][0].lower() + node[2:][1:]
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
			y = 1 - (index + 1) * proto_y_step # Adjust y-coordinate
			pos[prototype] = (0.05, y)  # Slightly to the right to avoid touching the plot border
			prototype_nodes.append(prototype)

		ax = axes_flat[idx]
		
		# Hide axes border and ticks
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_frame_on(False)

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
				x = (j + 1) * x_step + 0.1 # Adjust x to the right to accommodate prototypes
				pos[node['id']] = (x, y)
		
		ax = axes_flat[idx]
		
		colors = ["#b285f7", "#eb3223", "#f19737", "#a0ce63", "#4cafea"]
		filler_color = '#bfbfbf'
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

		if augmented and not compress_graph:
			nx.draw_networkx_nodes(G, pos, nodelist=prototype_nodes, node_color="#F8FF7D", node_size=1000, ax=ax, edgecolors='black', linewidths=0.5)
			nx.draw_networkx_edges(G, pos, edgelist=proto_edges, ax=ax, edge_color="red", arrows=True, arrowstyle="-|>,head_length=0.7,head_width=0.5", node_size=1000)
			nx.draw_networkx_edges(G, pos, edgelist=intra_level_edges, ax=ax, edge_color="#09EF01", arrows=True, arrowstyle="-|>,head_length=0.7,head_width=0.5", node_size=1000)
		
		nx.draw_networkx_edges(G, pos, edgelist=inter_level_edges, ax=ax, edge_color="black", arrows=True, arrowstyle="-|>,head_length=0.7,head_width=0.5", node_size=1000)
		nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=10, font_weight='bold', ax=ax)
		ax.set_title(f"Graph {idx + 1}")

	for ax in axes_flat[n:]:
		ax.axis('off')
	
	plt.tight_layout()
	plt.show()

def generate_graph(piece_start_time, piece_end_time, segments_filepath, motives_filepath, harmony_filepath, melody_filepath, ablation_levels=5):
	try:
		if ablation_levels < 1 or ablation_levels > 5:
			raise Exception("Ablation levels must be between 1 and 5")
		layers = parse_analyses.parse_segments_file(segments_filepath, piece_start_time, piece_end_time)
		layers.append(parse_analyses.parse_motives_file(piece_start_time, piece_end_time, motives_filepath))
		layers.extend(parse_analyses.parse_harmony_file(piece_start_time, piece_end_time, harmony_filepath)) # contains key level and chords level
		layers.append(parse_analyses.parse_melody_file(piece_start_time, piece_end_time, melody_filepath))
		layers = layers[:ablation_levels]
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
	motives_file = base_path + '_motives3.txt'
	harmony_file = base_path + '_functional_harmony.txt'
	melody_file = base_path + '_vamp_mtg-melodia_melodia_melody_contour.csv' # use extension _TEST.pickle for generating the STGs for the RE exam slides figures beethoven 461
	ablation_level = 2
	graph_and_layers = generate_graph(piece_start_time, piece_end_time, segments_file, motives_file, harmony_file, melody_file, ablation_levels=ablation_level)
	if graph_and_layers:
		G, layers = graph_and_layers
		augment_graph(G)
		# G_c = compress_graph(G)
		# layers_c = get_unsorted_layers_from_graph_by_index(G_c)
		visualize([G], [layers], augmented=False)
		sys.exit(0)
		
		hierarchical_status = 'hier' if '_scluster_scluster_segments.txt' in segments_file else 'flat'
		aug_graph_filepath = base_path + f"_augmented_graph_ablation_{ablation_level}level_{hierarchical_status}_RE.pickle" # use extension _RE.pickle for generating the STGs for the RE exam slides figures
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

	# substring = '_augmented_graph_ablation_1level'
	# delete_files_with_substring(DIRECTORY, substring)
	# sys.exit(0)

	# directory = DIRECTORY + '/chopin/classical_piano_midi_db/chpn-p7'
	# directory = DIRECTORY + '/mozart/kunstderfuge/mozart-l_menuet_6_(nc)werths'
	directory = DIRECTORY + '/beethoven/kunstderfuge/biamonti_461_(c)orlandi'
	# directory = DIRECTORY + '/beethoven/kunstderfuge/biamonti_811_(c)orlandi'
	# directory = DIRECTORY + '/chopin/classical_piano_midi_db/chpn-p7'

	tasks = []
	for dirpath, _, _ in os.walk(directory):
		motives_files = [file for file in glob.glob(os.path.join(dirpath, '*_motives3.txt')) if os.path.getsize(file) > 0]
		if motives_files:
			motives_file = motives_files[0] 
			midi_filepaths = glob.glob(os.path.join(dirpath, '*.mid'))
			midi_filepaths_caps = glob.glob(os.path.join(dirpath, '*.MID'))
			if midi_filepaths:
				midi_file = midi_filepaths[0]
				tasks.append(midi_file)
			elif midi_filepaths_caps:
				midi_file = midi_filepaths_caps[0]
				tasks.append(midi_file)
			else:
				raise Exception("No midi file but motives", dirpath)
	print(tasks)
	with Pool() as pool:
		pool.map(process_graphs, tasks)