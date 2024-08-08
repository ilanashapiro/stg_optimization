import os, sys
import pandas as pd
import random
import json 
import pickle
import numpy as np

DIRECTORY = "/home/ilshapiro/project"
sys.path.append(DIRECTORY)
sys.path.append(f"{DIRECTORY}/centroid")

import simanneal_centroid_run, simanneal_centroid_helpers

composer_graphs = {}

with open('dataset_composers_in_phylogeny.txt', 'r') as file:
  for line in file:
    composer_graphs[line.strip()] = {'kunstderfuge':[], 'classical_piano_midi_db':[]}

def get_approx_end_time(csv_path):
    df = pd.read_csv(csv_path)
    if 'onset_seconds' in df.columns:
      return df['onset_seconds'].max()
    else:
      raise ValueError(f"'onset_seconds' column not found in {csv_path}")

def build_composers_dict(base_dir=f"{DIRECTORY}/datasets"):
  for root, _, files in os.walk(base_dir):
    relative_path = os.path.relpath(root, base_dir)
    path_parts = relative_path.split(os.sep)

    if len(path_parts) >= 3:
      source_dir = path_parts[-2]
      composer_dir = path_parts[-3]
      
      for file in files:
        if file.endswith("_augmented_graph_flat.pickle"):
          full_path = os.path.join(root, file)
          csv_path = full_path.replace("_augmented_graph_flat.pickle", ".csv")

          if composer_dir in composer_graphs:
            duration = get_approx_end_time(csv_path)
            size = os.path.getsize(full_path)
            composer_graphs[composer_dir][source_dir].append((full_path, duration, size))

def filter_graphs_by_mean_duration(composer_graphs, max_diff=3000):
    # Step 1: Gather all durations across all composers
    all_sizes = []
    for graphs in composer_graphs.values():
        for graph_list in graphs.values():
            all_sizes.extend(size for _, _, size in graph_list)
    
    if not all_sizes:
        return {}
    
    # Step 2: Calculate the global mean duration
    mean_size = sum(all_sizes) / len(all_sizes)
    print("MEAN", mean_size)
    print("MIN", min(all_sizes))

    mean_size = 15000
    
    # Step 3: Filter graphs based on the global mean duration
    filtered_composer_graphs = {}
    for composer, graphs in composer_graphs.items():
        selected_graphs = []
        for source, graph_list in graphs.items():
            for graph, duration, size in graph_list:
                if abs(size - mean_size) <= max_diff:
                    selected_graphs.append((graph, duration, size))
        
        if selected_graphs:
            filtered_composer_graphs[composer] = selected_graphs
    
    return filtered_composer_graphs

filtered_composer_graphs_path = f"{DIRECTORY}/experiments/filtered_composer_graphs.txt"
if False:#os.path.exists(filtered_composer_graphs_path):
    with open(filtered_composer_graphs_path, 'r') as file:
      filtered_composer_graphs = json.load(file)
      print("LOADED", filtered_composer_graphs_path)
else:
  build_composers_dict()
  composer_graphs = {composer: graphs for composer, graphs in composer_graphs.items() if len(graphs['classical_piano_midi_db']) + len(graphs['kunstderfuge']) > 1}
  filtered_composer_graphs = filter_graphs_by_mean_duration(composer_graphs)
  with open(filtered_composer_graphs_path, 'w') as file:
    json.dump(filtered_composer_graphs, file, indent=4)
    print("SAVED", filtered_composer_graphs_path)

total_composers = 0
for composer, graphs in filtered_composer_graphs.items():
  print(composer, len(graphs))

def select_graphs(d):
  selected_graphs_dict = {}
  for composer, graph_filepaths in d.items():
    if graph_filepaths:
      # selected_graph = random.choice(graphs) # select it randomly
      selected_graph = min(graph_filepaths, key=(lambda path: os.path.getsize(path[0]))) # or find smallest graph in list. x[0] is the file path
      selected_graphs_dict[composer] = selected_graph
  return selected_graphs_dict

composer_selected_graph_fp_dict = select_graphs(filtered_composer_graphs)
composer_selected_graph_dict = {}
for composer, (pickle_file, duration, size) in composer_selected_graph_fp_dict.items():
  print(composer, duration, size)
  with open(pickle_file, 'rb') as f:
	  composer_selected_graph_dict[(composer, pickle_file)] = pickle.load(f)

def list_of_tuples_to_tuple_of_lists(list_of_tuples):
  transposed = list(zip(*list_of_tuples))
  return tuple(list(group) for group in transposed)

# we do list_of_tuples_to_tuple_of_lists for deterministic ordering instead of keys and then values, which idk if will be certain to be the same order
composers_list, STG_augmented_list = list_of_tuples_to_tuple_of_lists(composer_selected_graph_dict.items()) 
listA_G, idx_node_mapping, node_metadata_dict = simanneal_centroid_helpers.pad_adj_matrices(STG_augmented_list)

A_G1 = listA_G[0]
distances = [0]
for A_G2 in listA_G[:4]:
  _, struct_dist = simanneal_centroid_run.align_graph_pair(A_G1, A_G2, idx_node_mapping, node_metadata_dict)
  distances.append(struct_dist)

curr_composer, piece = composers_list[0]
print("LIST", composers_list)
for i, dist in enumerate(distances):
   print(f"{curr_composer} to {composers_list[i][0]} is {dist}")
   print(composers_list[i][1])
   print()
