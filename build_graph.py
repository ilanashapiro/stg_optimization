import networkx as nx
import matplotlib.pyplot as plt
import math 
import re 

def parse_form_file(file_path):
  with open(file_path, 'r') as file:
    data = file.read().strip().split('\n\n')  # Split into chunks by blank line

  layers = []
  for layer_idx, chunk in enumerate(data):
    lines = chunk.split('\n')
    layer = []
    for idx, line in enumerate(lines):
      start, end, id = line.split('\t')
      node_name = f"S{id}L{layer_idx + 1}"
      node_label = f"{node_name}N{idx}"
      node_id = f"{node_name}I({start},{end})"
      layer.append({'start': float(start), 'end': float(end), 'id': node_label, 'label': node_label})
    layers.append(layer)
  
  return layers

def parse_motives_file(file_path):
  with open(file_path, 'r') as file:
    data = file.read().strip().split('\n\n')  # Split into chunks by blank line

  layers = []
  pattern_layer = []
  pattern_num = 1

  for chunk in data:
    if chunk.startswith("pattern"):
      pattern_num += 1
      lines = chunk.split('\n')[1:]  # Skip the pattern line itself
      occurrence_num = 0
      start, end = None, None  # Initialize start and end times
      for line in lines:
        if line.startswith("occurrence"):
          if start is not None and end is not None:
            # Save the previous occurrence before starting a new one
            node_label = f"P{pattern_num}O{occurrence_num}"
            node_id = f"{node_label}I({start},{end})"
            pattern_layer.append({'start': float(start), 'end': float(end), 'id': node_id, 'label': node_label})
          occurrence_num += 1
          start, end = None, None  # Reset start and end for the new occurrence
        else:
          time, _ = line.split(',', 1)
          if start is None:
            start = time  # First line of occurrence sets the start time
          end = time  # Update end time with each line
      # Don't forget to add the last occurrence in the chunk
      if start is not None and end is not None:
        node_label = f"P{pattern_num}O{occurrence_num}"
        node_id = f"{node_label}I({start},{end})"
        pattern_layer.append({'start': float(start), 'end': float(end), 'id': node_id, 'label': node_label})

  # Append the pattern layer as the new bottom-most layer
  layers.append(pattern_layer)

  return layers

def create_graph(layers):
  def relabel_motive_nodes_with_index(G):
    # Extract start values, labels, and original node IDs
    node_starts = {}
    for node in G.nodes():
      # Assuming the start value is encapsulated within the label string
      match = re.search(r"^P(\d+)O(\d+)I\((\d+\.\d+),(\d+\.\d+)\)$", node)
      if match:
        start, _, _, _ = match.groups()
        start = float(start)  # Convert start to float
        node_starts[node] = start

    # Sort nodes by 'start' extracted from their label
    sorted_nodes = sorted(node_starts.items(), key=lambda x: x[1])

    # Prepare a mapping for node relabeling
    relabel_mapping = {}
    for idx, (node_id, _) in enumerate(sorted_nodes, start=1):
      current_label = G.nodes[node_id]['label']
      new_label = f"{current_label}N{idx}"
      
      # Update the 'label' attribute
      G.nodes[node_id]['label'] = new_label
      
      # Generate new node ID without "I(start, end)" and with "N{idx}" instead
      id_without_I = re.sub(r"I\(\d+\.\d+,\d+\.\d+\)", "", node_id)
      new_node_id = f"{id_without_I}N{idx}"
      relabel_mapping[node_id] = new_node_id

    # Relabel nodes in the graph according to the mapping
    G = nx.relabel_nodes(G, relabel_mapping)

    # Return the modified graph
    return G

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

  return relabel_motive_nodes_with_index(G)

def visualize_with_intervals(graph_list, layers_list, label_dicts):
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
    label_dict = label_dicts[idx]
    
    pos = {}  # Positions dictionary: node -> (x, y)
    layer_height = 1.0 / (len(layers) + 1)
    for i, layer in enumerate(layers):
      y = 1 - (i + 1) * layer_height  # Adjust y-coordinate
      
      # Sort nodes if necessary (e.g., last layer based on some attribute)
      if i == len(layers) - 1 and 'start' in layer[0]:
        layer = sorted(layer, key=lambda node: node['start'])
          
      x_step = 1.0 / (len(layer) + 1)
      for j, node in enumerate(layer):
        x = (j + 1) * x_step
        pos[node['id']] = (x, y)
    
    ax = axes_flat[idx]
    nx.draw(G, pos, labels=label_dict, with_labels=True, node_size=500, node_color="lightblue", font_size=8, edge_color="gray", arrows=True, ax=ax)
    ax.set_title(f"Graph {idx + 1}")
  
  # Hide any unused subplots in the grid
  for ax in axes_flat[n:]:
      ax.axis('off')
  
  plt.tight_layout()
  plt.show()

def get_layers_from_graph(G):
  # Regular expressions to match the ID/label formats
  structure_pattern = re.compile(r"^S(\d+)L(\d+)N(\d+)$")
  motive_pattern = re.compile(r"^P(\d+)O(\d+)N(\d+)$")

  # Step 1: Partition the nodes based on their ID/label format
  partition_structure = []
  partition_motives = []
  for node, data in G.nodes(data=True):
      label = data['label']
      # print(label)
      if structure_pattern.match(label):
          result = structure_pattern.search(label)
          if result:
            n3 = result.group(3)
            partition_structure.append({'id': node, 'label': label, 'index': int(n3)})
      elif motive_pattern.match(label):
          result = motive_pattern.search(label)
          if result:
            n3 = result.group(3)
            partition_motives.append({'id': node, 'label': label, 'index': int(n3)})

  # Step 2: For the partition_sli list, further partition by the L{n2} substring
  partition_structure_grouped = {}

  for item in partition_structure:
      # Extract the L{n2} part using regular expression
      match = re.search(r'L(\d+)', item['label'])
      if match:
          l_value = match.group(1)
          if l_value not in partition_structure_grouped:
              partition_structure_grouped[l_value] = []
          partition_structure_grouped[l_value].append(item)

  # Convert the grouped dictionary into a list of nested lists
  layers = list(partition_structure_grouped.values())
  layers.append(partition_motives)
  # print(partition_motives)
  return layers

def visualize_with_index(graph_list, layers_list):
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
        pos = {}  # Positions dictionary: node -> (x, y)
        layer_height = 1.0 / (len(layers) + 1)
        for i, layer in enumerate(layers):
            y = 1 - (i + 1) * layer_height  # Adjust y-coordinate
            # Sort nodes based on "index" attribute, particularly useful for the last layer
            if 'index' in layer[0]:  # Checking if 'index' exists in the dictionary of the first node
                layer = sorted(layer, key=lambda node: node['index'])
                
            x_step = 1.0 / (len(layer) + 1)
            for j, node in enumerate(layer):
                x = (j + 1) * x_step
                print("HERE", node)
                pos[node['id']] = (x, y)
        
        ax = axes_flat[idx]
        nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=8, edge_color="gray", arrows=True, ax=ax)
        ax.set_title(f"Graph {idx + 1}")
    
    # Hide any unused subplots in the grid
    for ax in axes_flat[n:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_graph(structure_filepath, motives_filepath):
  structure_layers = parse_form_file(structure_filepath)
  structure_label_dict = {d['id']: d['label'] for structure_layer in structure_layers for d in structure_layer}
  motive_layers = parse_motives_file(motives_filepath)
  motive_layers_dict = {d['id']: d['label'] for motive_layer in motive_layers for d in motive_layer}
  label_dict = {**structure_label_dict, **motive_layers_dict}
  layers = structure_layers + motive_layers
  G = create_graph(layers)
  layers = get_layers_from_graph(G) # FIX THIS TO BE DIRECT PARSE FROM THE DICTS
  return (G, layers, label_dict)

if __name__ == "__main__":
  G, layers, label_dict = generate_graph('LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short_segments.txt', 'LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short_motives.txt')
  visualize_with_intervals([G], [layers], [label_dict])