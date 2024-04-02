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
      node_label = f"S{id}L{layer_idx + 1}"
      node_id = f"{node_label}N{idx}"
      layer.append({'start': float(start), 'end': float(end), 'id': node_id, 'label': node_id})
    layers.append(layer)
  
  return layers

def parse_motives_file(file_path):
  with open(file_path, 'r') as file:
    data = file.read().strip().split('\n\n')  # Split into chunks by blank line

  motif_layer = []
  pattern_num = 0

  for chunk in data:
    if chunk.startswith("pattern"):
      pattern_num += 1
      lines = chunk.split('\n')[1:]  # Skip the pattern line itself
      occurrence_num = 0
      start, end = None, None
      for line in lines:
        if line.startswith("occurrence"):
          if start is not None and end is not None:
            # Save the previous occurrence before starting a new one
            node_label = f"P{pattern_num}O{occurrence_num}"
            motif_layer.append({'start': float(start), 'end': float(end), 'id': node_label, 'label': node_label})
          occurrence_num += 1
          start, end = None, None # Reset start and end for the new occurrence
        else:
          time, _ = line.split(',', 1)
          if start is None:
            start = time  # First line of occurrence sets the start time
          end = time
      # Add the last occurrence in the chunk
      if start is not None and end is not None:
        node_label = f"P{pattern_num}O{occurrence_num}"
        motif_layer.append({'start': float(start), 'end': float(end), 'id': node_label, 'label': node_label})

  # Sort by start time and add index based on the sort
  sorted_data = sorted(motif_layer, key=lambda x: x['start'])
  for idx, item in enumerate(sorted_data):
    item['id'] += f"N{idx}"
  
  return motif_layer

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

def get_unsorted_layers_from_graph_by_index(G):
  # Regular expressions to match the ID/label formats
  structure_pattern = re.compile(r"^S(\d+)L(\d+)N(\d+)$")
  motive_pattern = re.compile(r"^P(\d+)O(\d+)N(\d+)$")

  # Step 1: Partition the nodes based on their ID/label format
  partition_structure = []
  partition_motives = []
  for node, data in G.nodes(data=True):
    if structure_pattern.match(node):
      result = structure_pattern.search(node)
      if result:
        n3 = result.group(3)
        partition_structure.append({'id': node, 'label': data['label'], 'index': int(n3)})
    elif motive_pattern.match(node):
      result = motive_pattern.search(node)
      if result:
        n3 = result.group(3)
        partition_motives.append({'id': node, 'label': data['label'], 'index': int(n3)})

  # Step 2: For the partition_structure list, further partition by the L{n2} substring
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

  return layers

# def get_sorted_layers_from_graph_by_structure(G):
#     # Step 1: Identify all numeric nodes, excluding prototype nodes
#     numeric_nodes = [node for node in G.nodes() if isinstance(node, int)]

#     # Step 2: Identify the top-level nodes (nodes with no predecessors)
#     top_level = []
#     for node in numeric_nodes:
#       predecessors = list(G.predecessors(node))
#       # Check if all predecessors are prototype nodes (i.e., start with 'S' or 'P')
#       if all(pred.startswith('S') or pred.startswith('P') for pred in predecessors if isinstance(pred, str)):
#         first_top_node = node
#         break
#     # DOESNT WORK
    
#     top_level.append(node)

#     # Partition the graph into levels
#     levels = []
#     current_level = top_level
#     while current_level:
#         levels.append(current_level)
#         # Find all unique successors of the nodes in the current level
#         next_level = list({succ for node in current_level for succ in G.successors(node)})
#         current_level = next_level

#     # Step 3: Sort nodes within each level based on the order determined by edges
#     sorted_levels = []
#     for level in levels:
#       if len(level) <= 1:
#         sorted_levels.append(level)
#         continue
      
#       # Create a subgraph containing only the current level's nodes and sort them
#       subgraph = G.subgraph(level)
#       sorted_nodes = list(nx.topological_sort(subgraph))  # This sorts nodes based on directed edges
#       sorted_levels.append(sorted_nodes)

#     return sorted_levels

# augment with prototype nodes and intra-level layers
def augment_graph(G):
  layers = get_unsorted_layers_from_graph_by_index(G)
  
  # Step 1: Add prototype nodes and edges to instances
  for layer in layers:
    for node_info in layer:
      # Determine prototype label
      if 'L' in node_info['id']:
        proto_node_id = 'PrS' + node_info['id'].split('S')[1].split('L')[0]
      else:  # 'O' in node_info['id']
        proto_node_id = 'PrP' + node_info['id'].split('P')[1].split('O')[0]

      # Add prototype node if not already present
      if proto_node_id not in G:
        G.add_node(proto_node_id, label=proto_node_id)

      # Add edge from prototype node to instance node
      if not G.has_edge(proto_node_id, node_info['id']):
        G.add_edge(proto_node_id, node_info['id'])

  # Step 2: Add intra-level edges based on index
  for layer in layers:
    # Sort nodes within each layer by their index to ensure proper sequential connections
    sorted_layer_nodes = sorted(layer, key=lambda x: x['index'])
    
    # Iterate through sorted nodes to add edges
    for i in range(len(sorted_layer_nodes)-1):
      current_node_id = sorted_layer_nodes[i]['id']
      next_node_id = sorted_layer_nodes[i+1]['id']
      # Add directed edge from current node to next
      if not G.has_edge(current_node_id, next_node_id):
        G.add_edge(current_node_id, next_node_id)

# def replace_node_ids_with_integers(G):
#     # Step 1: Generate a new mapping for node IDs to integers, skipping prototype nodes
#     id_mapping = {}
#     new_id_counter = 1  # Start counter for new IDs

#     for node in list(G.nodes()):  # Use list to make a copy of node iterator
#         isNotPrototype = bool(re.search(r'N\d+$', node))
#         if isNotPrototype:  # Skip prototype nodes
#             # Assign a new unique integer ID
#             id_mapping[node] = new_id_counter
#             new_id_counter += 1

#     # Step 2: Create new nodes with integer IDs and transfer attributes
#     for old_id, new_id in id_mapping.items():
#         G.add_node(new_id, **G.nodes[old_id])

#     # Step 3: Update edges to use the new node IDs
#     # First, copy the edges because we'll modify the graph in-place
#     edges_to_update = [(u, v, d) for u, v, d in G.edges(data=True) if u in id_mapping or v in id_mapping]

#     # Remove old edges and add updated ones
#     for u, v, d in edges_to_update:
#         G.remove_edge(u, v)
#         new_u = id_mapping.get(u, u)  # Get new ID or keep the original if it's a prototype
#         new_v = id_mapping.get(v, v)  # Same as above
#         G.add_edge(new_u, new_v, **d)

#     # Step 4: Remove old nodes
#     for old_id in id_mapping.keys():
#         G.remove_node(old_id)

#     return G

def visualize(graph_list, layers_list, labels_dicts = None):
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
    layer_height = 1.0 / (len(layers) + 1)
    for i, layer in enumerate(layers):
      y = 1 - (i + 1) * layer_height  # Adjust y-coordinate
      # Sort nodes if necessary (i.e. sort last/motives layer based on start attribute)
      if i == len(layers) - 1 and 'index' in layer[0]:
        layer = sorted(layer, key=lambda node: node['index'])
          
      x_step = 1.0 / (len(layer) + 1)
      for j, node in enumerate(layer):
        x = (j + 1) * x_step
        pos[node['id']] = (x, y)
    
    ax = axes_flat[idx]
    nx.draw(G, pos, labels=labels_dict, with_labels=True, node_size=500, node_color="lightblue", font_size=8, edge_color="gray", arrows=True, ax=ax)
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
      pos[prototype] = (0.05, y)  # Slightly to the right to avoid touching the plot border
      prototype_nodes.append(prototype)

    # Regular node positioning, assuming layers are already sorted
    layer_height = 1.0 / (len(layers) + 1)
    for i, layer in enumerate(layers):
      y = 1 - (i + 1) * layer_height
      x_step = 1.0 / (len(layer) + 1)
      for j, node in enumerate(layer):
        x = (j + 1) * x_step + 0.1  # Adjust x to the right to accommodate prototypes
        pos[node['id']] = (x, y)
    
    ax = axes_flat[idx]
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, node_color="lightblue")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", arrows=True)
    proto_edges = [(u, v) for u, v in G.edges() if u in prototype_nodes]
    nx.draw_networkx_edges(G, pos, edgelist=proto_edges, ax=ax, edge_color="red", arrows=True)
    nx.draw_networkx_labels(G, pos, labels=labels_dict, ax=ax, font_size=8)
    
    ax.set_title(f"Graph {idx + 1}")
  
  for ax in axes_flat[n:]:
    ax.axis('off')
  
  plt.tight_layout()
  plt.show()

def generate_graph(structure_filepath, motives_filepath):
  layers = parse_form_file(structure_filepath)
  motive_layer = parse_motives_file(motives_filepath)
  layers.append(motive_layer)
  G = create_graph(layers)
  layers_with_index = get_unsorted_layers_from_graph_by_index(G) # for rendering purposes
  labels_dict = {d['id']: d['label'] for layer in layers_with_index for d in layer}
  return (G, layers_with_index, labels_dict)

if __name__ == "__main__":
  G, layers, _ = generate_graph('LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short_segments.txt', 'LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short_motives.txt')
  augment_graph(G)
  # replace_node_ids_with_integers(G)
  # layers = get_sorted_layers_from_graph_by_structure(G)
  visualize_p([G], [layers])