from cProfile import label
import networkx as nx
import matplotlib.pyplot as plt

def parse_form_file(file_path):
  label_mapping = {}  # Mapping: node id -> label

  with open(file_path, 'r') as file:
    data = file.read().strip().split('\n\n')  # Split into chunks by blank line

  layers = []
  for index, chunk in enumerate(data):
    lines = chunk.split('\n')
    layer = []
    for line in lines:
      start, end, id = line.split('\t')
      node_label = f"S{id}L{index + 1}"
      node_id = f"{node_label}I({start},{end})"
      label_mapping[node_id] = node_label
      layer.append({'start': float(start), 'end': float(end), 'id': node_id})
    layers.append(layer)
  
  return (layers, label_mapping)

def parse_motives_file(file_path):
  label_mapping = {}  # Mapping: node id -> label

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
            label_mapping[node_id] = node_label
            pattern_layer.append({'start': float(start), 'end': float(end), 'id': node_id})
          occurrence_num += 1
          start, end = None, None  # Reset start and end for the new occurrence
        else:
          time, _ = line.split(',', 1)
          if start is None:
            start = time  # First line of occurrence sets the start time
            print(start)
          end = time  # Update end time with each line
      # Don't forget to add the last occurrence in the chunk
      if start is not None and end is not None:
        node_label = f"P{pattern_num}O{occurrence_num}"
        node_id = f"{node_label}I({start},{end})"
        label_mapping[node_id] = node_label
        pattern_layer.append({'start': float(start), 'end': float(end), 'id': node_id})

  # Append the pattern layer as the new bottom-most layer
  layers.append(pattern_layer)

  return layers, label_mapping

def create_graph(layers):
  G = nx.DiGraph()

  for layer in layers:
    for node in layer:
      G.add_node(node['id'], start=node['start'], end=node['end'])

  for i in range(len(layers) - 1):
    for node_a in layers[i]:
      for node_b in layers[i + 1]:
        start_a, end_a = node_a['start'], node_a['end']
        start_b, end_b = node_b['start'], node_b['end']
        if (start_a <= start_b < end_a) or (start_a < end_b <= end_a):
          G.add_edge(node_a['id'], node_b['id'])
  
  return G

def visualize_k_partite_graph(G, layers, label_mapping):
  # Calculate positions: each layer will have its own X (or Y) coordinate for a top-down layout.
  pos = {}  # Positions dictionary: node -> (x, y)
  layer_height = 1.0 / (len(layers) + 1)
  for i, layer in enumerate(layers):
    # Adjust y-coordinate to start from the top
    y = 1 - (i + 1) * layer_height  # Inverting y-coordinate
    x_step = 1.0 / (len(layer) + 1)
    for j, node in enumerate(layer):
      x = (j + 1) * x_step  # x-coordinate based on position within the layer
      pos[node['id']] = (x, y)

  plt.figure(figsize=(8, 12))  # Adjusted figure size for vertical layout
  nx.draw(G, pos, labels=label_mapping, with_labels=True, node_size=500, node_color="lightblue", font_size=8, edge_color="gray", arrows=True)
  plt.show()

structure_layers, structure_label_mapping = parse_form_file('LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short_segments.txt')
motive_layers, motive_label_mapping = parse_motives_file('LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo__motives_seconds.txt')
label_mapping = {**structure_label_mapping, **motive_label_mapping}
layers = structure_layers + motive_layers
G = create_graph(layers)
visualize_k_partite_graph(G, layers, label_mapping)