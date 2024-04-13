import networkx as nx
import numpy as np
import random 

G1 = nx.DiGraph()
G1.add_node('PrS0', label='PrS0')
G1.add_node('PrS1', label='PrS1')
G1.add_node('PrS2', label='PrS2')
G1.add_node('PrP0', label='PrP0')
G1.add_node('PrP1', label='PrP1')
G1.add_node('S0L1N1', label='S0L1N1', index=1)
G1.add_node('S1L1N2', label='S1L1N2', index=2)
G1.add_node('S0L1N3', label='S0L1N3', index=3)
G1.add_node('S0L2N1', label='S0L2N1', index=1)
G1.add_node('S1L2N2', label='S1L2N2', index=2)
G1.add_node('S2L2N3', label='S2L2N3', index=3)
G1.add_node('S1L2N4', label='S1L2N4', index=4)
G1.add_node('P1O1N1', label='P1O1N1', index=1)
G1.add_node('P0O1N2', label='P0O1N2', index=2)
G1.add_node('P1O2N3', label='P1O2N3', index=3)

G1.add_edge('PrS0', 'S0L1N1')
G1.add_edge('PrS0', 'S0L1N3')
G1.add_edge('PrS0', 'S0L2N1')
G1.add_edge('PrS1', 'S1L1N2')
G1.add_edge('PrS1', 'S1L2N2')
G1.add_edge('PrS1', 'S1L2N4')
# G1.add_edge('S1L2N4', 'PrS1')
G1.add_edge('PrS2', 'S2L2N3')
G1.add_edge('PrP1', 'S2L2N3')
G1.add_edge('PrP1', 'P0O1N2')
G1.add_edge('PrP0', 'P1O1N1')
G1.add_edge('PrP1', 'P1O1N1')
G1.add_edge('PrP0', 'P1O2N3')
G1.add_edge('PrP1', 'P1O2N3')

# G1.add_edge('S0L1N1', 'S0L1N3')
# G1.add_edge('S0L1N1', 'S0L2N1')
G1.add_edge('S0L1N1', 'S1L2N2')
G1.add_edge('S0L1N1', 'S2L2N3')
G1.add_edge('S1L1N2', 'S0L1N3')
G1.add_edge('S1L1N2', 'S1L2N4')
G1.add_edge('S1L1N2', 'S0L2N1')
# G1.add_edge('S1L1N2', 'S1L2N2')
G1.add_edge('S1L1N2', 'S2L2N3')
G1.add_edge('S0L1N3', 'S2L2N3')
# G1.add_edge('S0L1N3', 'S1L2N4')
G1.add_edge('S0L2N1', 'S1L2N2')
G1.add_edge('S0L2N1', 'P1O1N1')
# G1.add_edge('S1L2N2', 'S2L2N3')
# G1.add_edge('S1L2N2', 'P0O1N2')
G1.add_edge('S2L2N3', 'S1L2N4')
G1.add_edge('S2L2N3', 'P1O2N3')
G1.add_edge('S1L2N4', 'P1O2N3')
G1.add_edge('P0O1N2', 'P1O1N1')
# G1.add_edge('P1O1N1', 'S1L2N2')
# G1.add_edge('P1O1N1', 'S1L1N2')
G1.add_edge('S1L1N2', 'P1O1N1')
G1.add_edge('P0O1N2', 'P1O2N3')

# no idea if this works
def generate_STG(num_levels, max_nodes_per_level):
  # Ensure at least one level has 'max_nodes_per_level' nodes
  levels = [random.randint(1, max_nodes_per_level) for _ in range(num_levels)]
  levels[random.randint(0, num_levels-1)] = max_nodes_per_level  # Ensure at least one level has 'max_nodes_per_level' nodes

  # Generate nodes for each level
  level_nodes = []
  proto_parents = {}
  for level_index in range(num_levels):
    nodes = []
    for node_index in range(levels[level_index]):
      if level_index < num_levels-1:  # S nodes
        node_id = f"S{level_index+1}L{level_index+1}N{node_index+1}"
      else:  # P nodes for the last level
        node_id = f"P{level_index+1}O{level_index+1}N{node_index+1}"
      nodes.append(node_id)

      # Assign prototype parents
      proto_id = f"PrS{level_index+1}" if level_index < num_levels-1 else f"PrP{level_index+1}"
      proto_parents[node_id] = proto_id

    level_nodes.append(nodes)

  # Now, let's create the adjacency matrix including prototypes
  total_nodes_count = sum(len(level) for level in level_nodes) + len(set(proto_parents.values()))
  adj_matrix = np.zeros((total_nodes_count, total_nodes_count), dtype=int)

  # Mapping of node ID to index in the adjacency matrix
  node_to_index = {proto: idx for idx, proto in enumerate(set(proto_parents.values()))}
  current_index = len(node_to_index)  # Start indexing instance nodes after all prototypes

  # Fill in the mapping for instance nodes
  for level in level_nodes:
    for node in level:
      node_to_index[node] = current_index
      current_index += 1

  # Create edges from prototypes to their instance nodes and from instances to parents in the level above
  for level_index, nodes in enumerate(level_nodes):
    for node in nodes:
      proto = proto_parents[node]
      adj_matrix[node_to_index[proto], node_to_index[node]] = 1  # Proto to instance edge

      # For simplicity, randomly connecting to one parent in the level above
      if level_index > 0:
        parent_node = random.choice(level_nodes[level_index - 1])
        adj_matrix[node_to_index[parent_node], node_to_index[node]] = 1

  return adj_matrix, node_to_index
