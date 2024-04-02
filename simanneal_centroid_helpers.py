import numpy as np
import networkx as nx 
import re 

def pad_adj_matrices(graphs):
  all_nodes = set()
  for G in graphs:
    all_nodes.update(G.nodes())
  node_idx_mapping = {node: i for i, node in enumerate(all_nodes)}
  idx_node_mapping = {v: k for k, v in node_idx_mapping.items()}
  new_adj_matrices = []
  
  for G in graphs:
    size = len(all_nodes)
    new_A = np.zeros((size, size))
    
    for node in G.nodes():
      for neighbor in G.neighbors(node):
        if G.has_edge(node, neighbor): # Since all STGs are directed this ONLY handles the directed case
          new_A[node_idx_mapping[node], node_idx_mapping[neighbor]] = 1
    
    new_adj_matrices.append(new_A)

  return new_adj_matrices, idx_node_mapping

def adj_matrix_to_graph(A, idx_node_mapping):
  # print(A[573], idx_node_mapping)
  G = nx.DiGraph()

  for i in range(A.shape[0]):
    for j in range(i, A.shape[1]):
      if A[i, j] > 0:
        G.add_edge(idx_node_mapping[i], idx_node_mapping[j])

  for node in G.nodes():
    G.nodes[node]['label'] = node
    match = re.search(r'N(\d+)$', node)
    if match:
      index_value = int(match.group(1))
      G.nodes[node]['index'] = index_value

  return G
