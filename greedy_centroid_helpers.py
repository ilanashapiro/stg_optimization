import numpy as np

def pad_adj_matrices(graphs):
  all_nodes = set()
  for G in graphs:
    all_nodes.update(G.nodes())
  node_mapping = {node: i for i, node in enumerate(all_nodes)}
  new_adj_matrices = []
  
  for G in graphs:
    size = len(all_nodes)
    new_adj_matrix = np.zeros((size, size))
    
    for node in G.nodes():
      for neighbor in G.neighbors(node):
        if G.has_edge(node, neighbor): # Since all STGs are directed this ONLY handles the directed case
          new_adj_matrix[node_mapping[node], node_mapping[neighbor]] = 1
    
    new_adj_matrices.append(new_adj_matrix)
  
  return new_adj_matrices, node_mapping
