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
  G = nx.DiGraph()

  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      source = idx_node_mapping[i]
      sink = idx_node_mapping[j] 
      if A[i, j] > 0:# and 'Pr' not in source: # remove prototype nodes
        G.add_edge(source, sink)
            
  for node in G.nodes():
    G.nodes[node]['label'] = node
    match = re.search(r'N(\d+)$', node)
    if match:
      index = int(match.group(1))
      G.nodes[node]['index'] = index

  return G

'''
types of transforms: add/remove instance-proto edge, add/remove intra-level edge, add/remove inter-level edge
should I iterate thru the entire set of equally high difference values from diff matrix since they introduce other transforms like below?

1. remove instance-proto edge -> replace with different instance-proto edge (consult difference matrix)
2. add instance-proto edge -> remove existing instance-proto edge if it exists
                              if not (i.e. adding new node), then need to also add valid intra-level edge and inter-level edge
3. add intra-level edge -> need to maintain linear chain property (how to do this??)
4. remove intra-level edge -> either replace with another intra-level edge to maintain linear chain, or remove all other edges (i.e. remove the node)
6. add inter-level edge -> if exceeds 2 parents, consult diff matrix to determine which existing inter-level edge to remove for that node
7. remove inter-level edge -> if it leave a no-parent node, consult diff matrix about how to replace that edge to make a new parent

finally, convert augmented STG back to regular STG

OR

try adding (very small) penalty function in energy for when you do an invalid edge operation
and this penalty gets much bigger as the temperature decreases, because at this point we want to basically be adding valid edges

and then i can try projecting onto the nearest valid adj matrix every so often, or just at the end (play around with this)
certainly do this at the end

and some notes:
make sure I am initializing the first alignment annealer with the identity matrix (doesn't really matter)
and then at each iteration of the centroid annealer, I'm maintaining the algimenents themselves and passing these in as the start alignment for the next alignment annealing
'''