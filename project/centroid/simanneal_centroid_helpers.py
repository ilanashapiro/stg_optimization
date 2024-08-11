import numpy as np
import networkx as nx 
import re 
import z3_matrix_projection_helpers as z3_helpers

def pad_adj_matrices(graphs):
  all_nodes = set()
  nodes_features_dict = {}
  for G in graphs:
    all_nodes.update(G.nodes())
    for node_id, data_dict in G.nodes(data=True):
      nodes_features_dict[node_id] = {k: v for k, v in data_dict.items() if k in ['features_dict', 'feature_name', 'source_layer_kind', 'layer_rank']} 
  
  # Convert the set to a sorted list to ensure a consistent order
  # OTHERWISE THIS FUNCTION IS NONDETERMINISTIC
  sorted_nodes = sorted(all_nodes) 
  node_idx_mapping = {node: i for i, node in enumerate(sorted_nodes)}
  idx_node_mapping = {v: k for k, v in node_idx_mapping.items()}
  new_adj_matrices = []
  
  for G in graphs:
    size = len(sorted_nodes)
    new_A = np.zeros((size, size))
    for node in G.nodes():
      for neighbor in G.neighbors(node):
        new_A[node_idx_mapping[node], node_idx_mapping[neighbor]] = 1 # Since all STGs are directed this ONLY handles the directed case
    new_adj_matrices.append(new_A)
  return new_adj_matrices, idx_node_mapping, nodes_features_dict

def adj_matrix_to_graph(A, idx_node_mapping, node_metadata_dict):
  G = nx.DiGraph()

  non_dummy_nodes = set()
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      source = idx_node_mapping[i]
      sink = idx_node_mapping[j] 
      if A[i, j] > 0:
        G.add_edge(source, sink)
        non_dummy_nodes.update([source, sink])
  
  # Add the remaining dummy nodes
  all_nodes = set(idx_node_mapping.values())
  remaining_nodes = all_nodes - non_dummy_nodes
  for node_id in remaining_nodes:
      G.add_node(node_id)

  for node_id in G.nodes():
    G.nodes[node_id].update(node_metadata_dict.get(node_id))
    G.nodes[node_id]['label'] = node_id # not using pretty labels for testing
    match = re.search(r'N(\d+(\.\d+)?)$', node_id) # matches ints and also decimal numbers
    if match:
      index = float(match.group(1))
      G.nodes[node_id]['index'] = index

  return G

def remove_all_dummy_nodes(A, idx_node_mapping):
  non_dummy_indices = list(np.where(np.any(A != 0, axis=0) | np.any(A != 0, axis=1))[0]) # NOTE: the sort order here is nondeterministic!!!
  self_loop_indices = list(np.where(np.diag(A) != 0)[0]) # we consider these dummys (these will only be protos by construction from our constraints, and all non-dummy nodes have constraints preventing self-loops)
  non_dummy_indices += self_loop_indices

  for idx in self_loop_indices: # double check we have no forbidden self-loops
    if np.count_nonzero(A[idx]) > 1 and np.count_nonzero(A[:, idx]) > 1:
      raise Exception("Self-loop found in non-dummy node", idx_node_mapping[idx])

  filtered_matrix = A[non_dummy_indices][:, non_dummy_indices]
  updated_mapping = {new_idx: idx_node_mapping[old_idx] for new_idx, old_idx in enumerate(non_dummy_indices)}
  return filtered_matrix, updated_mapping

def remove_unnecessary_dummy_nodes(A, idx_node_mapping, node_metadata_dict):
  node_idx_mapping = z3_helpers.invert_dict(idx_node_mapping)
  non_dummy_indices = np.where(np.any(A != 0, axis=0) | np.any(A != 0, axis=1))[0] # at least 1 incoming or outgoing edges, i.e. node isn't zero-artiy/dummy
  
  # add ONLY the possible prototypes, dummy or not
  # we do not add all prototypes blindly, just those for feature VALUES that appear in the non-dummy instance nodes
  # i.e. for melody node M4N3, we make sure we have protos PrAbs_interval:4 and PrInterval_sign:+, but not PrInterval_sign:- etc
  proto_node_indices = set()
  prototype_features_partition = z3_helpers.partition_prototype_features(idx_node_mapping, node_metadata_dict) # dict: prototype feature -> prototype nodes of that feature
  for non_dummy_index in non_dummy_indices:
    node_id = idx_node_mapping[non_dummy_index]
    if z3_helpers.is_instance(node_id):
      for feature, value in node_metadata_dict[node_id]['features_dict'].items():
        all_proto_ids_for_feature = prototype_features_partition[feature]
        filtered_proto_ids = list(filter(lambda proto: str(value) in proto, all_proto_ids_for_feature))
        proto_node_indices.update([node_idx_mapping[proto_id] for proto_id in filtered_proto_ids])
  # proto_node_indices = [proto_node_idx for proto_node_idx, proto_node_id in idx_node_mapping.items() if z3_helpers.is_proto(proto_node_id)] # ALL the prototypes (dummy or not)
  
  filtered_indices = list(set(non_dummy_indices) | proto_node_indices)
  filtered_matrix = A[filtered_indices][:, filtered_indices]
  updated_mapping = {new_idx: idx_node_mapping[old_idx] for new_idx, old_idx in enumerate(filtered_indices)}
  return filtered_matrix, updated_mapping

# Generates random n x n permutation alignment matrix
def random_alignment(n):
	perm_indices = np.random.permutation(n)
	identity = np.eye(n)
	return identity[perm_indices]


'''
RULES:
0. It is valid for *any* node to have zero outgoing or incoming edges (this is called a dummy node). However, if the node does have one or more outgoing or incoming edges, then the following rules must apply:

1 [PROTOTYPE-INSTANCE RELATIONSHIPS]. prototype nodes (starting with "Pr") must not have any incoming edges.
1a. prototype nodes of the form "PrS{n}" must only have edges to nodes of the form "S{n1}L{n2}N{n3}"
1b. prototype nodes of the form "PrP{n}" must only have edges to nodes of the form "P{n1}O{n2}N{n3}"
1c. every non-prototype node (S{n1}L{n2}N{n3} or P{n1}O{n2}N{n3}) must be the child of exactly one prototype node

2 [INTER-LEVEL TEMPORAL RELATIONSHIPS]. If n2 > 0 for a S{n1}L{n2}N{n3} node, then that node must have either 1 or 2 parents of the form S{n1'}L{n2 - 1}N{n3'}

3 [INTRA-LEVEL TEMPORAL RELATIONSHIPS]. Consider the set S of S{n1}L{n2}N{n3} nodes that all have the same n2 (i.e. level). Given node n in this set, if n is not a dummy node, then n should have exactly 1 incoming edge from another node in S, and exactly 1 outgoing edge to yet another node in S. There should also be exactly *one* non-dummy node in S that only has an incoming edge from another node in S, and no outgoing. In this way, the non-dummy nodes in S form a linear chain.
3a. Given 2 nodes a1 and a2 that are adjacent in the same-level linear chain (i.e. a1 and a2 are both in S and a1 has an edge from itself to a2), then a1 and a2 should NOT have the same prototype node as a parent. 

EDGES I NEVER WANT TO ADD THAT WILL ALWAYS MAKE THE SCORE WORSE:
1. instance->prototype edges
2. prototype-prototype edges
-----3. edges across non-adjacent levels----> NOPE ACTUALLY THIS COULD BE AN INTERMEDIATE STEP TO VALIDITY IF WE ARE REMOVING AN ENTIRE LEVEL ETC
3. edges from instace to the wrong prototype

TRYING TO KEEP IT VALID AT EVERY STEP:
types of transforms: add/remove instance-proto edge, add/remove intra-level edge, add/remove inter-level edge
should I iterate thru the entire set of equally high difference values from diff matrix since they introduce other transforms like below?

1. remove instance-proto edge -> replace with different instance-proto edge (consult difference matrix)
2. add instance-proto edge -> remove existing instance-proto edge if it exists
                              if not (i.e. adding new node), then need to also add valid intra-level edge and inter-level edge
3. add intra-level edge -> need to maintain linear chain property (how to do this??)
4. remove intra-level edge -> either replace with another intra-level edge to maintain linear chain, or remove all other edges (i.e. remove the node)
6. add inter-level edge -> if exceeds 2 parents, consult diff matrix to determine which existing inter-level edge to remove for that node
7. remove inter-level edge -> if it leave a no-parent node, consult diff matrix about how to replace that edge to make a new parent

BUT THIS CAN MAKE THE ANNEALER GET STUCK

SO HOW ABOUT:

try adding (very small) penalty function in energy for when you do an invalid edge operation
and this penalty gets much bigger as the temperature decreases, because at this point we want to basically be adding valid edges

and then i can try projecting onto the nearest valid adj matrix every so often, or just at the end (play around with this)
certainly do this at the end

and some notes:
make sure I am initializing the first alignment annealer with the identity matrix (doesn't really matter)
and then at each iteration of the centroid annealer, I'm maintaining the algimenents themselves and passing these in as the start alignment for the next alignment annealing
'''