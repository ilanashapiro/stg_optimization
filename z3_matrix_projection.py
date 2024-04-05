from pickletools import optimize
from platform import node
import z3
import numpy as np 
import json
import re
import z3_matrix_projection_helpers as helpers 

centroid = np.loadtxt('centroid1.txt')
with open("centroid_node_mapping1.txt", 'r') as file:
  idx_node_mapping = json.load(file)
  idx_node_mapping = {int(k): v for k, v in idx_node_mapping.items()}

node_idx_mapping = {v: k for k, v in idx_node_mapping.items()}
n = len(idx_node_mapping) 
opt = z3.Optimize()

levels_partition = helpers.partition_levels(idx_node_mapping)
max_seg_level = len(levels_partition.keys()) - 1

# Declare Z3 variables to enforce constraints on
# Create a matrix in Z3 for adjacency; A[i][j] == 1 means an edge from i to j
A = np.array([[z3.Bool(f"A_{i}_{j}") for j in range(n)] for i in range(n)])
A_partition_submatrices_list = helpers.create_partition_submatrices(A, idx_node_mapping, node_idx_mapping, levels_partition)

# Additional Z3 variables to denote whether a node is a prototype or an instance
is_prototype = {}
is_instance = {}

for index, node_id in idx_node_mapping.items():
  proto_match = re.match(r"^Pr", node_id)
  
  is_prototype[index] = z3.Bool(f"is_prototype_{index}")
  is_instance[index] = z3.Bool(f"is_instance_{index}")
  
  if proto_match:
    prototype_cond = is_prototype[index]
    instance_cond = z3.Not(is_instance[index])
  else:
    prototype_cond = z3.Not(is_prototype[index])
    instance_cond = is_instance[index]
  
  opt.add(prototype_cond, instance_cond)

# Constraint: Every instance node must be the child of exactly one prototype node
def add_prototype_to_instance_constraints():
  for i in range(n):
    if i in is_instance:
      incoming_prototype_edges = z3.Sum([z3.If(z3.And(A[j][index], is_prototype[j]), 1, 0) for j in range(n)])
      exactly_one_prototype_parent = incoming_prototype_edges == 1 # Ensure exactly one incoming edge from a prototype unless it's a dummy node
      opt.add(exactly_one_prototype_parent)

# Constraint: Every instance node not at the top level of the hierarchy, must have 1 or 2 parents in the level above it
def add_inter_level_parent_counts_constraints():
  for level, partition_nodes in levels_partition.items():
    for node_id in partition_nodes:
      parsed = helpers.parse_node_id(node_id)
      if parsed:
        node_index = node_idx_mapping[node_id]
        if parsed[0] == 'S':
          print(level)
          if level > 0: # top level doesn't have instance parents by construction in simanneal
            potential_parents = levels_partition[level]
            parent_connections = z3.Sum([z3.If(A[node_idx_mapping[parent_id]][node_index], 1, 0) for parent_id in potential_parents])
            opt.add(z3.Or(parent_connections == 1, parent_connections == 2))     
        elif parsed[0] == 'P':
          potential_parents = levels_partition[max_seg_level]
          parent_connections = z3.Sum([z3.If(A[node_idx_mapping[parent_id]][node_index], 1, 0) for parent_id in potential_parents])
          opt.add(z3.Or(parent_connections == 1, parent_connections == 2))

# Constraint: The instance nodes in every partition should form a linear chain
def add_intra_level_linear_chain():
  for (A_sub_matrix, idx_node_submap) in A_partition_submatrices_list.values():
    partition_nodes = list(idx_node_submap.values()) 
    n = len(partition_nodes) 
    
    start_nodes = [z3.Bool(f"start_{node}") for node in partition_nodes]
    end_nodes = [z3.Bool(f"end_{node}") for node in partition_nodes]
    
    for i in range(len(partition_nodes)):
      # Directly use sub-matrix to count incoming/outgoing edges for node i
      num_incoming_edges = z3.Sum([z3.If(A_sub_matrix[j, i], 1, 0) for j in range(n) if j != i])
      num_outgoing_edges = z3.Sum([z3.If(A_sub_matrix[i, j], 1, 0) for j in range(n) if j != i])
      
      opt.add(start_nodes[i] == (num_outgoing_edges == 1) & (num_incoming_edges == 0))
      opt.add(end_nodes[i] == (num_incoming_edges == 1) & (num_outgoing_edges == 0))
      opt.add((~start_nodes[i] & ~end_nodes[i]) == ((num_incoming_edges == 1) & (num_outgoing_edges == 1)))
    
    # Ensure exactly one start node and one end node in the partition
    opt.add(z3.Sum([z3.If(start_node, 1, 0) for start_node in start_nodes]) == 1)
    opt.add(z3.Sum([z3.If(end_node, 1, 0) for end_node in end_nodes]) == 1)

def add_no_intra_level_adj_prototypes():
  return

def add_inter_level_parent_relationship_constraints():
  return

add_prototype_to_instance_constraints()
print("HERE1")
add_inter_level_parent_counts_constraints()
print("HERE2")
add_intra_level_linear_chain()
print("HERE3")

objective = z3.Sum([z3.If(A[i][j] != bool(centroid[i][j]), 1, 0) for i in range(n) for j in range(n)])
opt.minimize(objective)
print("HERE4")
if opt.check() == z3.sat:
  model = opt.model()
  print("Closest valid graph's adjacency matrix:")
  for i in range(n):
    print([model.evaluate(A[i][j]) for j in range(n)])
else:
  print("Problem has no solution")