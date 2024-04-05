import z3
import numpy as np 
import json
import re
import z3_matrix_projection_helpers as helpers 

centroid = np.loadtxt('centroid1.txt')
with open("centroid_node_mapping1.txt", 'r') as file:
  node_mapping = json.load(file)
  node_mapping = {int(k): v for k, v in node_mapping.items()}

n = len(node_mapping) 
solver = z3.Optimize()

levels_partition = helpers.partition_levels(node_mapping)
max_seg_level = len(levels_partition.keys()) - 1

# Declare Z3 variables to enforce constraints on
# Create a matrix in Z3 for adjacency; A[i][j] == 1 means an edge from i to j
A = [[z3.Bool(f"A_{i}_{j}") for j in range(n)] for i in range(n)]

# Additional Z3 variables to denote whether a node is a prototype or an instance
is_prototype = {}
is_instance = {}

for index, node_id in node_mapping.items():
  proto_match = re.match(r"^Pr", node_id)
  
  is_prototype[index] = z3.Bool(f"is_prototype_{index}")
  is_instance[index] = z3.Bool(f"is_instance_{index}")
  
  if proto_match:
    # It's a prototype, so set is_prototype to True and is_instance to False
    # Note: Z3's And, Or, Implies, etc., can enforce logical conditions
    prototype_cond = is_prototype[index]
    instance_cond = z3.Not(is_instance[index])
  else:
    # It's an instance, so set is_instance to True and is_prototype to False
    prototype_cond = z3.Not(is_prototype[index])
    instance_cond = is_instance[index]
  
  # Enforcing the determined conditions
  solver.add(prototype_cond, instance_cond)

# Constraint: Every instance node must be the child of exactly one prototype node
def add_prototype_to_instance_constraints(is_instance):
  for i in range(n):
    if i in is_instance:
      incoming_prototype_edges = z3.Sum([z3.If(z3.And(A[j][index], is_prototype[j]), 1, 0) for j in range(n)])
      exactly_one_prototype_parent = incoming_prototype_edges == 1 # Ensure exactly one incoming edge from a prototype unless it's a dummy node
      solver.add(exactly_one_prototype_parent)

# Constraint: Every instance node not at the top level of the hierarchy, must have 1 or 2 parents in the level above it
def add_inter_level_temporal_constraints():
  for index, node_id in node_mapping.items():
    parsed = helpers.parse_node_id(node_id)
    if parsed:
      if parsed[0] == 'S':
        zero_indexed_level = parsed[2] - 1 # bc levels are encoded as 1-indexed
        if zero_indexed_level > 0: # top level doesn't have instance parents
          potential_parents = levels_partition[zero_indexed_level]
          parent_connections = z3.Sum([z3.If(A[j][index], 1, 0) for j in potential_parents])
          solver.add(z3.Or(parent_connections == 1, parent_connections == 2))
      elif parsed[0] == 'P':
        potential_parents = levels_partition[max_seg_level]
        parent_connections = z3.Sum([z3.If(A[j][index], 1, 0) for j in potential_parents])
        solver.add(z3.Or(parent_connections == 1, parent_connections == 2))


