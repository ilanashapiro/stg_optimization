import z3
import numpy as np 
import json
import re

A_g = np.loadtxt('centroid.txt')
with open("centroid_node_mapping.txt", 'r') as file:
  centroid_node_mapping = json.load(file)
  centroid_node_mapping = {int(k): v for k, v in centroid_node_mapping.items()}

n = len(centroid_node_mapping) 

# Declare Z3 variables to enforce constraints on
# Create a matrix in Z3 for adjacency; A[i][j] == 1 means an edge from i to j
A = [[z3.Bool(f"A_{i}_{j}") for j in range(n)] for i in range(n)]

# Additional Z3 variables to denote whether a node is a prototype or an instance
is_prototype = {}
is_instance = {}

for index, node_id in centroid_node_mapping.items():
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
  z3.Solver().add(prototype_cond, instance_cond)

def is_dummy_node(n, A):
  incoming_edges = z3.Sum([z3.If(A[i][n], 1, 0) for i in range(n)])
  outgoing_edges = z3.Sum([z3.If(A[n][j], 1, 0) for j in range(n)])
  return z3.And(incoming_edges == 0, outgoing_edges == 0)

def parse_node_id(node_id):
  # Pattern for segment nodes S{n1}L{n2}N{n3}
  s_pattern = re.compile(r'S(\d+)L(\d+)N(\d+)')
  s_match = s_pattern.match(node_id)
  if s_match:
    n1, n2, n3 = map(int, s_match.groups())
    return ('S', n1, n2, n3)
  
  # Pattern for motiv nodes P{n1}O{n2}N{n3}
  p_pattern = re.compile(r'P(\d+)O(\d+)N(\d+)')
  p_match = p_pattern.match(node_id)
  if p_match:
    n1, n2, n3 = map(int, p_match.groups())
    return ('P', n1, n2, n3)

# Constraint: Every non-prototype, non-dummy node must be the child of exactly one prototype node
for i in range(n):
  if i in is_instance:
    dummy_condition = is_dummy_node(index, A)
    incoming_prototype_edges = z3.Sum([z3.If(z3.And(A[j][index], is_prototype[j]), 1, 0) for j in range(n)])
    exactly_one_prototype_parent = z3.If(dummy_condition, True, incoming_prototype_edges == 1) # Ensure exactly one incoming edge from a prototype unless it's a dummy node
    z3.Solver().add(exactly_one_prototype_parent)
  print(i)

#------------------------------------------------------------------------

# def apply_segmentation_node_parent_constraints(index, n2, A, centroid_node_mapping):
#   dummy_condition = is_dummy_node(index, A)
#   potential_parents = [j for j, other_id in centroid_node_mapping.items() if other_id.startswith('S') and f'L{n2-1}' in other_id]
#   parent_connections = z3.Sum([z3.If(A[j][index], 1, 0) for j in potential_parents])
#   constraint = z3.If(dummy_condition, True, z3.Or(parent_connections == 1, parent_connections == 2))
#   z3.solver.add(constraint)

# def apply_motif_node_parent_constraints(index, A, centroid_node_mapping, highest_s_level):
#   dummy_condition = is_dummy_node(index, A)
#   for j, other_id in centroid_node_mapping.items():

#   potential_parents = [j for j, other_id in centroid_node_mapping.items() if other_id.startswith('S') and other_id.startswith(other_id)[1] == highest_s_level]
#   parent_connections = z3.Sum([z3.If(A[j][index], 1, 0) for j in potential_parents])
#   constraint = z3.If(dummy_condition, True, z3.Or(parent_connections == 1, parent_connections == 2))
#   z3.solver.add(constraint)

# # Constraint: For each non-dummy node, enforce the constraint on its parents
# for index, node_id in centroid_node_mapping.items():
#   parsed = parse_node_id(node_id)  # Assuming this function now identifies the type and parses IDs
#   if parsed:
#     if parsed[0] == 'S':
#       _, n1, n2, n3 = parsed
#       if n2 > 1: # since the first level doesn't need 
#         apply_segmentation_node_parent_constraints(index, n2, A, centroid_node_mapping)
    
#     elif parsed[0] == 'P':
#         _, n1, n2, n3 = parsed
#         apply_motif_node_parent_constraints(index, A, centroid_node_mapping, highest_s_level)

