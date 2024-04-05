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

NodeSort = z3.IntSort()
NO_NODE = -1

# Uninterpreted functions
is_prototype = z3.Function('is_prototype', NodeSort, z3.BoolSort()) # Int -> Bool
is_instance = z3.Function('is_instance', NodeSort, z3.BoolSort())

instance_parent1 = z3.Function('instance_parent1', NodeSort, NodeSort)
instance_parent2 = z3.Function('instance_parent2', NodeSort, NodeSort)
proto_parent = z3.Function('proto_parent', NodeSort, NodeSort)

# orderings for linear chain
# IMPORTANT: the nodes indices for these functions refer to the RELEVANT PARTITION SUBMATRIX, NOT the entire centroid matrix A!!!
pred = z3.Function('pred', NodeSort, NodeSort)
succ = z3.Function('succ', NodeSort, NodeSort)
rank = z3.Function('rank', NodeSort, z3.IntSort())

for index, node_id in idx_node_mapping.items():
  proto_match = re.match(r"^Pr", node_id)

  # Ensure that a node cannot be both a prototype and an instance simultaneously
  # by asserting that one is true implies the other is false.
  if proto_match:
    opt.add(is_prototype(index) == True)
    opt.add(is_instance(index) == False)
  else:
    opt.add(is_prototype(index) == False)
    opt.add(is_instance(index) == True)

# Constraint: Every instance node must be the child of exactly one prototype node
def add_prototype_to_instance_constraints():
  for i in range(n):
    opt.add(is_instance(i) == True)

    for j in range(n):
      if j != i:  # avoid self loops
        # If j is the proto parent of i, then proto_parent(i) should equal j
        opt.add(z3.Implies(z3.And(A[j][i], is_prototype(j)), proto_parent(i) == j))

    incoming_prototype_edges = z3.Sum([
      z3.If(z3.And(A[j][i], is_prototype(j)), 1, 0) for j in range(n)
      if j != i # avoid self loops
    ])
    opt.add(incoming_prototype_edges == 1)

# Constraint: Every instance node not at the top level of the hierarchy, must have 1 or 2 parents in the level above it
def add_inter_level_parent_counts_constraints():
  for level, partition_nodes in levels_partition.items():
    for node_id in partition_nodes:
      parsed = helpers.parse_node_id(node_id)
      if parsed:
        node_index = node_idx_mapping[node_id]
        if level > 0: # top level doesn't have instance parents by construction in simanneal
          potential_parents = levels_partition[level - 1]
          parent_count = z3.Sum([z3.If(A[node_idx_mapping[parent_id]][node_index], 1, 0) for parent_id in potential_parents])
          opt.add(z3.Or(parent_count == 1, parent_count == 2))

          # Assign the 1 or 2 parents to non-zero level instance nodes for future reference in the constraint about parent orders based on the linear chain
          for parent_id1 in potential_parents:
            # Assign the first parent
            parent_condition1, parent_index1 = A[node_idx_mapping[parent_id1]][node_index], node_idx_mapping[parent_id1]
            opt.add(z3.Implies(parent_condition1, instance_parent1(node_index) == parent_index1))
            for parent_id2 in potential_parents:
              parent_condition2, parent_index2 = A[node_idx_mapping[parent_id2]][node_index], node_idx_mapping[parent_id2]
              if parent_index1 != parent_index2:
                # Attempt to assign a second parent, ensuring it's different from the first
                has_two_distinct_parents = z3.And(parent_condition1, parent_condition2) 
                opt.add(z3.Implies(has_two_distinct_parents, instance_parent2(node_index) == parent_index2))
          
          # Ensure instance_parent1 and instance_parent2 are equal if only one parent exists
          # this is probably redundant
          only_one_parent = parent_count == 1
          opt.add(z3.Implies(only_one_parent, instance_parent2(node_index) == instance_parent1(node_index)))

# Constraint: The instance nodes in every partition should form a linear chain
def add_intra_level_linear_chain():
  for (A_sub_matrix, idx_node_submap) in A_partition_submatrices_list.values():
    partition_node_ids = list(idx_node_submap.values())
    start_nodes = []
    end_nodes = []
    for node in partition_node_ids:
      start_nodes.append(z3.Bool(f"start_{node}"))
      end_nodes.append(z3.Bool(f"end_{node}"))
      rank[node] = z3.Int(f'rank_{node}')
    
    for i in range(len(partition_node_ids)):
      # Directly use sub-matrix to count incoming/outgoing edges for node i
      num_incoming_edges = z3.Sum([z3.If(A_sub_matrix[j, i], 1, 0) for j in range(len(partition_node_ids)) if j != i])
      num_outgoing_edges = z3.Sum([z3.If(A_sub_matrix[i, j], 1, 0) for j in range(len(partition_node_ids)) if j != i])
      
      opt.add(start_nodes[i] == (num_outgoing_edges == 1) & (num_incoming_edges == 0))
      opt.add(end_nodes[i] == (num_incoming_edges == 1) & (num_outgoing_edges == 0))
      opt.add((~start_nodes[i] & ~end_nodes[i]) == ((num_incoming_edges == 1) & (num_outgoing_edges == 1))) # ~, & are the logical operators in Z3
    
    # Ensure exactly one start node and one end node in the partition
    opt.add(z3.Sum([z3.If(start_node, 1, 0) for start_node in start_nodes]) == 1)
    opt.add(z3.Sum([z3.If(end_node, 1, 0) for end_node in end_nodes]) == 1)

    # Define relationships for linearly adjacent nodes
    for i in range(len(partition_node_ids)):
      for j in range(len(partition_node_ids)):
        if i != j:  # Avoid self-loops
          edge_i_to_j = A_sub_matrix[i, j]
          opt.add(z3.Implies(edge_i_to_j, succ(i) == j))
          opt.add(z3.Implies(edge_i_to_j, pred(j) == i))
          opt.add(z3.Implies(edge_i_to_j, rank(i) < rank(j)))

# Constraint: adjacent nodes in the intra-level linear chain should not have the same prototype
def add_unique_intra_level_consec_prototypes():
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