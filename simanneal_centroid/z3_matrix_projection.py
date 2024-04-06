import z3
import numpy as np 
import json
import re
import sys
import z3_matrix_projection_helpers as helpers 
import math 

centroid = np.loadtxt('centroid1.txt')
with open("centroid_node_mapping1.txt", 'r') as file:
  idx_node_mapping = json.load(file)
  idx_node_mapping = {int(k): v for k, v in idx_node_mapping.items()}

node_idx_mapping = {v: k for k, v in idx_node_mapping.items()}
n = len(idx_node_mapping) 
opt = z3.Solver()

levels_partition = helpers.partition_levels(idx_node_mapping)
max_seg_level = len(levels_partition.keys()) - 1

# Declare Z3 variables to enforce constraints on
# Create a matrix in Z3 for adjacency; A[i][j] == 1 means an edge from i to j
A = np.array([[z3.Bool(f"A_{i}_{j}") for j in range(n)] for i in range(n)])
A_partition_submatrices_list = helpers.create_partition_submatrices(A, idx_node_mapping, node_idx_mapping, levels_partition)

NodeSort = z3.IntSort()

# Uninterpreted functions
instance_parent1 = z3.Function('instance_parent1', NodeSort, NodeSort)
instance_parent2 = z3.Function('instance_parent2', NodeSort, NodeSort)
proto_parent = z3.Function('proto_parent', NodeSort, NodeSort)

# orderings for linear chain
# IMPORTANT: the nodes indices for these functions refer to the RELEVANT PARTITION SUBMATRIX, NOT the entire centroid matrix A!!!
pred = z3.Function('pred', NodeSort, NodeSort)
succ = z3.Function('succ', NodeSort, NodeSort)
start = z3.Function('start', z3.IntSort(), NodeSort) # level -> node index in relevant submatrix
end = z3.Function('end', z3.IntSort(), NodeSort)
rank = z3.Function('rank', NodeSort, z3.IntSort())

idx_node_mapping_prototype = {idx: node_id for idx, node_id in idx_node_mapping.items() if node_id.startswith("Pr")}
idx_node_mapping_instance = {idx: node_id for idx, node_id in idx_node_mapping.items() if not node_id.startswith("Pr")}

# Constraint: Every instance node must be the child of exactly one prototype node
def add_prototype_to_instance_constraints():
  num_prototype_nodes = len(idx_node_mapping_prototype)
  num_instance_nodes = len(idx_node_mapping_instance)
  for instance_i in range(num_instance_nodes):
    incoming_prototype_edges = z3.Sum([z3.If(A[proto_i][instance_i], 1, 0) for proto_i in range(num_prototype_nodes)])
    opt.add(incoming_prototype_edges == 1)
    for proto_i in range(num_prototype_nodes): # store the prototype parent info
      opt.add(z3.Implies(A[proto_i][instance_i], proto_parent(instance_i) == proto_i))

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
  for level, (A_sub_matrix, idx_node_submap) in A_partition_submatrices_list.items():
    partition_node_ids = list(idx_node_submap.values())
    num_partition_nodes = len(partition_node_ids)
    start_nodes = []
    end_nodes = []
    for node in partition_node_ids:
      start_nodes.append(z3.Bool(f"start_{node}"))
      end_nodes.append(z3.Bool(f"end_{node}"))
    
    for i in range(num_partition_nodes):
      # Directly use sub-matrix to count incoming/outgoing edges for node i
      num_incoming_edges = z3.Sum([z3.If(A_sub_matrix[j, i], 1, 0) for j in range(num_partition_nodes) if j != i])
      num_outgoing_edges = z3.Sum([z3.If(A_sub_matrix[i, j], 1, 0) for j in range(num_partition_nodes) if j != i])
      
      opt.add(start_nodes[i] == (num_outgoing_edges == 1) & (num_incoming_edges == 0))
      opt.add(end_nodes[i] == (num_incoming_edges == 1) & (num_outgoing_edges == 0))
      opt.add((~start_nodes[i] & ~end_nodes[i]) == ((num_incoming_edges == 1) & (num_outgoing_edges == 1))) # ~, & are the logical operators in Z3

      opt.add(z3.Implies(start_nodes[i], start(level) == i))
      opt.add(z3.Implies(end_nodes[i], end(level) == i))
    
    # Ensure exactly one start node and one end node in the partition
    opt.add(z3.Sum([z3.If(start_node, 1, 0) for start_node in start_nodes]) == 1)
    opt.add(z3.Sum([z3.If(end_node, 1, 0) for end_node in end_nodes]) == 1)

    # Define relationships for linearly adjacent nodes
    for i in range(num_partition_nodes):
      for j in range(num_partition_nodes):
        if i != j:  # Avoid self-loops
          edge_i_to_j = A_sub_matrix[i, j]
          opt.add(z3.Implies(edge_i_to_j, succ(i) == j))
          opt.add(z3.Implies(edge_i_to_j, pred(j) == i))
          opt.add(z3.Implies(edge_i_to_j, rank(i) < rank(j)))

# Constraint: adjacent nodes in the intra-level linear chain should not have the same prototype
def add_level_prototype_and_instance_parent_constraints():
  for level, (_, idx_node_submap) in A_partition_submatrices_list.items():
    for i, node_id in idx_node_submap.items():
      opt.add(z3.Implies(i != end(level), proto_parent(i) != proto_parent(succ(i)))) # no 2 linearly adjacent nodes can have the same prototype parent
      
      if level > 0:
        segment_level = re.match(r"S\d+L\d+N\d+", node_id)
        motif_level = re.match(r"P\d+O\d+N\d+", node_id)
        if segment_level:
          # rules for contiguous and total segmentation
          opt.add(z3.Implies(i != end(level), rank(instance_parent2(i)) <= rank(instance_parent1(succ(i))))) # each node's first parent must not come before the prev node's last parent
          opt.add(z3.Implies(i == end(level), instance_parent2(i) == end(level - 1))) # the final node must have the prev level's last node as a parent
          opt.add(z3.Implies(i == start(level), instance_parent1(i) == start(level - 1))) # the first node must have the prev level's first node as a parent
        elif motif_level:
          # rules for disjoint, non-contiguous sections
          opt.add(z3.Implies(i != end(level), rank(instance_parent1(i)) <= rank(instance_parent1(succ(i))))) # each node's first parent must not come before the prev node's first parent (since this is non-contiguous and we can have overlapping motifs)
        else:
          print("ERROR")
          sys.exit(0)

add_prototype_to_instance_constraints()
print("HERE1")
# add_inter_level_parent_counts_constraints()
# print("HERE2")
# add_intra_level_linear_chain()
# print("HERE3")
# add_level_prototype_and_instance_parent_constraints()
# print("HERE4")

objective = z3.Sum([z3.If(A[i][j] != bool(centroid[i][j]), 1, 0) for i in range(n) for j in range(n)])
# opt.minimize(objective)
print("HERE5")

# if opt.check() == z3.sat:
#   model = opt.model()
#   print("Closest valid graph's adjacency matrix:")
#   for i in range(n):
#     print([model.evaluate(A[i][j]) for j in range(n)])
# else:
#   print("Problem has no solution")

# Iteratively refine the solution
objective_value = math.inf
while True:
  if opt.check() == z3.sat:
      m = opt.model()
      current_objective_value = m.eval(objective, model_completion=True)
      print(f"Found solution with objective value: {current_objective_value}")
      # Update the best known objective value
      if current_objective_value.as_long() < objective_value:
          objective_value = current_objective_value

          # Add a constraint to find a better solution
          opt.add(objective < objective_value)
      else:
          # If no improvement, break from the loop
          break
  else:
      # If unsat, no further solutions can be found; break from the loop
      print("No more solutions found.")
      break
