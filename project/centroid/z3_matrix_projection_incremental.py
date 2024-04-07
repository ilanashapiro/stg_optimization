import z3
import numpy as np 
import json
import re
import sys
import z3_matrix_projection_helpers as z3_helpers 
import z3_tests 
import simanneal_centroid_helpers as simanneal_helpers 
import math 
import networkx as nx
import sys
import time

sys.path.append("/Users/ilanashapiro/Documents/constraints_project/project")
import build_graph

centroid = np.loadtxt('centroid.txt')
with open("centroid_node_mapping.txt", 'r') as file:
  idx_node_mapping = json.load(file)
  idx_node_mapping = {int(k): v for k, v in idx_node_mapping.items()}

# G = z3_tests.G1
# centroid = nx.to_numpy_array(G)
# idx_node_mapping = {index: node for index, node in enumerate(G.nodes())}

node_idx_mapping = {v: k for k, v in idx_node_mapping.items()}
n = len(idx_node_mapping) 
opt = z3.Solver()

levels_partition = z3_helpers.partition_levels(idx_node_mapping)
max_seg_level = len(levels_partition.keys()) - 1

print("HERE0", time.perf_counter())

# Declare Z3 variables to enforce constraints on
# Create a matrix in Z3 for adjacency; A[i][j] == 1 means an edge from i to j
A = np.array([[z3.Bool(f"A_{i}_{j}") for j in range(n)] for i in range(n)])
A_partition_submatrices_list = z3_helpers.create_partition_submatrices(A, idx_node_mapping, node_idx_mapping, levels_partition)

NodeSort = z3.IntSort()
is_not_dummy = z3.BoolVector('is_not_dummy', n)

# Uninterpreted functions
instance_parent1 = z3.Function('instance_parent1', NodeSort, NodeSort)
instance_parent2 = z3.Function('instance_parent2', NodeSort, NodeSort)
proto_parent = z3.Function('proto_parent', NodeSort, NodeSort, NodeSort) # level number, index within level partition -> prototype index in entire centroid matrix A

# orderings for linear chain
# IMPORTANT: the nodes indices for these functions refer to the RELEVANT PARTITION SUBMATRIX, NOT the entire centroid matrix A!!!
pred = z3.Function('pred', NodeSort, NodeSort)
succ = z3.Function('succ', NodeSort, NodeSort)
start = z3.Function('start', z3.IntSort(), NodeSort) # level -> node index in relevant submatrix
end = z3.Function('end', z3.IntSort(), NodeSort)
rank = z3.Function('rank', NodeSort, z3.IntSort())

idx_node_mapping_prototype = {idx: node_id for idx, node_id in idx_node_mapping.items() if node_id.startswith("Pr")}
idx_node_mapping_instance = {idx: node_id for idx, node_id in idx_node_mapping.items() if not node_id.startswith("Pr")}

# Constraint: the graph can't have self loops, and set the dummys
def add_dummys_and_no_self_loops_constraint():
  for i in range(n):
    total_edges = z3.Sum([z3.If(A[i][j], 1, 0) for j in range(n) if j != i] + [z3.If(A[j][i], 1, 0) for j in range(n) if j != i])
    opt.add(is_not_dummy[i] == (total_edges != 0))
    opt.add(A[i][i] == False)

# Constraint: Every instance node must be the child of exactly one prototype node, no instance to proto edges, 
# and every proto->instance edge needs to be between nodes of the same type
def add_prototype_to_instance_constraints(level, idx_node_submap, valid_idx_node_mapping_prototype): # valid mean seg-seg, and motif-motif
  for instance_idx_subA, instance_node_id in idx_node_submap.items():
    instance_idx_A = node_idx_mapping[instance_node_id]
    incoming_prototype_edges = z3.Sum([z3.If(A[proto_idx][instance_idx_A], 1, 0) for proto_idx in valid_idx_node_mapping_prototype.keys()])
    opt.add(z3.Implies(is_not_dummy[instance_idx_A], incoming_prototype_edges == 1)) # each instance node has exactly 1 proto parent

    for proto_idx, proto_node_id, in valid_idx_node_mapping_prototype.items():
      # level, index in the submatrix partition of instance nodes for that level -> index of proto parent w.r.t. the entire centroid matrix A
      opt.add(z3.Implies(A[proto_idx][instance_idx_A], proto_parent(level, instance_idx_subA) == proto_idx))
      
      # ensure no instance -> proto edges
      opt.add(A[instance_idx_A][proto_idx] == False) 

      # ensure no invalid proto-instance connections --> NOW ATTEMPTING BY CONSTRUCTION W INCREMENTAL SOLVING


# Constraint: no edges between prototypes
def add_prototype_to_prototype_constraints():
  for proto_i in idx_node_mapping_prototype.keys():
    for proto_j in idx_node_mapping_prototype.keys():
        if proto_i != proto_j:  # Exclude self-loops
            opt.add(A[proto_i][proto_j] == False)

# Constraint: Every instance node not at the top level of the hierarchy, must have 1 or 2 parents in the level above it
# idx_node_submap1 is a level above idx_node_submap2
# NEED TO FIX THIS: A_sub_matrix is only for a SINGLE LEVEL -- i want the matrix for the PAIRED LEVELS
def add_adj_level_parent_counts_constraints(A_sub_matrix, idx_node_submap1, idx_node_submap2):
  for i_subA, node_id in idx_node_submap2.items():
    parsed = z3_helpers.parse_node_id(node_id)
    if parsed:
      i_A = node_idx_mapping[node_id]

      parent_count = z3.Sum([z3.If(A_sub_matrix[parent_submap_index][i_subA], 1, 0) for parent_submap_index in idx_node_submap1.keys()])
      opt.add(z3.Or(parent_count == 1, parent_count == 2, z3.Not(is_not_dummy[i_A])))
      
      # Assign the 1 or 2 parents to non-zero level instance nodes for future reference in the constraint about parent orders based on the linear chain
      prev_level_node_info = list(idx_node_submap1.items())
      for list_idx, (parent_index1_subA, parent_id1) in enumerate(prev_level_node_info):
        parent_index1_A = node_idx_mapping[parent_id1]
        opt.add(A_sub_matrix[i_subA][parent_index1_subA] == False) # cannot have edge from level i to level i - 1, children must be on level below

        parent_condition1 = A_sub_matrix[parent_index1_subA][i_subA]
        # if 1 parent, then instance_parent1 and instance_parent2 are the same
        opt.add(z3.Implies(z3.And(parent_count == 1, parent_condition1), z3.And(instance_parent1(i_subA) == parent_index1_subA, instance_parent2(i_subA) == parent_index1_subA))) 
        
        for parent_index2_subA, parent_id2 in prev_level_node_info[list_idx+1:]: # ensure we're only looking at distinct tuples of parents, otherwise we are UNSAT
          parent_condition2 = A_sub_matrix[parent_index1_subA][i_subA]
          opt.add(z3.Implies(z3.And(parent_condition1, parent_condition2, parent_count == 2), 
                            z3.And(instance_parent1(i_subA) == parent_index1_subA, instance_parent2(i_subA) == parent_index2_subA)))

# Constraint: The instance nodes in the given partition should form a linear chain
def add_intra_level_linear_chain(level, A_sub_matrix, idx_node_submap):
  partition_node_ids = list(idx_node_submap.values())
  num_partition_nodes = len(partition_node_ids)
  start_nodes = []
  end_nodes = []
  for node in partition_node_ids:
    start_nodes.append(z3.Bool(f"start_{node}"))
    end_nodes.append(z3.Bool(f"end_{node}"))
  
  for i_subA, node_id in idx_node_submap.items():
    # Directly use sub-matrix to count incoming/outgoing edges for node i
    num_incoming_edges = z3.Sum([z3.If(A_sub_matrix[j, i_subA], 1, 0) for j in range(len(partition_node_ids)) if j != i_subA])
    num_outgoing_edges = z3.Sum([z3.If(A_sub_matrix[i_subA, j], 1, 0) for j in range(len(partition_node_ids)) if j != i_subA])
    
    opt.add(start_nodes[i_subA] == z3.And(num_outgoing_edges == 1, num_incoming_edges == 0))
    opt.add(end_nodes[i_subA] == z3.And(num_incoming_edges == 1, num_outgoing_edges == 0))
    i_A = node_idx_mapping[node_id]
    is_intermediate_chain_node = z3.And(z3.Not(start_nodes[i_subA]), z3.Not(end_nodes[i_subA]), is_not_dummy[i_A])
    opt.add(is_intermediate_chain_node == ((num_incoming_edges == 1) & (num_outgoing_edges == 1)))

    opt.add(z3.Implies(start_nodes[i_subA], start(level) == i_subA))
    opt.add(z3.Implies(end_nodes[i_subA], end(level) == i_subA))
  
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

# Constraint: segmentation: each node's first parent must not come before the prev node's last parent in the chain, and start and end must align to start and end above 
#             motif: node's first parent must not come before the prev node's first parent (since this is non-contiguous and we can have overlapping motifs)
def add_level_prototype_and_instance_parent_constraints(level, idx_node_submap):
  for i_subA, node_id in idx_node_submap.items(): 
    i_A = node_idx_mapping[node_id]
    opt.add(z3.Implies(z3.And(i_subA != end(level)), proto_parent(level, i_subA) != proto_parent(level, succ(i_subA)))) # no 2 linearly adjacent nodes can have the same prototype parent
    if level > 0:
      segment_level = re.match(r"S\d+L\d+N\d+", node_id)
      motif_level = re.match(r"P\d+O\d+N\d+", node_id)
      if segment_level:
        # rules for contiguous and total segmentation
        opt.add(z3.Implies(z3.And(is_not_dummy[i_A], i_subA != end(level)), rank(instance_parent2(i_subA)) <= rank(instance_parent1(succ(i_subA))))) # each node's first parent must not come before the prev node's last parent
        opt.add(z3.Implies(z3.And(is_not_dummy[i_A], i_subA == end(level)), instance_parent2(i_subA) == end(level - 1))) # the final node must have the prev level's last node as a parent
        opt.add(z3.Implies(z3.And(is_not_dummy[i_A], i_subA == start(level)), instance_parent1(i_subA) == start(level - 1))) # the first node must have the prev level's first node as a parent
      elif motif_level:
        # rules for disjoint, non-contiguous sections
        opt.add(z3.Implies(z3.And(is_not_dummy[i_A], i_subA != end(level)), rank(instance_parent1(i_subA)) <= rank(instance_parent1(succ(i_subA))))) # each node's first parent must not come before the prev node's first parent (since this is non-contiguous and we can have overlapping motifs)
      else:
        print("ERROR")
        sys.exit(0)

# this still looks at the entire matrix ..... 
# def add_inter_level_edge_violation_constraints(level, idx_node_submap):
  # for other_level, (_, other_idx_node_submap) in A_partition_submatrices_list.items():
  #   # No edges between non-adjacent levels. No edges from level i to level i - 1 is now in add_prototype_to_instance_constraints
  #   if abs(level - other_level) > 1:
  #     for i_node_id in idx_node_submap.values():
  #       for j_node_id in other_idx_node_submap.values():
  #         i_in_A = node_idx_mapping[i_node_id]
  #         j_in_A = node_idx_mapping[j_node_id]
  #         opt.add(A[i_in_A][j_in_A] == False)

# objective = z3.Sum([z3.If(A[i][j] != bool(centroid[i][j]), 1, 0) for i in range(n) for j in range(n)])
# opt.minimize(objective)

for level, (A_sub_matrix, idx_node_submap) in A_partition_submatrices_list.items():
  print(f"LEVEL {level}", time.perf_counter())
  opt.push()  # Save the current optimizer state for potential backtracking
  
  if level > 0:
    (prev_A_sub_matrix, prev_idx_node_submap) = A_partition_submatrices_list[level-1]
    add_intra_level_linear_chain(level, A_sub_matrix, idx_node_submap)
    print(f"add_intra_level_linear_chain", time.perf_counter())
    add_level_prototype_and_instance_parent_constraints(level, idx_node_submap)
    print(f"add_level_prototype_and_instance_parent_constraints", time.perf_counter())
    add_adj_level_parent_counts_constraints(idx_node_submap, prev_idx_node_submap)
    print(f"add_adj_level_parent_counts_constraints", time.perf_counter())
    
    if opt.check() == z3.sat:
        print(f"Consecutive levels {level - 1} and {level} are satisfiable", time.perf_counter())
        # At this point, instead of printing the model, you could extract and save the satisfying conditions/constraints
        
        # Do not pop the state here as we want to keep building upon the added constraints
  opt.pop()

# After iterating through all levels, you can check for overall satisfiability
if opt.check() == z3.sat:
    print("Final structure across all levels is satisfiable")
    final_model = opt.model()
    # Here, final_model contains the optimal structure across all levels
else:
    print("Unable to find a satisfiable structure across all levels")

