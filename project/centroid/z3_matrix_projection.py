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
def add_prototype_to_instance_constraints():
  for level, (_, idx_node_submap) in A_partition_submatrices_list.items():
    for instance_idx_subA, instance_node_id in idx_node_submap.items():
      instance_idx_A = node_idx_mapping[instance_node_id]
      incoming_prototype_edges = z3.Sum([z3.If(A[proto_idx][instance_idx_A], 1, 0) for proto_idx in idx_node_mapping_prototype.keys()])
      opt.add(z3.Implies(is_not_dummy[instance_idx_A], incoming_prototype_edges == 1)) # each instance node has exactly 1 proto parent

      for proto_idx, proto_node_id, in idx_node_mapping_prototype.items():
        # level, index in the submatrix partition of instance nodes for that level -> index of proto parent w.r.t. the entire centroid matrix A
        opt.add(z3.Implies(A[proto_idx][instance_idx_A], proto_parent(level, instance_idx_subA) == proto_idx))
        
        # ensure no instance -> proto edges
        opt.add(A[instance_idx_A][proto_idx] == False) 

        proto_type = z3_helpers.get_node_type(proto_node_id) # ensure no invalid proto-instance connections
        instance_type = z3_helpers.get_node_type(instance_node_id)
        if ((proto_type == "SEG_PROTO" and instance_type == "MOTIF_INSTANCE") or 
            (proto_type == "MOTIF_PROTO" and instance_type == "SEG_INSTANCE")):
          opt.add(A[proto_idx][instance_idx_A] == False)

# Constraint: no edges between prototypes
def add_prototype_to_prototype_constraints():
  for proto_i in idx_node_mapping_prototype.keys():
    for proto_j in idx_node_mapping_prototype.keys():
        if proto_i != proto_j:  # Exclude self-loops
            opt.add(A[proto_i][proto_j] == False)

# Constraint: Every instance node not at the top level of the hierarchy, must have 1 or 2 parents in the level above it
def add_inter_level_parent_counts_constraints():
  for level, (_, idx_node_submap) in A_partition_submatrices_list.items():
    for i_subA, node_id in idx_node_submap.items():
      parsed = z3_helpers.parse_node_id(node_id)
      if parsed:
        i_A = node_idx_mapping[node_id]
        if level > 0: # top level doesn't have instance parents by construction in simanneal
          (_, prev_level_idx_node_submap) = A_partition_submatrices_list[level - 1]
          parent_count = z3.Sum([z3.If(A[node_idx_mapping[parent_id]][i_A], 1, 0) for parent_id in prev_level_idx_node_submap.values()])
          opt.add(z3.Or(parent_count == 1, parent_count == 2, z3.Not(is_not_dummy[i_A])))
          
          # Assign the 1 or 2 parents to non-zero level instance nodes for future reference in the constraint about parent orders based on the linear chain
          prev_level_node_info = list(prev_level_idx_node_submap.items())
          for list_idx, (parent_index1, parent_id1) in enumerate(prev_level_node_info):
            parent_condition1 = A[node_idx_mapping[parent_id1]][i_A]
            # if 1 parent, then instance_parent1 and instance_parent2 are the same
            opt.add(z3.Implies(z3.And(parent_count == 1, parent_condition1), z3.And(instance_parent1(i_subA) == parent_index1, instance_parent2(i_subA) == parent_index1))) 
            
            for parent_index2, parent_id2 in prev_level_node_info[list_idx+1:]: # ensure we're only looking at distinct tuples of parents, otherwise we are UNSAT
              parent_condition2 = A[node_idx_mapping[parent_id2]][i_A]
              opt.add(z3.Implies(z3.And(parent_condition1, parent_condition2, parent_count == 2), 
                                z3.And(instance_parent1(i_subA) == parent_index1, instance_parent2(i_subA) == parent_index2)))

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

# Constraint: adjacent nodes in the intra-level linear chain should not have the same prototype
def add_level_prototype_and_instance_parent_constraints():
  for level, (_, idx_node_submap) in A_partition_submatrices_list.items():
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
      
    for other_level, (_, other_idx_node_submap) in A_partition_submatrices_list.items():
      # No edges between non-adjacent levels, and no edges from level i to level i - 1
      if abs(level - other_level) > 1 or level - other_level == 1:
        for i_node_id in idx_node_submap.values():
          for j_node_id in other_idx_node_submap.values():
            i_in_A = node_idx_mapping[i_node_id]
            j_in_A = node_idx_mapping[j_node_id]
            opt.add(A[i_in_A][j_in_A] == False)


add_dummys_and_no_self_loops_constraint()
print("HERE1", time.perf_counter())
add_prototype_to_instance_constraints()
print("HERE2", time.perf_counter())
add_prototype_to_prototype_constraints()
print("HERE3", time.perf_counter())
add_inter_level_parent_counts_constraints()
print("HERE4", time.perf_counter())
add_intra_level_linear_chain()
print("HERE5", time.perf_counter())
add_level_prototype_and_instance_parent_constraints()
print("HERE6", time.perf_counter())

objective = z3.Sum([z3.If(A[i][j] != bool(centroid[i][j]), 1, 0) for i in range(n) for j in range(n)])
# opt.minimize(objective)
print("HERE7", time.perf_counter())

# if opt.check() == z3.sat:
#   model = opt.model()
#   print("HERE8", time.perf_counter())
#   print("Closest valid graph's adjacency matrix:", model)
#   for i in range(n):
#     result = np.array([[1 if model.evaluate(A[i, j]) else 0 for j in range(n)] for i in range(n)])
#     G = simanneal_helpers.adj_matrix_to_graph(centroid, idx_node_mapping)
#     g = simanneal_helpers.adj_matrix_to_graph(result, idx_node_mapping)
#     layers_G = build_graph.get_unsorted_layers_from_graph_by_index(G)
#     layers_g = build_graph.get_unsorted_layers_from_graph_by_index(g)
#     build_graph.visualize_p([G, g], [layers_G, layers_g])
# else:
#   print("Problem has no solution")

objective_value = math.inf
while True:
  if opt.check() == z3.sat:
      print("HERE8", time.perf_counter())
      m = opt.model()
      current_objective_value = m.eval(objective, model_completion=True)
      print(f"Found solution with objective value: {current_objective_value}")

      if current_objective_value.as_long() < objective_value:
          objective_value = current_objective_value
          opt.add(objective < objective_value)
      else:
          # If no improvement, break from the loop
          break
  else:
      # If unsat, no further solutions can be found; break from the loop
      print("No more solutions found.")
      break
