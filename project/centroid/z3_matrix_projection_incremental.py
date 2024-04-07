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

# centroid = np.loadtxt('centroid.txt')
# with open("centroid_node_mapping.txt", 'r') as file:
# 	idx_node_mapping = json.load(file)
# 	idx_node_mapping = {int(k): v for k, v in idx_node_mapping.items()}

G = z3_tests.G1
centroid = nx.to_numpy_array(G)
idx_node_mapping = {index: node for index, node in enumerate(G.nodes())}

def invert_dict(d):
	return {v: k for k, v in d.items()}

node_idx_mapping = invert_dict(idx_node_mapping)
n_A = len(idx_node_mapping) 
opt = z3.Solver()

instance_levels_partition = z3_helpers.partition_instance_levels(idx_node_mapping) # dict: level -> instance nodes at that level
prototype_kinds_partition = z3_helpers.partition_prototype_kinds(idx_node_mapping) # dict: prototype kind -> prototype nodes of that kind
max_seg_level = len(instance_levels_partition.keys()) - 1

print("HERE0", time.perf_counter())

# Declare Z3 variables to enforce constraints on
# Create a matrix in Z3 for adjacency; A[i][j] == 1 means an edge from i to j
A = np.array([[z3.Bool(f"A_{i}_{j}") for j in range(n_A)] for i in range(n_A)])
A_partition_instance_submatrices_list = z3_helpers.create_instance_partition_submatrices(A, node_idx_mapping, instance_levels_partition)
A_partition_instance_submatrices_list_with_proto = z3_helpers.create_instance_proto_partition_submatrices(A, node_idx_mapping, instance_levels_partition, prototype_kinds_partition)
A_adj_instance_submatrices_list = z3_helpers.create_adjacent_level_instance_partition_submatrices(A, node_idx_mapping, instance_levels_partition)
A_adj_instance_submatrices_list_with_proto = z3_helpers.create_adjacent_level_proto_and_instance_partition_submatrices(A, node_idx_mapping, instance_levels_partition, prototype_kinds_partition)

NodeSort = z3.IntSort()
is_not_dummy = z3.BoolVector('is_not_dummy', n_A)

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

# Constraint: the graph can't have self loops, and set the dummys, for instances
# submatrix_with_proto is the submatrix for a single instance level, with the possible prototypes (these are the requirements to check for dummys)
def add_dummys_and_no_self_loops_constraint_instances(submatrix_with_proto, idx_node_submap):
	n = len(submatrix_with_proto)
	for i in range(n):
		if (z3_helpers.is_instance(idx_node_submap[i])):
			total_edges = z3.Sum([z3.If(submatrix_with_proto[i][j], 1, 0) for j in range(n) if j != i] + [z3.If(submatrix_with_proto[j][i], 1, 0) for j in range(n) if j != i])
			opt.add(is_not_dummy[i] == (total_edges != 0))
			opt.add(submatrix_with_proto[i][i] == False)

# Constraint: the graph can't have self loops, and set the dummys, for instances
# submatrix_with_proto is the submatrix for a single instance level, with the possible prototypes (these are the requirements to check for dummys)
def add_dummys_and_no_self_loops_constraint_protos(submatrix_with_proto, idx_node_submap):
	n = submatrix_with_proto.size[0]
	for i in range(n):
		if (z3_helpers.is_proto(idx_node_submap[i])):
			total_edges = z3.Sum([z3.If(submatrix_with_proto[i][j], 1, 0) for j in range(n) if j != i] + [z3.If(submatrix_with_proto[j][i], 1, 0) for j in range(n) if j != i])
			opt.add(is_not_dummy[i] == (total_edges != 0))
			opt.add(submatrix_with_proto[i][i] == False)

# Constraint: the graph can't have self loops, and set the dummys, for both node kinds (instances and proto)
# submatrix_with_proto is the submatrix for a single instance level, with the possible prototypes (these are the requirements to check for dummys)
def add_dummys_and_no_self_loops_constraint(submatrix_with_proto):
	n = submatrix_with_proto.size[0]
	for i in range(n):
		total_edges = z3.Sum([z3.If(submatrix_with_proto[i][j], 1, 0) for j in range(n) if j != i] + [z3.If(submatrix_with_proto[j][i], 1, 0) for j in range(n) if j != i])
		opt.add(is_not_dummy[i] == (total_edges != 0))
		opt.add(submatrix_with_proto[i][i] == False)

# Constraint: Every instance node must be the child of exactly one prototype node, no instance to proto edges, 
# (every proto->instance edge needs to be between nodes of the same type but this is implicit bc of the staged computation)
# submatrix consists of a single level with the possible prototypes for that level kind
# ------------- NEED TO FIX
# def add_prototype_to_instance_constraints(level, submatrix, idx_node_submap, valid_proto_ids): # valid mean seg-seg, and motif-motif
# 	node_idx_submap = invert_dict(idx_node_submap)
# 	for instance_idx, instance_node_id in idx_node_submap.items():
# 		incoming_prototype_edges = z3.Sum([z3.If(submatrix[proto_idx][instance_idx_A], 1, 0) for proto_id in valid_proto_ids])
# 		opt.add(z3.Implies(is_not_dummy[instance_idx_A], incoming_prototype_edges == 1)) # each instance node has exactly 1 proto parent

# 		for proto_idx, proto_node_id, in valid_idx_node_mapping_prototype.items():
# 			# level, index in the submatrix partition of instance nodes for that level -> index of proto parent w.r.t. the entire centroid matrix A
# 			opt.add(z3.Implies(submatrix[proto_idx][instance_idx_A], proto_parent(level, instance_idx_subA) == proto_idx))
			
# 			# ensure no instance -> proto edges
# 			opt.add(submatrix[instance_idx_A][proto_idx] == False) 

# 			# ensure no invalid proto-instance connections --> REMOVING FOR NOW TO SEE IF I CAN DO W INCREMENTAL SOLVING
# 			# proto_type = z3_helpers.get_node_type(proto_node_id) # ensure no invalid proto-instance connections
# 			# instance_type = z3_helpers.get_node_type(instance_node_id)
# 			# if ((proto_type == "SEG_PROTO" and instance_type == "MOTIF_INSTANCE") or 
# 			# 		(proto_type == "MOTIF_PROTO" and instance_type == "SEG_INSTANCE")):
# 			# 	opt.add(A[proto_idx][instance_idx_A] == False)

# Constraint: no edges between prototypes
def add_prototype_to_prototype_constraints(proto_submatrix):
	for proto_i in proto_submatrix:
		for proto_j in proto_submatrix:
				if proto_i != proto_j:  # Exclude self-loops
						opt.add(A[proto_i][proto_j] == False)

# Constraint: Every instance node not at the top level of the hierarchy, must have 1 or 2 parents in the level above it
# idx_node_submap1 is a level above idx_node_submap2
# A_sub_matrix1, idx_node_submap1 are the parents of/level above A_sub_matrix2, idx_node_submap2
def add_adj_level_parent_counts_constraints(combined_submatrix, idx_node_submap1, idx_node_submap2, combined_idx_node_submap, combined_idx_node_submap_with_proto):
	combined_node_idx_submap = invert_dict(combined_idx_node_submap)
	combined_node_idx_submap_with_proto = invert_dict(combined_idx_node_submap_with_proto)
	
	for i_subA2, node_id2 in idx_node_submap2.items():
		i_combined = combined_node_idx_submap[node_id2]

		parent_count = z3.Sum([z3.If(combined_submatrix[combined_node_idx_submap[node_id1]][i_combined], 1, 0) 
														for node_id1
														in idx_node_submap1.values()])
		parent_is_not_dummy = is_not_dummy[combined_node_idx_submap_with_proto[node_id2]] # using the matrix augmented with prototypes since this is the full context that determines if a node is a dummy
		opt.add(z3.And(parent_is_not_dummy, z3.Or(parent_count == 1, parent_count == 2)))

		# Assign the 1 or 2 parents to non-zero level instance nodes for future reference in the constraint about parent orders based on the linear chain
		for list_idx, (parent_index1_subA1, parent_id1) in enumerate(list(idx_node_submap1.items())):
			parent_index1_combined = combined_node_idx_submap[parent_id1]
			opt.add(combined_submatrix[i_combined][parent_index1_combined] == False) # cannot have edge from level i to level i - 1, children must be on level below

			parent_condition1 = combined_submatrix[parent_index1_combined][i_combined]
			# if 1 parent, then instance_parent1 and instance_parent2 are the same
			opt.add(z3.Implies(z3.And(parent_count == 1, parent_condition1), z3.And(instance_parent1(i_subA2) == parent_index1_subA1, instance_parent2(i_subA2) == parent_index1_subA1))) 
			
			for parent_index2_subA1, parent_id2 in list(idx_node_submap1.items())[list_idx+1:]: # ensure we're only looking at distinct tuples of parents, otherwise we are UNSAT
				parent_index2_combined = combined_node_idx_submap[parent_id2]
				parent_condition2 = combined_submatrix[parent_index2_combined][i_combined]
				opt.add(z3.Implies(z3.And(parent_condition1, parent_condition2, parent_count == 2), 
													z3.And(instance_parent1(i_subA2) == parent_index1_subA1, instance_parent2(i_subA2) == parent_index2_subA1)))

# Constraint: The instance nodes in the given partition should form a linear chain
def add_intra_level_linear_chain(level, A_sub_matrix, idx_node_submap, idx_node_submap_with_proto):
	node_idx_submap_with_proto = invert_dict(idx_node_submap_with_proto)
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

		is_not_dummy_node = is_not_dummy[node_idx_submap_with_proto[node_id]]
		is_intermediate_chain_node = z3.And(z3.Not(start_nodes[i_subA]), z3.Not(end_nodes[i_subA]), is_not_dummy_node)
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
def add_instance_parent_constraints(level, idx_node_submap, idx_node_submap_with_proto):
	node_idx_submap_with_proto = invert_dict(idx_node_submap_with_proto)
	for i_subA, node_id in idx_node_submap.items(): 
		# opt.add(z3.Implies(z3.And(i_subA != end(level)), proto_parent(level, i_subA) != proto_parent(level, succ(i_subA)))) # no 2 linearly adjacent nodes can have the same prototype parent
		if level > 0:
			segment_level = re.match(r"S\d+L\d+N\d+", node_id)
			motif_level = re.match(r"P\d+O\d+N\d+", node_id)
			is_not_dummy_node = is_not_dummy[node_idx_submap_with_proto[node_id]]
			if segment_level:
				# rules for contiguous and total segmentation
				opt.add(z3.Implies(z3.And(is_not_dummy_node, i_subA != end(level)), rank(instance_parent2(i_subA)) <= rank(instance_parent1(succ(i_subA))))) # each node's first parent must not come before the prev node's last parent
				opt.add(z3.Implies(z3.And(is_not_dummy_node, i_subA == end(level)), instance_parent2(i_subA) == end(level - 1))) # the final node must have the prev level's last node as a parent
				opt.add(z3.Implies(z3.And(is_not_dummy_node, i_subA == start(level)), instance_parent1(i_subA) == start(level - 1))) # the first node must have the prev level's first node as a parent
			elif motif_level:
				# rules for disjoint, non-contiguous sections
				opt.add(z3.Implies(z3.And(is_not_dummy_node, i_subA != end(level)), rank(instance_parent1(i_subA)) <= rank(instance_parent1(succ(i_subA))))) # each node's first parent must not come before the prev node's first parent (since this is non-contiguous and we can have overlapping motifs)
			else:
				print("ERROR")
				sys.exit(0)

def add_objective(submatrix, idx_node_submap):
	objective = z3.Sum([z3.If(submatrix[i][j] != bool(centroid[node_idx_mapping[node_id1]][node_idx_mapping[node_id2]]), 1, 0) 
										 for (i, node_id1) in idx_node_submap.items()
										 for (j, node_id2) in idx_node_submap.items()])
	opt.minimize(objective)

# save state ONLY for that level, and possible prototypes (if any)
# the submatrix will often contains a pair of adjacent levels -- we're only interested in the relevant level
def save_level_state(level, submatrix, idx_node_mapping, model):
	state = []
	for i in range(submatrix.shape[0]):
		node_id = idx_node_mapping[i]
		for j in range(submatrix.shape[1]):
			var = submatrix[i, j]  # Access the Z3 variable at this position in the submatri
			instance_node_info = z3_helpers.parse_instance_node_id(node_id)
			if instance_node_info:
				type = instance_node_info[0]
				node_level = instance_node_info[2] if type == 'S' else max_seg_level + 1

				is_instance_at_level = instance_node_info and node_level - 1 == level # since the level in the node-id is 1-indexed
				if is_instance_at_level or z3_helpers.is_proto(node_id):
					evaluated_var = model.eval(var, model_completion=True)  # Evaluate this variable in the model
					state.append((var, evaluated_var))
	return state

def restore_level_state(level, level_states):
	for var, evaluated_var in level_states[level]:
		opt.add(var == evaluated_var)

level_states = {}
# parent_level < next_level, meaning parent_level is HIGHER in the hierarchy than next_level (i.e. parent level of next_level)
for (parent_level, child_level), (A_combined_submatrix, combined_idx_node_submap) in A_adj_instance_submatrices_list.items():
	print(f"LEVELS {parent_level, child_level}", time.perf_counter())
	opt.push()  # Save the current optimizer state for potential backtracking

	if parent_level in level_states:
		restore_level_state(parent_level, level_states)
	if child_level in level_states:
		restore_level_state(child_level, level_states)

	(A_submatrix1, idx_node_submap1) = A_partition_instance_submatrices_list[parent_level]
	(A_submatrix2, idx_node_submap2) = A_partition_instance_submatrices_list[child_level]

	(A_submatrix1_with_proto, idx_node_submap1_with_proto) = A_partition_instance_submatrices_list_with_proto[parent_level]
	(A_submatrix2_with_proto, idx_node_submap2_with_proto) = A_partition_instance_submatrices_list_with_proto[child_level]
	(_, combined_idx_node_submap_with_proto) = A_adj_instance_submatrices_list_with_proto[(parent_level, child_level)]

	add_dummys_and_no_self_loops_constraint_instances(A_submatrix1_with_proto, idx_node_submap1_with_proto)
	add_dummys_and_no_self_loops_constraint_instances(A_submatrix2_with_proto, idx_node_submap2_with_proto)
	print("HERE1", time.perf_counter())

	# add_prototype_to_instance_constraints()
	# add_prototype_to_prototype_constraints()

	add_intra_level_linear_chain(parent_level, A_submatrix1, idx_node_submap1, idx_node_submap1_with_proto)
	add_intra_level_linear_chain(child_level, A_submatrix2, idx_node_submap2, idx_node_submap2_with_proto)
	print("HERE2", time.perf_counter())

	# add_instance_parent_constraints(parent_level, idx_node_submap1, idx_node_submap1_with_proto)
	# add_instance_parent_constraints(child_level, idx_node_submap2, idx_node_submap2_with_proto)
	# print("HERE3", time.perf_counter())

	# add_adj_level_parent_counts_constraints(A_combined_submatrix, idx_node_submap1, idx_node_submap2, combined_idx_node_submap, combined_idx_node_submap_with_proto)
	# print("HERE4", time.perf_counter())

	if opt.check() == z3.sat:
		print(f"Consecutive levels {parent_level} and {child_level} are satisfiable", time.perf_counter())
		model = opt.model()
		level_states[parent_level] = save_level_state(parent_level, A_submatrix1, idx_node_submap1, model)
		level_states[child_level] = save_level_state(child_level, A_submatrix2, idx_node_submap2, model)
		print((child_level, parent_level), level_states)
	else:
		print(f"Consecutive levels {parent_level} and {child_level} are not satisfiable")
	
	opt.pop()

print(level_states)
# After iterating through all levels, you can check for overall satisfiability
for level in instance_levels_partition.keys():
	print("LEVEL", level, level_states[level])
	restore_level_state(level, level_states)

if opt.check() == z3.sat:
	print("Final structure across all levels is satisfiable")
	final_model = opt.model()
	print(A)
	result = np.array([[1 if model.eval(A[i, j], model_completion=True) else 0 for j in range(n_A)] for i in range(n_A)])
	print(result, idx_node_mapping)
	G = simanneal_helpers.adj_matrix_to_graph(centroid, idx_node_mapping)
	g = simanneal_helpers.adj_matrix_to_graph(result, idx_node_mapping)
	layers_G = build_graph.get_unsorted_layers_from_graph_by_index(G)
	layers_g = build_graph.get_unsorted_layers_from_graph_by_index(g)
	build_graph.visualize_p([G, g], [layers_G, layers_g])
else:
		print("Unable to find a satisfiable structure across all levels")

