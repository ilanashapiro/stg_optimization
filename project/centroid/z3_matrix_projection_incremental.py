import z3
import numpy as np 
import json
import re
import sys, os
import z3_matrix_projection_helpers as z3_helpers 
import simanneal_centroid_helpers as simanneal_helpers 
# import simanneal_centroid_tests as simanneal_tests
import math 
import networkx as nx
import sys
import time
import pickle

# DIRECTORY = "/home/ilshapiro/project"
DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"

sys.path.append(DIRECTORY)
import build_graph

# ------------------------------------ Globals ------------------------------------

# inputs to the program -- MUST BE DEFINED
approx_centroid = None
idx_node_mapping = None
node_metadata_dict = None

node_idx_mapping = None
n_A = None
opt = None
rank_to_flat_levels_mapping = None
instance_levels_partition = None
prototype_features_partition = None

# Z3 variables and sorts
A = None
A_partition_instance_submatrices_list = None
A_partition_instance_submatrices_list_with_proto = None
A_adjacent_instance_submatrices_list = None
NodeSort = None
NodeSetSort = None

# Uninterpreted functions
instance_parent1 = None
instance_parent2 = None
proto_parents = None
succ = None
start = None
end = None
rank = None

# Dicts
idx_node_mapping_prototype = None
idx_node_mapping_instance = None

def initialize_globals(approx_centroid_val, idx_node_mapping_val, node_metadata_dict_val):
	global approx_centroid, idx_node_mapping, node_metadata_dict
	global node_idx_mapping, n_A, opt, rank_to_flat_levels_mapping
	global instance_levels_partition, prototype_features_partition, A
	global A_partition_instance_submatrices_list, A_partition_instance_submatrices_list_with_proto
	global A_adjacent_instance_submatrices_list, NodeSort, NodeSetSort
	global instance_parent1, instance_parent2, proto_parents, succ, start, end, rank
	global idx_node_mapping_prototype, idx_node_mapping_instance

	approx_centroid = approx_centroid_val
	idx_node_mapping = idx_node_mapping_val
	node_metadata_dict = node_metadata_dict_val

	node_idx_mapping = z3_helpers.invert_dict(idx_node_mapping)
	n_A = len(idx_node_mapping)
	opt = z3.Optimize()
	opt.set('timeout', 10000) # in milliseconds. 300000ms = 5mins
	opt.set("enable_lns", True)

	rank_to_flat_levels_mapping = z3_helpers.get_flat_levels_mapping(node_metadata_dict)
	instance_levels_partition = z3_helpers.partition_instance_levels(idx_node_mapping, node_metadata_dict, rank_to_flat_levels_mapping) # dict: level -> instance nodes at that level
	prototype_features_partition = z3_helpers.partition_prototype_features(idx_node_mapping, node_metadata_dict)  # dict: prototype feature -> prototype nodes of that feature

	# Declare Z3 variables to enforce constraints on
	A = np.array([[z3.Bool(f"A_{i}_{j}") for j in range(n_A)] for i in range(n_A)]) # Create a matrix in Z3 for adjacency; A[i][j] == 1 means an edge from i to j
	A_partition_instance_submatrices_list = z3_helpers.create_instance_partition_submatrices(A, node_idx_mapping, instance_levels_partition)
	A_partition_instance_submatrices_list_with_proto = z3_helpers.create_instance_with_proto_partition_submatrices(A, node_idx_mapping, instance_levels_partition, prototype_features_partition, node_metadata_dict)
	A_adjacent_instance_submatrices_list = z3_helpers.create_adjacent_level_instance_partition_submatrices(A, node_idx_mapping, instance_levels_partition)
	# A_partition_instance_submatrices_list_with_context = z3_helpers.create_level_partition_submatrices_with_context(A, node_idx_mapping, instance_levels_partition, prototype_kinds_partition) # USED FOR DUMMYS
	# A_adjacent_partition_submatrices_with_context = z3_helpers.create_adjacent_level_partition_submatrices_with_context(A, node_idx_mapping, instance_levels_partition, prototype_kinds_partition) # USED FOR DUMMYS

	NodeSort = z3.IntSort()
	NodeSetSort = z3.SetSort(NodeSort)

	# Uninterpreted functions
	instance_parent1 = z3.Function('instance_parent1', NodeSort, NodeSort)
	# def instance_parent1(child_i):
	# 	assert(isinstance(child_i, int))
	# 	return z3.Const(f"instance_parent1{child_i}", NodeSort)
	instance_parent2 = z3.Function('instance_parent2', NodeSort, NodeSort) # index within CHILD level partition -> prototype index in PARENT level partition
	proto_parents = z3.Function('proto_parents', NodeSort, NodeSetSort) # index within CHILD level partition -> prototype index in PARENT level partition

	# try to reformulate the problem in a PURELY FINITE DOMAIN (i.e. w.o. uninterpreted functions)
	# orderings for linear chain
	# IMPORTANT: the nodes indices for these functions refer to the RELEVANT PARTITION SUBMATRIX, NOT the entire centroid matrix A!!!
	succ = z3.Function('succ', NodeSort, NodeSort)
	start = z3.Function('start', z3.IntSort(), NodeSort) # flat level -> node index in relevant submatrix
	end = z3.Function('end', z3.IntSort(), NodeSort) # flat -> node index in relevant submatrix
	rank = z3.Function('rank', NodeSort, z3.IntSort())

	idx_node_mapping_prototype = {idx: node_id for idx, node_id in idx_node_mapping.items() if node_id.startswith("Pr")}
	idx_node_mapping_instance = {idx: node_id for idx, node_id in idx_node_mapping.items() if not node_id.startswith("Pr")}

	print("done initializing")

# Constraint: the graph can't have self loops, and set the dummys, for INSTANCES (I realize never need to check if a prototype is a dummy)
# submatrix_with_context is the submatrix for a single instance level, with context (these are the requirements to check for dummys)
# context is: nodes from the level below if not the last level, nodes from the level above if not the first level, and the prototype nodes of the same kind as the CURRENT level nodes
# NO LONGER DOING VERSION WITH DUMMYS DUE TO TIMEOUT
# def add_dummys_and_no_self_loops_constraint_instances(submatrix_with_context, idx_node_submap):
# 	n = len(submatrix_with_context)
# 	for i in range(n):
# 		if (z3_helpers.is_instance(idx_node_submap[i])):
# 			total_edges = z3.Sum([z3.If(submatrix_with_context[i][j], 1, 0) for j in range(n) if j != i] + [z3.If(submatrix_with_context[j][i], 1, 0) for j in range(n) if j != i])
# 			opt.add(is_not_dummy[i] == (total_edges != 0))
# 			opt.add(submatrix_with_context[i][i] == False) # the self loops is actually probably redundant if we're using the optimizer

# Constraint: no edges between prototypes
def add_prototype_to_prototype_constraints(idx_node_submap):
	for node_id1 in idx_node_submap.values():
		if z3_helpers.is_proto(node_id1):
			for node_id2 in idx_node_submap.values():
				if z3_helpers.is_proto(node_id2) and node_id1 != node_id2: # Exclude self-loops
						opt.add(A[node_idx_mapping[node_id1]][node_idx_mapping[node_id2]] == False)

# Constraint: Every instance node must be the child of exactly one prototype node, no instance to proto edges, 
# (every proto->instance edge needs to be between nodes of the same type but this is implicit bc of the staged computation)
# submatrix consists of a single level with the possible prototypes for that level kind
def add_prototype_to_instance_constraints(level, instance_proto_submatrix, idx_node_submap_instance_proto, node_metadata_dict):
	node_idx_submap_instance_proto = z3_helpers.invert_dict(idx_node_submap_instance_proto)
	(instance_only_submatrix, idx_node_submap_instance_only) = A_partition_instance_submatrices_list[level]
	define_linearly_adjacent_instance_relations(level, instance_only_submatrix)

	level_var = z3.Int(f"level:{level}")
	no_consecutive_repeat_layers = ['S', 'K', 'C'] # layers who can't have consecutive nodes be equivalent in the linear chain
	for instance_submap_index, instance_node_id in idx_node_submap_instance_only.items(): # instance_submap_index is wrt instance_only_submatrix
		instance_index = node_idx_submap_instance_proto[instance_node_id] # instance_index is wrt instance_proto_submatrix
		layer_id = z3_helpers.get_layer_id(instance_node_id)
		instance_features = node_metadata_dict[instance_node_id]['features_dict']
		instance_submap_index_var = z3.Int(f"level:{level}_i_subA:{instance_submap_index}")
		
		non_consec_feature_conditions = [] # one condition per feature. each condition in the list says curr node and next node in linear chain must not have the same proto parent node FOR THAT FEATURE 
		for instance_feature in instance_features:
			proto_node_ids = prototype_features_partition[instance_feature]
			proto_node_indices_for_feature = list(map(lambda proto_node_id: node_idx_submap_instance_proto[proto_node_id], proto_node_ids)) # list of proto indices wrt idx_submap_instance_proto, of that feature
			
			# constraint: we want one prototype per instance feature, so the num incoming proto edges per instance feature should be exactly one from the associated proto feature nodes set
			num_incoming_prototype_edges_for_feature = z3.Sum([z3.If(instance_proto_submatrix[proto_index][instance_index], 1, 0) for proto_index in proto_node_indices_for_feature]) # count the proto->instance edges for this feature
			opt.add(num_incoming_prototype_edges_for_feature == 1) 

			if layer_id in no_consecutive_repeat_layers:
				# constraint: for each proto node p for this feature, if p is a proto parent of the current instance, then p must not be a proto parent of the following instance node
				# overall meaning: curr instance node and next instance node must not have the same proto parent for this particular feature 
				# and one proto parent each is ensured by the previous constraint, opt.add(num_incoming_prototype_edges_for_feature == 1) 
				non_consec_feature_conditions.append(z3.And(
					[z3.Implies(
							z3.IsMember(proto_index, proto_parents(instance_submap_index_var)), # this particular proto is a parent of the current instance node
							z3.Not(z3.IsMember(proto_index, proto_parents(succ(instance_submap_index_var)))) # this particular proto is NOT a parent of the next instance node
						)
					for proto_index in proto_node_indices_for_feature]
				))

		if layer_id in no_consecutive_repeat_layers:	
			# constraint: for each node in a layer where you can't have consecutive identical nodes, must satisfy that if the curr node
			# isn't the end of its linear chain, then the curr node and next node in linear chain must have at least 1 feature with different proto parents
			opt.add(z3.Implies(instance_submap_index_var != end(level_var), z3.Or(non_consec_feature_conditions)))

		all_possible_proto_ids = [node_id for node_id in node_idx_submap_instance_proto.keys() if z3_helpers.is_proto(node_id)] # these are all the proto_ids for all the features in the current instance_proto_submatrix partition 
		for proto_id in all_possible_proto_ids:
			proto_index = node_idx_submap_instance_proto[proto_id]
			proto_instance_edge = instance_proto_submatrix[proto_index][instance_index]
			
			# assign proto parents for each instance node
			opt.add(proto_instance_edge == z3.IsMember(proto_index, proto_parents(instance_submap_index_var)))
			
			# ensure no instance -> proto edges
			opt.add(instance_proto_submatrix[instance_index][proto_index] == False) 

			# ensure no invalid proto-instance connections --> FIXED WITH INCREMENTAL SOLVING (see nonincremental version for original constraints)
		
# Constraint: Every instance node not at the top level of the hierarchy, must have 1 or 2 parents in the level above it
# idx_node_submap1 is a level above idx_node_submap2
# A_sub_matrix1, idx_node_submap1 are the parents of/level above A_sub_matrix2, idx_node_submap2
def add_instance_parent_count_constraints(combined_submatrix, 
																					parent_level,
																					idx_node_submap1, 
																					child_level,
																					idx_node_submap2, 
																					combined_idx_node_submap):
	combined_node_idx_submap = z3_helpers.invert_dict(combined_idx_node_submap)
	
	for i_subA2, node_id2 in idx_node_submap2.items():
		child_idx_combined = combined_node_idx_submap[node_id2]
		child_i_subA_var = z3.Int(f"level:{child_level}_i_subA:{i_subA2}")
		parent_count = z3.Sum([z3.If(combined_submatrix[combined_node_idx_submap[node_id1]][child_idx_combined], 1, 0) 
														for node_id1
														in idx_node_submap1.values()])
		opt.add(z3.Or(parent_count == 1, parent_count == 2))
		# opt.add(parent_count == 1)

		rank1 = rank(instance_parent1(child_i_subA_var))
		rank2 = rank(instance_parent2(child_i_subA_var))
		# opt.add(rank1 == rank2)
		# opt.add(z3.If(parent_count == 1, rank1 == rank2, rank1 < rank2))
		opt.add(z3.Implies(parent_count == 1, rank1 == rank2))
		opt.add(z3.Implies(parent_count == 2, rank1 < rank2))

		# Assign the 1 or 2 parents to non-zero level instance nodes for future reference in the constraint about parent orders based on the linear chain
		for list_idx, (parent_index1_subA1, parent_id1) in enumerate(list(idx_node_submap1.items())):
			parent_index1_combined = combined_node_idx_submap[parent_id1]
			opt.add(combined_submatrix[child_idx_combined][parent_index1_combined] == False) # cannot have edge from level i to level i - 1, children must be on level below

			parent_condition1 = combined_submatrix[parent_index1_combined][child_idx_combined]
			parent_index1_subA1_var = z3.Int(f"level:{parent_level}_i_subA:{parent_index1_subA1}")
			
			one_parent_condition = z3.And(
				parent_condition1 == (instance_parent1(child_i_subA_var) == parent_index1_subA1_var), 
				parent_condition1 == (instance_parent2(child_i_subA_var) == parent_index1_subA1_var)
			)

			two_parent_condition = z3.Or([
				z3.Or([
					z3.And(
						parent_condition1 == (instance_parent1(child_i_subA_var) == parent_index1_subA1_var),
						combined_submatrix[combined_node_idx_submap[parent_id2]][child_idx_combined] == (instance_parent2(child_i_subA_var) == z3.Int(f"level:{parent_level}_i_subA:{parent_index2_subA1}")),
					),
					z3.And(
						parent_condition1 == (instance_parent2(child_i_subA_var) == parent_index1_subA1_var),
						combined_submatrix[combined_node_idx_submap[parent_id2]][child_idx_combined] == (instance_parent1(child_i_subA_var) == z3.Int(f"level:{parent_level}_i_subA:{parent_index2_subA1}")),
					)
				]) for parent_index2_subA1, parent_id2 in list(idx_node_submap1.items()) if parent_id2 != parent_id1
			])

			# opt.add(one_parent_condition)
			opt.add(z3.Implies(parent_count == 1, one_parent_condition))
			opt.add(z3.Implies(parent_count == 2, two_parent_condition))
				
# Constraint: An instance level with a single node means that single node is the start and end of the linear chain
# idx_node_submap should be only the submap for this single-node level!
def add_intra_level_linear_chain_for_single_node_level(level, idx_node_submap):
	if len(idx_node_submap.keys()) != 1:
		raise Exception("Tried to create single node level chain for multi-nodes level")
	i_subA = list(idx_node_submap.keys())[0]
	i_subA_var = z3.Int(f"level:{level}_i_subA:{i_subA}")
	level_var = z3.Int(f"level:{level}")
	opt.add(start(level_var) == i_subA_var)
	opt.add(end(level_var) == i_subA_var)
	
# Constraint: The instance nodes in the given partition should form a linear chain
def add_intra_level_linear_chain(level, 
																 A_submatrix, 
																 idx_node_submap):
	partition_node_ids = list(idx_node_submap.values())
	num_partition_nodes = len(partition_node_ids)
	
	start_conditions = []
	end_conditions = []
	for i_subA in idx_node_submap.keys():
		# Directly use sub-matrix to count incoming/outgoing edges for node i within the level
		num_incoming_edges = z3.Sum([z3.If(A_submatrix[j, i_subA], 1, 0) for j in range(num_partition_nodes) if j != i_subA])
		num_outgoing_edges = z3.Sum([z3.If(A_submatrix[i_subA, j], 1, 0) for j in range(num_partition_nodes) if j != i_subA])
	
		is_start = z3.And(num_outgoing_edges == 1, num_incoming_edges == 0)
		is_end = z3.And(num_incoming_edges == 1, num_outgoing_edges == 0)

		start_conditions.append(is_start)
		end_conditions.append(is_end)

		is_intermediate_chain_node = z3.And(num_outgoing_edges == 1, num_incoming_edges == 1) #, is_not_dummy_node) ---> not necessary unless we have INSTANCE dummys
		opt.add(z3.Or(is_start, is_end, is_intermediate_chain_node))
		
		i_subA_var = z3.Int(f"level:{level}_i_subA:{i_subA}")
		level_var = z3.Int(f"level:{level}")

		# this is bidirectional iff because we do NOT want start(level_var) == i_subA_var if start_nodes[node_id] is false, and same for end nodes
		opt.add(is_start == (start(level_var) == i_subA_var))
		opt.add(is_end == (end(level_var) == i_subA_var))
	
	# Ensure exactly one start node and one end node in the partition
	opt.add(z3.AtMost(*start_conditions, 1))
	opt.add(z3.AtMost(*end_conditions, 1))
	opt.add(z3.AtLeast(*start_conditions, 1))
	opt.add(z3.AtLeast(*end_conditions, 1))

	define_linearly_adjacent_instance_relations(level, A_submatrix)

def reconstruct_intra_level_linear_chain(level, 
																 A_submatrix, 
																 idx_node_submap):
	partition_node_ids = list(idx_node_submap.values())
	num_partition_nodes = len(partition_node_ids)
	
	for i_subA in idx_node_submap.keys():
		# Directly use sub-matrix to count incoming/outgoing edges for node i within the level
		num_incoming_edges = z3.Sum([z3.If(A_submatrix[j, i_subA], 1, 0) for j in range(num_partition_nodes) if j != i_subA])
		num_outgoing_edges = z3.Sum([z3.If(A_submatrix[i_subA, j], 1, 0) for j in range(num_partition_nodes) if j != i_subA])
		
		# the index i_subA is in the single instance-only partition for that level
		is_start = z3.And(num_outgoing_edges == 1, num_incoming_edges == 0)
		is_end = z3.And(num_incoming_edges == 1, num_outgoing_edges == 0)

		i_subA_var = z3.Int(f"level:{level}_i_subA:{i_subA}")
		level_var = z3.Int(f"level:{level}")
		opt.add(is_start == (start(level_var) == i_subA_var))
		opt.add(is_end == (end(level_var) == i_subA_var))

	define_linearly_adjacent_instance_relations(level, A_submatrix)


# consider: A_submatrix has only one 1 in each row/col bc it's the submatrix of a single instance level that forms a linear chain
# can encode rank as integers and compare ints directly, but still asymptotically an overhead
# try to associate a bitvector with each index, and this directly gives us succ, pred, rank, start, and end 
def define_linearly_adjacent_instance_relations(level, A_submatrix):
	for i in range(len(A_submatrix)):
		for j in range(len(A_submatrix)):
			if i != j: # Avoid self-loops
				edge_i_to_j = A_submatrix[i, j]
				i_sub_A_var = z3.Int(f"level:{level}_i_subA:{i}")
				j_sub_A_var = z3.Int(f"level:{level}_i_subA:{j}")
				# again, these indices are all w.r.t. the SINGLE LEVEL INSTANCE ONLY partition
				opt.add(edge_i_to_j == (succ(i_sub_A_var) == j_sub_A_var)) # if and only if
				opt.add(z3.Implies(edge_i_to_j, rank(i_sub_A_var) < rank(j_sub_A_var))) # just if

def add_instance_parent_relationship_constraints(parent_level, child_level, idx_node_submap):
	child_level_var = z3.Int(f"level:{child_level}")
	parent_level_var = z3.Int(f"level:{parent_level}")
	for i_subA, node_id in idx_node_submap.items(): 
		non_overlap_layers = ['S', 'K', 'C', 'M'] # contiguous layers that don't have overlapping nodes. segmentation, keys, chords, melody, but not motifs/patterns
		layer_id = z3_helpers.get_layer_id(node_id)

		child_i_subA_var = z3.Int(f"level:{child_level}_i_subA:{i_subA}")
		if layer_id in non_overlap_layers: # rules for contiguous, non-overlapping layers
			# each node's LAST parent must not come after the next node's FIRST parent
			opt.add(z3.Implies(child_i_subA_var != end(child_level_var), rank(instance_parent2(child_i_subA_var)) <= rank(instance_parent1(succ(child_i_subA_var))))) 
		else: # rules for disjoint, non-contiguous, non-spanning sections (i.e. motifs/patterns)
			# each node's FIRST parent must not come after the next node's FIRST parent (since this is non-contiguous and we can have overlapping e.g. motifs)
			opt.add(z3.Implies(child_i_subA_var != end(child_level_var), rank(instance_parent1(child_i_subA_var)) <= rank(instance_parent1(succ(child_i_subA_var))))) 

		opt.add(z3.Implies(child_i_subA_var == start(child_level_var), instance_parent1(child_i_subA_var) == start(parent_level_var))) # the first node must have the prev level's first node as a parent
		opt.add(z3.Implies(child_i_subA_var == end(child_level_var), instance_parent2(child_i_subA_var) == end(parent_level_var))) # the final node must have the prev level's last node as a parent

# for solver loop version
def get_objective(submatrix, idx_node_submap):
	return z3.Sum([z3.If(submatrix[i][j] != bool(approx_centroid[node_idx_mapping[node_id1]][node_idx_mapping[node_id2]]), 1, 0) 
										 for (i, node_id1) in idx_node_submap.items()
										 for (j, node_id2) in idx_node_submap.items()])
 
# for optimizer version
def add_objective(submatrix, idx_node_submap):
	objective = get_objective(submatrix, idx_node_submap)
	opt.minimize(objective)

# save state ONLY for that level's instances
# the submatrix will often contain a pair of adjacent levels -- we're only interested in the relevant level
def save_instance_level_state_pair(levels_pair, submatrix, idx_node_mapping, model):
	print("SAVING INSTANCE LEVELS", levels_pair)
	state = []
	for i in range(len(submatrix)):
		node_id = idx_node_mapping[i] # this will be the SOURCE NODE for subsequent edges
		if z3_helpers.is_instance(node_id):
			for j in range(len(submatrix)):
				var = submatrix[i, j]  # Access the Z3 variable at this position in the submatrix
				evaluated_var = model.eval(var, model_completion=True)  # Evaluate this variable in the model
				state.append((var, evaluated_var))
	return state

# save state ONLY for that level's prototypes
def save_proto_level_state(level_submatrix, idx_node_mapping, model):
	state = []
	for i in range(level_submatrix.shape[0]):
		node_id = idx_node_mapping[i] # this will be the SOURCE NODE for subsequent edges
		if z3_helpers.is_proto(node_id):
			for j in range(level_submatrix.shape[1]):
				var = level_submatrix[i, j]  # Access the Z3 variable at this position in the submatrix
				evaluated_var = model.eval(var, model_completion=True)  # Evaluate this variable in the model
				state.append((var, evaluated_var))
	return state

def restore_level_state(levels_pair, level_states):
	print("RESTORING", levels_pair)
	for var, evaluated_var in level_states[levels_pair]:
		opt.add(var == evaluated_var)

def add_soft_constraints_for_submap(submatrix, idx_node_submap):
	for (i_subA, node_id1) in idx_node_submap.items():
		for (j_subA, node_id2) in idx_node_submap.items():
			opt.add_soft(submatrix[i_subA][j_subA] == bool(approx_centroid[node_idx_mapping[node_id1]][node_idx_mapping[node_id2]]))

def run(final_centroid_filename, final_idx_node_mapping_filename):
	level_states = {}
	# parent_level < next_level, meaning parent_level is HIGHER in the hierarchy than next_level (i.e. parent level of next_level)
	for (parent_level, child_level), (A_combined_submatrix, combined_idx_node_submap) in sorted(A_adjacent_instance_submatrices_list.items()):
		print(f"LEVEL PAIR FOR INSTANCE CONSTRAINTS ({parent_level}, {child_level})", time.perf_counter())
		opt.push()  # Save the current optimizer state for potential backtracking

		add_soft_constraints_for_submap(A_combined_submatrix, combined_idx_node_submap)
	
		(A_submatrix1, idx_node_submap1) = A_partition_instance_submatrices_list[parent_level]
		(A_submatrix2, idx_node_submap2) = A_partition_instance_submatrices_list[child_level]
		
		if parent_level == 0:
			if len(instance_levels_partition[parent_level]) > 1:
				add_intra_level_linear_chain(parent_level, A_submatrix1, idx_node_submap1)
			else:
				add_intra_level_linear_chain_for_single_node_level(parent_level, idx_node_submap1)
		else:
			prev_levels_pair = (parent_level - 1, child_level - 1)
			restore_level_state(prev_levels_pair, level_states)
			if len(instance_levels_partition[parent_level]) > 1:
				reconstruct_intra_level_linear_chain(parent_level, A_submatrix1, idx_node_submap1)
			else:
				add_intra_level_linear_chain_for_single_node_level(parent_level, idx_node_submap1)
		
		if child_level not in level_states:
			if len(instance_levels_partition[child_level]) > 1:
				add_intra_level_linear_chain(child_level, A_submatrix2, idx_node_submap2)
			else: # for all single-node *child* levels, we need to define that the node is the start and end of its linear chain
				# this is in order for it to get the correct parents (bc the start/end of linear chain in spanning levels has first/last parents at the ends of the parent chain
				add_intra_level_linear_chain_for_single_node_level(child_level, idx_node_submap2)
			
			add_instance_parent_count_constraints(A_combined_submatrix, parent_level, idx_node_submap1, child_level, idx_node_submap2, combined_idx_node_submap)
			add_instance_parent_relationship_constraints(parent_level, child_level, idx_node_submap2) # we ONLY want to do this for the child node
		
		add_objective(A_combined_submatrix, combined_idx_node_submap)

		# crashes with timeout bc the model at that timeout might not be sat
		# def on_model(m):
			# print("MODEL")
			# objective = get_objective(A_combined_submatrix, combined_idx_node_submap)
			# current_objective_value = m.eval(objective, model_completion=True).as_long()
			# print("CURRENT COST", current_objective_value)
		# opt.set_on_model(on_model)

		print("DONE ADDING CONSTRAINTS")
		result = opt.check()
		if result != z3.unsat:
			if result != z3.sat:
				print(f"Continuing with best-effort guess after timeout for instance levels {parent_level} and {child_level}")
			else:
				print(f"Consecutive levels {parent_level} and {child_level} are satisfiable", time.perf_counter())
			model = opt.model()
			level_states[(parent_level, child_level)] = save_instance_level_state_pair((parent_level, child_level), A_combined_submatrix, combined_idx_node_submap, model)
		else:
			print(f"Consecutive levels {parent_level} and {child_level} are not satisfiable")

		opt.pop()
		print()

	for level, (instance_proto_submatrix, idx_node_submap) in sorted(A_partition_instance_submatrices_list_with_proto.items()):
		print(f"LEVEL FOR PROTO CONSTRAINTS {level}", time.perf_counter())
		opt.push()  # Save the current optimizer state for potential backtracking

		add_soft_constraints_for_submap(instance_proto_submatrix, idx_node_submap)
		if level == 0: # the instance levels are stored by partition, so the keys are in pairs of the levels in that partition
			levels_pair = (0, 1)
		else:
			levels_pair = (level - 1, level)
		restore_level_state(levels_pair, level_states)

		# we need the linear chain functions defined for the subsequent proto-instance constraints
		(instance_only_submatrix, idx_node_submap_instance_only) = A_partition_instance_submatrices_list[level]
		if len(instance_levels_partition[level]) > 1:
			reconstruct_intra_level_linear_chain(level, instance_only_submatrix, idx_node_submap_instance_only)
		else: 
			add_intra_level_linear_chain_for_single_node_level(level, idx_node_submap_instance_only)

		add_prototype_to_prototype_constraints(idx_node_submap)
		add_prototype_to_instance_constraints(level, instance_proto_submatrix, idx_node_submap, node_metadata_dict)
		add_objective(instance_proto_submatrix, idx_node_submap)

		# crashes with timeout bc the model might be unsat
		# def on_model(m):
		# 	objective = get_objective(A_combined_submatrix, combined_idx_node_submap)
		# 	current_objective_value = m.eval(objective, model_completion=True).as_long()
		# 	print("CURRENT COST", current_objective_value)
		# opt.set_on_model(on_model)
		
		# with open(f"smtlib{(parent_level, child_level)}.txt", 'w') as file:
		# 	file.write(opt.sexpr())
		# 	z3.set_param(verbose = 4)
		
		result = opt.check()
		if result != z3.unsat:
			if result != z3.sat:
				print(f"Continuing with best-effort guess after timeout for proto level {level} ")
			else:
				print(f"Level {level} is satisfiable for proto constraints", time.perf_counter())
				
			model = opt.model()
			proto_state = save_proto_level_state(instance_proto_submatrix, idx_node_submap, model)
			level_states[levels_pair] += proto_state
		else:
			print(f"Level {level} is not satisfiable for proto constraints")

		opt.pop()
		print()

	# After iterating through all levels, you can check for overall satisfiability
	for levels_pair in sorted(list(level_states.keys())):
		# print("FINAL STATE AT LEVEL", level, level_states[level])
		restore_level_state(levels_pair, level_states)

	result = opt.check()
	if result == z3.sat:
		print("Final structure across all levels is satisfiable", time.perf_counter())
		final_model = opt.model()
		result = np.array([[1 if final_model.eval(A[i, j], model_completion=True) else 0 for j in range(n_A)] for i in range(n_A)])

		# FOR SAVING MATRIX VERSION
		final_result, final_idx_node_mapping = simanneal_helpers.remove_all_dummy_nodes(result, idx_node_mapping) # because we now have possible proto dummy nodes
		np.savetxt(final_centroid_filename, final_result)
		print("Saved final centroid at", final_centroid_filename)

		with open(final_idx_node_mapping_filename, 'w') as file:
			json.dump(final_idx_node_mapping, file)
		print(f"Saved: {final_idx_node_mapping_filename}")

		# FOR SAVING NETWORKX GRAPH VERSION AND VISUALIZE
		# G = simanneal_helpers.adj_matrix_to_graph(approx_centroid, idx_node_mapping, node_metadata_dict)
		# g = simanneal_helpers.adj_matrix_to_graph(result, idx_node_mapping, node_metadata_dict)

		# final_centroid_filename = os.path.join(DIRECTORY + '/centroid/test_centroid_final.pickle')
		# with open(final_centroid_filename, 'wb') as f:
		# 	pickle.dump(g, f)
		# 	print("Saved final centroid at", final_centroid_filename)

		# layers_G = build_graph.get_unsorted_layers_from_graph_by_index(G)
		# layers_g = build_graph.get_unsorted_layers_from_graph_by_index(g)

		# build_graph.visualize([G, g], [layers_G, layers_g])
	else:
			print("Unable to find a satisfiable structure across all levels")

if __name__ == "__main__":
  # NOTE: uncomment for viewing already repaired centroids
	centroid = np.loadtxt("/Users/ilanashapiro/Documents/constraints_project/project/centroid/test_graph_output_files/final_centroid_test.txt")
	with open("/Users/ilanashapiro/Documents/constraints_project/project/centroid/test_graph_output_files/approx_centroid_node_metadata_test.txt", 'r') as file:
		node_metadata_dict = json.load(file)
	with open("/Users/ilanashapiro/Documents/constraints_project/project/centroid/test_graph_output_files/final_centroid_idx_node_mapping_test.txt", 'r') as file:
		centroid_idx_node_mapping = {int(k): v for k, v in json.load(file).items()}
	
	g = simanneal_helpers.adj_matrix_to_graph(centroid, centroid_idx_node_mapping, node_metadata_dict)
	
	layers_g = build_graph.get_unsorted_layers_from_graph_by_index(g)
	build_graph.visualize([g], [layers_g])
	sys.exit(0)
	
	# NOTE: IMPORTANT -- assuming all unnecessary dummys (i.e. all instance dummys and impoossible proto dummys) have been removed ALREADY
	approx_centroid = np.loadtxt(DIRECTORY + '/centroid/approx_centroid_test.txt')
	with open(DIRECTORY + '/centroid/test_graph_output_files/approx_centroid_idx_node_mapping_test.txt', 'r') as file:
		idx_node_mapping = json.load(file)
		idx_node_mapping = {int(k): v for k, v in idx_node_mapping.items()}
	with open(DIRECTORY + '/centroid/test_graph_output_files/approx_centroid_node_metadata_test.txt', 'r') as file:
		node_metadata_dict = json.load(file)

	initialize_globals(approx_centroid, idx_node_mapping, node_metadata_dict)
	final_centroid_filename = DIRECTORY + '/centroid/final_centroid_test.txt'
	final_idx_node_mapping_filename = "final_centroid_idx_node_mapping_test.txt"
	run(final_centroid_filename, final_idx_node_mapping_filename)