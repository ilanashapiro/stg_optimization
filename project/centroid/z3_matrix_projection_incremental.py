import z3
import numpy as np 
import json
import re
import sys, os
import z3_matrix_projection_helpers as z3_helpers 
import z3_tests 
import simanneal_centroid_helpers as simanneal_helpers 
# import simanneal_centroid_tests as simanneal_tests
import math 
import networkx as nx
import sys
import time
import pickle

DIRECTORY = "/home/ilshapiro/project"
DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"

sys.path.append(DIRECTORY)
import build_graph

approx_centroid = np.loadtxt(DIRECTORY + '/centroid/approx_centroid_test.txt')
with open(DIRECTORY + '/centroid/approx_centroid_idx_node_mapping_test.txt', 'r') as file:
	idx_node_mapping = json.load(file)
	idx_node_mapping = {int(k): v for k, v in idx_node_mapping.items()}
with open(DIRECTORY + '/centroid/approx_centroid_node_metadata_test.txt', 'r') as file:
	node_metadata_dict = json.load(file)

approx_centroid, idx_node_mapping = simanneal_helpers.remove_dummy_nodes(approx_centroid, idx_node_mapping)

def visualize_centroid(approx_centroid, repaired_centroid):
	G = simanneal_helpers.adj_matrix_to_graph(approx_centroid, idx_node_mapping, node_metadata_dict)
	g = simanneal_helpers.adj_matrix_to_graph(repaired_centroid, idx_node_mapping, node_metadata_dict)
	layers_G = build_graph.get_unsorted_layers_from_graph_by_index(G)
	layers_g = build_graph.get_unsorted_layers_from_graph_by_index(g)
	build_graph.visualize_p([G, g], [layers_G, layers_g])
	sys.exit(0)

repaired_centroid = np.loadtxt(DIRECTORY + '/centroid/centroid_test_final.txt')
visualize_centroid(approx_centroid, repaired_centroid)

# G = z3_tests.G1
# approx_centroid = nx.to_numpy_array(G)
# idx_node_mapping = {index: node for index, node in enumerate(G.nodes())}

def invert_dict(d):
	return {v: k for k, v in d.items()}

node_idx_mapping = invert_dict(idx_node_mapping)
n_A = len(idx_node_mapping) 
opt = z3.Optimize()
opt.set('timeout', 10000) # in milliseconds. 300000ms = 5mins
opt.set("enable_lns", True)

instance_levels_partition = z3_helpers.partition_instance_levels(idx_node_mapping, node_metadata_dict) # dict: level -> instance nodes at that level
prototype_features_partition = z3_helpers.partition_prototype_features(idx_node_mapping, node_metadata_dict) # dict: prototype feature -> prototype nodes of that feature

# Declare Z3 variables to enforce constraints on
# Create a matrix in Z3 for adjacency; A[i][j] == 1 means an edge from i to j
A = np.array([[z3.Bool(f"A_{i}_{j}") for j in range(n_A)] for i in range(n_A)])
A_partition_instance_submatrices_list = z3_helpers.create_instance_partition_submatrices(A, node_idx_mapping, instance_levels_partition)
A_partition_instance_submatrices_list_with_proto = z3_helpers.create_instance_with_proto_partition_submatrices(A, node_idx_mapping, instance_levels_partition, prototype_features_partition, node_metadata_dict)
A_adjacent_instance_submatrices_list = z3_helpers.create_adjacent_level_instance_partition_submatrices(A, node_idx_mapping, instance_levels_partition)
# A_partition_instance_submatrices_list_with_context = z3_helpers.create_level_partition_submatrices_with_context(A, node_idx_mapping, instance_levels_partition, prototype_kinds_partition) # USED FOR DUMMYS
# A_adjacent_partition_submatrices_with_context = z3_helpers.create_adjacent_level_partition_submatrices_with_context(A, node_idx_mapping, instance_levels_partition, prototype_kinds_partition) # USED FOR DUMMYS

NodeSort = z3.IntSort()

# Uninterpreted functions
instance_parent1 = z3.Function('instance_parent1', NodeSort, NodeSort)
# def instance_parent1(child_i):
# 	assert(isinstance(child_i, int))
# 	return z3.Const(f"instance_parent1{child_i}", NodeSort)

instance_parent2 = z3.Function('instance_parent2', NodeSort, NodeSort)
proto_parent = z3.Function('proto_parent', NodeSort, NodeSort, NodeSort) # level number, index within level partition -> prototype index in entire centroid matrix A

# try to reformulate the problem in a PURELY FINITE DOMAIN (i.e. w.o. uninterpreted functions)
# orderings for linear chain
# IMPORTANT: the nodes indices for these functions refer to the RELEVANT PARTITION SUBMATRIX, NOT the entire centroid matrix A!!!
succ = z3.Function('succ', NodeSort, NodeSort)
start = z3.Function('start', z3.IntSort(), z3.IntSort(), NodeSort) # (primary level, secondary level) -> node index in relevant submatrix
end = z3.Function('end', z3.IntSort(), z3.IntSort(), NodeSort) # (primary level, secondary level) -> node index in relevant submatrix
rank = z3.Function('rank', NodeSort, z3.IntSort())

idx_node_mapping_prototype = {idx: node_id for idx, node_id in idx_node_mapping.items() if node_id.startswith("Pr")}
idx_node_mapping_instance = {idx: node_id for idx, node_id in idx_node_mapping.items() if not node_id.startswith("Pr")}

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
def add_prototype_to_instance_constraints(level, instance_proto_submatrix, instance_proto_idx_node_submap):
	node_idx_submap_instance_proto = invert_dict(instance_proto_idx_node_submap)
	(instance_only_submatrix, idx_node_submap_instance_only) = A_partition_instance_submatrices_list[rank]
	define_linearly_adjacent_instance_relations(instance_only_submatrix)

	for instance_index in idx_node_submap_instance_only.keys():
		if level != len(instance_levels_partition) - 1: # NEED TO FIX
			# no 2 linearly adjacent nodes can have the same prototype parent (for seg nodes)
			opt.add(z3.Implies(z3.And(instance_index != end(level)), proto_parent(level, instance_index) != proto_parent(level, succ(instance_index)))) 

		valid_proto_ids = [node_id for node_id in idx_node_submap.values() if z3_helpers.is_proto(node_id)]  # valid means seg-seg, and motif-motif
		incoming_prototype_edges = z3.Sum([z3.If(instance_proto_submatrix[node_idx_submap_instance_proto[proto_id]][instance_index], 1, 0) for proto_id in valid_proto_ids])
		opt.add(incoming_prototype_edges == 1)

		for proto_id in valid_proto_ids:
			proto_index = node_idx_submap_instance_proto[proto_id]
			opt.add(z3.Implies(instance_proto_submatrix[proto_index][instance_index], proto_parent(level, instance_index) == proto_index))
			
			# ensure no instance -> proto edges
			opt.add(instance_proto_submatrix[instance_index][proto_index] == False) 

			# ensure no invalid proto-instance connections --> FIXED WITH INCREMENTAL SOLVING (see nonincremental version for original constraints)

# Constraint: Every instance node not at the top level of the hierarchy, must have 1 or 2 parents in the level above it
# idx_node_submap1 is a level above idx_node_submap2
# A_sub_matrix1, idx_node_submap1 are the parents of/level above A_sub_matrix2, idx_node_submap2
def add_instance_parent_count_constraints(combined_submatrix, 
																					idx_node_submap1, 
																					idx_node_submap2, 
																					combined_idx_node_submap):
	combined_node_idx_submap = invert_dict(combined_idx_node_submap)
	
	for i_subA2, node_id2 in idx_node_submap2.items():
		child_idx_combined = combined_node_idx_submap[node_id2]
		parent_count = z3.Sum([z3.If(combined_submatrix[combined_node_idx_submap[node_id1]][child_idx_combined], 1, 0) 
														for node_id1
														in idx_node_submap1.values()])
		opt.add(z3.Or(parent_count == 1, parent_count == 2))

		# Assign the 1 or 2 parents to non-zero level instance nodes for future reference in the constraint about parent orders based on the linear chain
		for list_idx, (parent_index1_subA1, parent_id1) in enumerate(list(idx_node_submap1.items())):
			parent_index1_combined = combined_node_idx_submap[parent_id1]
			opt.add(combined_submatrix[child_idx_combined][parent_index1_combined] == False) # cannot have edge from level i to level i - 1, children must be on level below

			parent_condition1 = combined_submatrix[parent_index1_combined][child_idx_combined]
			# if 1 parent, then instance_parent1 and instance_parent2 are the same
			opt.add(z3.Implies(z3.And(parent_count == 1, parent_condition1), 
											z3.And(instance_parent1(i_subA2) == parent_index1_subA1, 
														instance_parent2(i_subA2) == parent_index1_subA1, 
														rank(instance_parent1(i_subA2)) == rank(instance_parent2(i_subA2)))))
			
			for parent_index2_subA1, parent_id2 in list(idx_node_submap1.items())[list_idx+1:]: # ensure we're only looking at distinct tuples of parents, otherwise we are UNSAT
				parent_index2_combined = combined_node_idx_submap[parent_id2]
				parent_condition2 = combined_submatrix[parent_index2_combined][child_idx_combined]
				opt.add(z3.Implies(z3.And(parent_condition1, parent_condition2, parent_count == 2), 
													z3.And(instance_parent1(i_subA2) == parent_index1_subA1, 
																instance_parent2(i_subA2) == parent_index2_subA1, 
																rank(instance_parent1(i_subA2)) < rank(instance_parent2(i_subA2)))))

# Constraint: An instance level with a single node means that single node is the start and end of the linear chain
# idx_node_submap should be only the submap for this single-node level!
def add_intra_level_linear_chain_for_single_node_level(level, idx_node_submap):
	if len(idx_node_submap.keys()) != 1:
		raise Exception("Tried to create single node level chain for multi-nodes level")
	i_subA = list(idx_node_submap.keys())[0]
	opt.add(start(level) == i_subA)
	opt.add(end(level) == i_subA)
	
# Constraint: The instance nodes in the given partition should form a linear chain
def add_intra_level_linear_chain(level, 
																 A_submatrix, 
																 idx_node_submap):
	partition_node_ids = list(idx_node_submap.values())
	num_partition_nodes = len(partition_node_ids)
	start_nodes = []
	end_nodes = []
	for node in partition_node_ids:
		start_nodes.append(z3.Bool(f"start_{node}"))
		end_nodes.append(z3.Bool(f"end_{node}"))
	
	for i_subA in idx_node_submap.keys():
		# Directly use sub-matrix to count incoming/outgoing edges for node i within the level
		num_incoming_edges = z3.Sum([z3.If(A_submatrix[j, i_subA], 1, 0) for j in range(num_partition_nodes) if j != i_subA])
		num_outgoing_edges = z3.Sum([z3.If(A_submatrix[i_subA, j], 1, 0) for j in range(num_partition_nodes) if j != i_subA])
		
		# the index i_subA is in the single instance-only partition for that level
		opt.add(start_nodes[i_subA] == z3.And(num_outgoing_edges == 1, num_incoming_edges == 0))
		opt.add(end_nodes[i_subA] == z3.And(num_incoming_edges == 1, num_outgoing_edges == 0))

		is_intermediate_chain_node = z3.And(z3.Not(start_nodes[i_subA]), z3.Not(end_nodes[i_subA]))#, is_not_dummy_node)
		opt.add(is_intermediate_chain_node == ((num_incoming_edges == 1) & (num_outgoing_edges == 1)))
		opt.add(z3.Implies(start_nodes[i_subA], start(level) == i_subA))
		opt.add(z3.Implies(end_nodes[i_subA], end(level) == i_subA))
	
	# Ensure exactly one start node and one end node in the partition
	opt.add(z3.Sum([z3.If(start_node, 1, 0) for start_node in start_nodes]) == 1)
	opt.add(z3.Sum([z3.If(end_node, 1, 0) for end_node in end_nodes]) == 1)

	define_linearly_adjacent_instance_relations(A_submatrix)

# consider: A_submatrix has only one 1 in each row/col bc it's the submatrix of a single instance level that forms a linear chain
# can encode rank as integers and compare ints directly, but still asymptotically an overhead
# try to associate a bitvector with each index, and this directly gives us succ, pred, rank, start, and end 
def define_linearly_adjacent_instance_relations(A_submatrix):
	for i in range(len(A_submatrix)):
		for j in range(len(A_submatrix)):
			if i != j:  # Avoid self-loops
				edge_i_to_j = A_submatrix[i, j]
				opt.add(z3.Implies(edge_i_to_j, z3.And(succ(i) == j, rank(i) < rank(j)))) # again, these indices are all w.r.t. the SINGLE LEVEL INSTANCE ONLY partition

def add_instance_parent_relationship_constraints(parent_level, child_level, idx_node_submap):
	for i_subA, node_id in idx_node_submap.items(): 
		non_overlap_layers = ['S', 'K', 'C', 'M'] # contiguous layers that don't have overlapping nodes. segmentation, keys, chords, melody, but not motifs/patterns
		spanning_layers = ['S', 'K', 'C', 'M'] # layers whose nodes span the total piece. segmentation, keys, chords, melody, but not motifs/patterns
		layer_id = z3_helpers.get_layer_id(node_id)

		if layer_id in non_overlap_layers: # rules for contiguous, non-overlapping layers
			# each node's LAST parent must not come after the next node's FIRST parent
			opt.add(z3.Implies(i_subA != end(child_level), rank(instance_parent2(i_subA)) <= rank(instance_parent1(succ(i_subA))))) 
		else: # rules for disjoint, non-contiguous, non-spanning sections (i.e. motifs/patterns)
			# each node's FIRST parent must not come after the next node's FIRST parent (since this is non-contiguous and we can have overlapping e.g. motifs)
			opt.add(z3.Implies(i_subA != end(child_level), rank(instance_parent1(i_subA)) <= rank(instance_parent1(succ(i_subA))))) 
		if layer_id in spanning_layers: # rules for spanning layers
			opt.add(z3.Implies(i_subA == start(child_level), instance_parent1(i_subA) == start(parent_level))) # the first node must have the prev level's first node as a parent
			opt.add(z3.Implies(i_subA == end(child_level), instance_parent2(i_subA) == end(parent_level))) # the final node must have the prev level's last node as a parent

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
# the submatrix will often contains a pair of adjacent levels -- we're only interested in the relevant level
def save_instance_level_state(level, submatrix, idx_node_mapping, node_metadata_dict, model):
	print("SAVING", level)
	state = []
	for i in range(len(submatrix)):
		node_id = idx_node_mapping[i] # this will be the SOURCE NODE for subsequent edges
		for j in range(len(submatrix)):
			var = submatrix[i, j]  # Access the Z3 variable at this position in the submatrix
			node_level = tuple(node_metadata_dict[node_id]['layer_rank'])
			is_instance_at_level = z3_helpers.is_instance(node_id) and node_level == level 
			if is_instance_at_level:
				evaluated_var = model.eval(var, model_completion=True)  # Evaluate this variable in the model
				state.append((var, evaluated_var))
	return state

# save state ONLY for that level's prototypes
def save_proto_level_state(submatrix, idx_node_mapping, model):
	state = []
	for i in range(submatrix.shape[0]):
		node_id = idx_node_mapping[i] # this will be the SOURCE NODE for subsequent edges
		if z3_helpers.is_proto(node_id):
			for j in range(submatrix.shape[1]):
				var = submatrix[i, j]  # Access the Z3 variable at this position in the submatrix
				evaluated_var = model.eval(var, model_completion=True)  # Evaluate this variable in the model
				state.append((var, evaluated_var))
	return state

def restore_level_state(level, level_states):
	for var, evaluated_var in level_states[level]:
		opt.add(var == evaluated_var)

def add_soft_constraints_for_submap(submatrix, idx_node_submap):
	for (i_subA, node_id1) in idx_node_submap.items():
		for (j_subA, node_id2) in idx_node_submap.items():
			opt.add_soft(submatrix[i_subA][j_subA] == bool(approx_centroid[node_idx_mapping[node_id1]][node_idx_mapping[node_id2]]))

level_states = {}
# parent_level < next_level, meaning parent_level is HIGHER in the hierarchy than next_level (i.e. parent level of next_level)
for (parent_level, child_level), (A_combined_submatrix, combined_idx_node_submap) in sorted(A_adjacent_instance_submatrices_list.items()):
	opt.push()  # Save the current optimizer state for potential backtracking
  
	# NOTE: we never need to restore level states here, because for each (parent, child) pair we only save the parent each time, 
	# and the next (parent, child) pair is (child, grandchild) of the prev parent, neither of which would ever be saved in the prev iteration or we end up UNSAT

	add_soft_constraints_for_submap(A_combined_submatrix, combined_idx_node_submap)

	(A_submatrix1, idx_node_submap1) = A_partition_instance_submatrices_list[parent_level]
	(A_submatrix2, idx_node_submap2) = A_partition_instance_submatrices_list[child_level]

	if parent_level not in level_states and len(instance_levels_partition[parent_level]) > 1:
		add_intra_level_linear_chain(parent_level, A_submatrix1, idx_node_submap1)
	if child_level not in level_states:
		if len(instance_levels_partition[child_level]) > 1:
			add_intra_level_linear_chain(child_level, A_submatrix2, idx_node_submap2)
		else: # for all single-node *child* levels, we need to define that the node is the start and end of its linear chain
			# this is in order for it to get the correct parents (bc the start/end of linear chain in spanning levels has first/last parents at the ends of the parent chain
			add_intra_level_linear_chain_for_single_node_level(child_level, idx_node_submap2)
		add_instance_parent_relationship_constraints(parent_level, child_level, idx_node_submap2) # we ONLY want to do this for the child node
	
	add_instance_parent_count_constraints(A_combined_submatrix, idx_node_submap1, idx_node_submap2, combined_idx_node_submap)
	add_objective(A_combined_submatrix, combined_idx_node_submap)

	# with open(f"smtlib{(parent_level, child_level)}.txt", 'w') as file:
	# 	file.write(opt.sexpr())
		# z3.set_param(verbose = 4)

	# crashes with timeout bc the model might be unsat
	# def on_model(m):
	# 	objective = get_objective(A_combined_submatrix, combined_idx_node_submap)
	# 	current_objective_value = m.eval(objective, model_completion=True).as_long()
	# 	print("CURRENT COST", current_objective_value)
	# opt.set_on_model(on_model)

	result = opt.check()
	if result != z3.unsat:
		if result != z3.sat:
			print(f"Continuing with best-effort guess after timeout for instance levels {parent_level} and {child_level}")
		else:
			print(f"Consecutive levels {parent_level} and {child_level} are satisfiable", time.perf_counter())
		model = opt.model()
		level_states[parent_level] = save_instance_level_state(parent_level, A_combined_submatrix, combined_idx_node_submap, node_metadata_dict, model)
		if child_level == max(instance_levels_partition.keys()):
			level_states[child_level] = save_instance_level_state(child_level, A_combined_submatrix, combined_idx_node_submap, node_metadata_dict, model)
	elif result == z3.unsat:
		print(f"Consecutive levels {parent_level} and {child_level} are not satisfiable")

	opt.pop()
	print()

for level, (instance_proto_submatrix, idx_node_submap) in A_partition_instance_submatrices_list_with_proto.items():
	print(f"LEVEL FOR PROTO CONSTRAINTS {level}", time.perf_counter())
	opt.push()  # Save the current optimizer state for potential backtracking

	add_soft_constraints_for_submap(instance_proto_submatrix, idx_node_submap)
	restore_level_state(level, level_states)
	add_prototype_to_prototype_constraints(idx_node_submap)
	# add_prototype_to_instance_constraints(level, instance_proto_submatrix, idx_node_submap)
	add_objective(instance_proto_submatrix, idx_node_submap)

	# crashes with timeout bc the model might be unsat
	# def on_model(m):
	# 	objective = get_objective(A_combined_submatrix, combined_idx_node_submap)
	# 	current_objective_value = m.eval(objective, model_completion=True).as_long()
	# 	print("CURRENT COST", current_objective_value)
	# opt.set_on_model(on_model)
	
	if opt.check() != z3.unsat:
		if result != z3.sat:
			print(f"Continuing with best-effort guess after timeout for proto level {level} ")
		else:
			print(f"Level {level} is satisfiable for proto constraints", time.perf_counter())
			
		model = opt.model()
		proto_state = save_proto_level_state(instance_proto_submatrix, idx_node_submap, model)
		level_states[level] += proto_state
	else:
		print(f"Level {level} is not satisfiable for proto constraints")

	opt.pop()

# After iterating through all levels, you can check for overall satisfiability
for level in instance_levels_partition.keys():
	# print("FINAL STATE AT LEVEL", level, level_states[level])
	restore_level_state(level, level_states)

if opt.check() == z3.sat:
	print("Final structure across all levels is satisfiable", time.perf_counter())
	final_model = opt.model()
	print(final_model)
	result = np.array([[1 if final_model.eval(A[i, j], model_completion=True) else 0 for j in range(n_A)] for i in range(n_A)])
	print(result, idx_node_mapping)

	# FOR SAVING MATRIX VERSION
	final_centroid_filename = DIRECTORY + '/centroid/centroid_test_final.txt'
	np.savetxt(final_centroid_filename, result)
	print("Saved final centroid at", final_centroid_filename)

	# FOR SAVING NETWORKX GRAPH VERSION AND VISUALIZE
	# G = simanneal_helpers.adj_matrix_to_graph(approx_centroid, idx_node_mapping, node_metadata_dict)
	# g = simanneal_helpers.adj_matrix_to_graph(result, idx_node_mapping, node_metadata_dict)

	# final_centroid_filename = os.path.join(DIRECTORY + '/centroid/centroid_test_final.pickle')
	# with open(final_centroid_filename, 'wb') as f:
	# 	pickle.dump(g, f)
	# 	print("Saved final centroid at", final_centroid_filename)

	# layers_G = build_graph.get_unsorted_layers_from_graph_by_index(G)
	# layers_g = build_graph.get_unsorted_layers_from_graph_by_index(g)

	# build_graph.visualize_p([G, g], [layers_G, layers_g])
else:
		print("Unable to find a satisfiable structure across all levels")

