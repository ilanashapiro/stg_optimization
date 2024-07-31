import re
import numpy as np
import sys

def get_layer_id(node):
	for layer_id in ['S', 'P', 'K', 'C', 'M']:
		if node.startswith(layer_id):
			return layer_id

def is_instance(node_id):
	return not is_proto(node_id)

def is_proto(node_id):
	return node_id.startswith('Pr')

def parse_instance_node_id(node_id):
	s_match = re.match(r'S(\d+)L(\d+)N(\d+)', node_id)
	p_match = re.match(r'P(\d+)O(\d+)N(\d+)', node_id)
	if s_match:
		n1, n2, n3 = map(int, s_match.groups())
		return ('S', n1, n2, n3)
	elif p_match:
		n1, n2, n3 = map(int, p_match.groups())
		return ('P', n1, n2, n3)

def partition_prototype_features(idx_node_mapping, node_metadata_dict):
	prototype_features_dict = {}
	for node_id in idx_node_mapping.values():
		if is_proto(node_id):
			feature_name = node_metadata_dict[node_id]['feature_name']
			prototype_features_dict.setdefault(feature_name, []).append(node_id)
	return prototype_features_dict


# partition A by level into matrices with all the instance nodes of that level, and the prototypes of that kind
def create_instance_proto_partition_submatrices(A, node_idx_mapping, instance_levels_partition, prototype_kinds_partition):
	level_submatrices = {}
	for level, node_ids in instance_levels_partition.items():
		parsed = parse_instance_node_id(node_ids[0])
		if parsed:
				kind = parsed[0]

		# Include prototype nodes of the same kind as the current level nodes
		prototype_node_ids = prototype_kinds_partition.get(kind, [])
		combined_node_ids = node_ids + prototype_node_ids

		indices = [node_idx_mapping[node_id] for node_id in combined_node_ids]
		sub_matrix = A[np.ix_(indices, indices)]
		sub_matrix_mapping = {i: node_id for i, node_id in enumerate(combined_node_ids)}
		level_submatrices[level] = (sub_matrix, sub_matrix_mapping)
	return level_submatrices

def partition_instance_levels(idx_node_mapping, node_metadata_dict):
	# key: zero-indexed layer rank tuple (primary layer level, secondary subhierarchy level)
	# value: list of node IDs
	node_levels = {} 
	for node_id in idx_node_mapping.values():
		if is_instance(node_id):
			layer_rank = tuple(node_metadata_dict[node_id]['layer_rank'])
			node_levels.setdefault(layer_rank, []).append(node_id)
	return node_levels

def create_instance_partition_submatrices(A, node_idx_mapping, instance_levels_partition):
	sub_matrices = {}
	for level, node_ids in instance_levels_partition.items():
		indices = [node_idx_mapping[node_id] for node_id in node_ids]
		sub_matrix = A[np.ix_(indices, indices)]
		sub_matrix_mapping = {i: node_id for i, node_id in enumerate(node_ids)}
		sub_matrices[level] = (sub_matrix, sub_matrix_mapping)
	return sub_matrices

def create_adjacent_level_instance_partition_submatrices(A, node_idx_mapping, instance_levels_partition):
	adjacent_level_submatrices = {}
	sorted_instance_levels_partition_list = sorted(instance_levels_partition.items())

	for level_idx, (layer_rank, node_ids) in enumerate(sorted_instance_levels_partition_list[1:], start=1):
		(prev_layer_rank, prev_level_node_ids) = sorted_instance_levels_partition_list[level_idx - 1]
		combined_node_ids = prev_level_node_ids + node_ids

		indices = [node_idx_mapping[node_id] for node_id in combined_node_ids]
		sub_matrix = A[np.ix_(indices, indices)]
		sub_matrix_mapping = {i: node_id for i, node_id in enumerate(combined_node_ids)}
		adjacent_level_submatrices[(prev_layer_rank, layer_rank)] = (sub_matrix, sub_matrix_mapping)
	return adjacent_level_submatrices

def create_instance_with_proto_partition_submatrices(A, node_idx_mapping, instance_levels_partition, prototype_features_partition, node_metadata_dict):
	instance_level_submatrices_with_proto = {}
	for level in instance_levels_partition:
		instance_node_ids = instance_levels_partition[level]

		i = 0
		inst_layer_features = []
		while not inst_layer_features and i < len(instance_node_ids): # we want to find the first non-filler inst node so we can get the features dict for this level
			inst_layer_features = node_metadata_dict[instance_node_ids[i]]['features_dict'].keys()
			i += 1

		prototype_node_ids = []
		for feature in inst_layer_features:
			prototype_node_ids.extend(prototype_features_partition.get(feature))
		combined_node_ids = instance_node_ids + prototype_node_ids
		
		indices_A = [node_idx_mapping[node_id] for node_id in combined_node_ids]
		sub_matrix = A[np.ix_(indices_A, indices_A)]
		sub_matrix_mapping = {i: node_id for i, node_id in enumerate(combined_node_ids)}
		instance_level_submatrices_with_proto[level] = (sub_matrix, sub_matrix_mapping)
	return instance_level_submatrices_with_proto

# this was never updated for the more robust STGs because it's not currently needed
# can update in future if needed, this is the old version 
# def create_adjacent_level_proto_and_instance_partition_submatrices(A, node_idx_mapping, instance_levels_partition, prototype_kinds_partition):
#   adjacent_level_submatrices = {}
#   for level, node_ids in instance_levels_partition.items():
#     if level == 0:
#       continue
		
#     if level == len(instance_levels_partition) - 1:
#       kinds = ['S', 'P']
#     else:
#       kinds = ['S']

#     prev_level_node_ids = instance_levels_partition[level - 1]
#     combined_node_ids = prev_level_node_ids + node_ids

#     # Include prototype nodes of the same kind as the current level nodes
#     prototype_node_ids = []
#     for kind in kinds:
#       prototype_node_ids += prototype_kinds_partition.get(kind, [])
#     combined_node_ids += prototype_node_ids

#     indices = [node_idx_mapping[node_id] for node_id in combined_node_ids]
#     sub_matrix = A[np.ix_(indices, indices)]
#     sub_matrix_mapping = {i: node_id for i, node_id in enumerate(combined_node_ids)}
#     adjacent_level_submatrices[(level - 1, level)] = (sub_matrix, sub_matrix_mapping)

#   return adjacent_level_submatrices

# USED FOR DUMMYS -- DO NOT DELETE
# this was never updated for the more robust STGs because we're not doing dummys because of timeout
# can update in future if needed, this is the old version 
# def create_adjacent_level_partition_submatrices_with_context(A, node_idx_mapping, instance_levels_partition, prototype_kinds_partition):
#   adjacent_level_submatrices_with_context = {}
#   total_levels = len(instance_levels_partition)
	
#   for level1 in range(total_levels):  # Iterate to the second-to-last level to ensure pairs
#     if level1 == total_levels - 1:
#       break

#     level2 = level1 + 1
#     combined_node_ids = instance_levels_partition[level1] + instance_levels_partition[level2]
		
#     # Include nodes from the level above level1 if not the first level
#     if level1 > 0:
#       combined_node_ids.extend(instance_levels_partition[level1 - 1])

#     # Include nodes from the level below level2 if not the last level
#     if level2 < total_levels - 1:
#       combined_node_ids.extend(instance_levels_partition[level2 + 1])
		
#     # Include prototype nodes for both levels
#     prototypes_set = set()
#     for lvl in [level1, level2]:
#       kind = 'P' if lvl == total_levels - 1 else 'S'
#       prototype_node_ids = prototype_kinds_partition.get(kind, [])
#       prototypes_set.update(prototype_node_ids)
		
#     combined_node_ids.extend(prototypes_set)

#     indices = [node_idx_mapping[node_id] for node_id in combined_node_ids]
#     sub_matrix = A[np.ix_(indices, indices)]
#     sub_matrix_mapping = {i: node_id for i, node_id in enumerate(combined_node_ids)}

#     adjacent_level_submatrices_with_context[(level1, level2)] = (sub_matrix, sub_matrix_mapping)
	
#   return adjacent_level_submatrices_with_context

# USED FOR DUMMYS -- DO NOT DELETE
# this was never updated for the more robust STGs because we're not doing dummys because of timeout
# can update in future if needed, this is the old version 
# def create_level_partition_submatrices_with_context(A, node_idx_mapping, instance_levels_partition, prototype_kinds_partition):
#   level_submatrices_with_context = {}
#   total_levels = len(instance_levels_partition)

#   for level, node_ids in instance_levels_partition.items():
#     kind = 'P' if level == total_levels - 1 else 'S'
#     combined_node_ids = list(node_ids)  # Create a copy to avoid modifying the original

#     # Include nodes from the level above if not the first level
#     if level > 0:
#       prev_level_node_ids = instance_levels_partition[level - 1]
#       combined_node_ids.extend(prev_level_node_ids)

#     # Include nodes from the level below if not the last level
#     if level < total_levels - 1:
#       next_level_node_ids = instance_levels_partition[level + 1]
#       combined_node_ids.extend(next_level_node_ids)

#     # Include prototype nodes of the same kind as the current level nodes
#     prototype_node_ids = prototype_kinds_partition.get(kind, [])
#     combined_node_ids.extend(prototype_node_ids)

#     indices = [node_idx_mapping[node_id] for node_id in combined_node_ids]
#     sub_matrix = A[np.ix_(indices, indices)]
#     sub_matrix_mapping = {i: node_id for i, node_id in enumerate(combined_node_ids)}
#     level_submatrices_with_context[level] = (sub_matrix, sub_matrix_mapping)

#   return level_submatrices_with_context