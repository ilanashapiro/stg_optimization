import re
import numpy as np
import sys

def is_instance(node_id):
  s_match = re.match(r'S(\d+)L(\d+)N(\d+)', node_id)
  p_match = re.match(r'P(\d+)O(\d+)N(\d+)', node_id)
  return s_match or p_match

def is_proto(node_id):
  pr_match = re.match(r'Pr([SP])(\d+)', node_id)
  return pr_match

def parse_instance_node_id(node_id):
  s_match = re.match(r'S(\d+)L(\d+)N(\d+)', node_id)
  p_match = re.match(r'P(\d+)O(\d+)N(\d+)', node_id)
  if s_match:
    n1, n2, n3 = map(int, s_match.groups())
    return ('S', n1, n2, n3)
  elif p_match:
    n1, n2, n3 = map(int, p_match.groups())
    return ('P', n1, n2, n3)

def parse_prototype_node_id(node_id):
  pr_match = re.match(r'Pr([SP])(\d+)', node_id)
  if pr_match:
    # Returns a tuple with 'Pr', the kind ('S' or 'P'), and the numeric part as elements
    return ('Pr',) + pr_match.groups()

def partition_prototype_kinds(idx_node_mapping):
  prototype_dict = {'S': [], 'P': []}  # Initialize to store prototype nodes by their kind ('S' or 'P')

  for node_id in idx_node_mapping.values():
    proto_node_info = parse_prototype_node_id(node_id)
    if proto_node_info:
      prototype_kind = proto_node_info[1] # 'S' or 'P'
      prototype_dict[prototype_kind].append(node_id)

  return prototype_dict

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

def partition_instance_levels(idx_node_mapping):
  max_seg_level = 0
  node_levels = {}  # key: level, value: list of node IDs
  motif_nodes = []

  for node_id in idx_node_mapping.values():
    node_id_info = parse_instance_node_id(node_id)
    if node_id_info:
      node_kind, _, level, _ = node_id_info
      zero_indexed_level = level - 1
      if node_kind == 'S':
        max_seg_level = max(max_seg_level, zero_indexed_level)
        node_levels.setdefault(zero_indexed_level, []).append(node_id)
      elif node_kind == 'P': # collect the motif/pattern nodes and add to levels after the max seg level is determined
        motif_nodes.append(node_id)
  
  node_levels[max_seg_level + 1] = motif_nodes
  print("GET LEVELS inST", node_levels)
  return node_levels

def create_instance_partition_submatrices(A, node_idx_mapping, instance_levels_partition):
  sub_matrices = {}
  for level, node_ids in instance_levels_partition.items():
    indices = [node_idx_mapping[node_id] for node_id in node_ids]
    sub_matrix = A[np.ix_(indices, indices)]
    sub_matrix_mapping = {i: node_id for i, node_id in enumerate(node_ids)}
    sub_matrices[level] = (sub_matrix, sub_matrix_mapping)
  return sub_matrices

def create_adjacent_level_instance_partition_submatrices(A, node_idx_mapping, levels_partition):
  print("LEVELS PARTITION ------", levels_partition, "--------")
  adjacent_level_submatrices = {}
  for level, node_ids in levels_partition.items():
    if level == 0:
      continue

    prev_level_node_ids = levels_partition[level - 1]
    combined_node_ids = prev_level_node_ids + node_ids

    indices = [node_idx_mapping[node_id] for node_id in combined_node_ids]
    sub_matrix = A[np.ix_(indices, indices)]
    sub_matrix_mapping = {i: node_id for i, node_id in enumerate(combined_node_ids)}
    adjacent_level_submatrices[(level - 1, level)] = (sub_matrix, sub_matrix_mapping)
  return adjacent_level_submatrices

def create_adjacent_level_proto_and_instance_partition_submatrices(A, node_idx_mapping, instance_levels_partition, prototype_kinds_partition):
  adjacent_level_submatrices = {}
  for level, node_ids in instance_levels_partition.items():
    if level == 0:
      continue
    
    if level == len(instance_levels_partition) - 1:
      kinds = ['S', 'P']
    else:
      kinds = ['S']

    prev_level_node_ids = instance_levels_partition[level - 1]
    combined_node_ids = prev_level_node_ids + node_ids

    # Include prototype nodes of the same kind as the current level nodes
    prototype_node_ids = []
    for kind in kinds:
      prototype_node_ids += prototype_kinds_partition.get(kind, [])
    combined_node_ids += prototype_node_ids

    indices = [node_idx_mapping[node_id] for node_id in combined_node_ids]
    sub_matrix = A[np.ix_(indices, indices)]
    sub_matrix_mapping = {i: node_id for i, node_id in enumerate(combined_node_ids)}
    adjacent_level_submatrices[(level - 1, level)] = (sub_matrix, sub_matrix_mapping)

  return adjacent_level_submatrices

def get_node_type(node_id):
  seg_proto_match = re.match(r"PrS\d+", node_id)
  seg_instance_match = re.match(r"S\d+L\d+N\d+", node_id)
  motif_proto_match = re.match(r"PrP\d+", node_id)
  motif_instance_match = re.match(r"P\d+O\d+N\d+", node_id)
  if seg_proto_match:
    return "SEG_PROTO"
  elif seg_instance_match:
    return "SEG_INSTANCE"
  elif motif_proto_match:
    return "MOTIF_PROTO"
  elif motif_instance_match:
    return "MOTIF_INSTANCE"
  print("ERROR")
  sys.exit(0)