import re
import numpy as np
import sys

def parse_node_id(node_id):
  s_match = re.match(r'S(\d+)L(\d+)N(\d+)', node_id)
  p_match = re.match(r'P(\d+)O(\d+)N(\d+)', node_id)
  if s_match:
    n1, n2, n3 = map(int, s_match.groups())
    return ('S', n1, n2, n3)
  elif p_match:
    n1, n2, n3 = map(int, p_match.groups())
    return ('P', n1, n2, n3)

def partition_levels(idx_node_mapping):
  max_seg_level = 0
  node_levels = {}  # key: level, value: list of node IDs
  motif_nodes = []

  for node_id in idx_node_mapping.values():
    node_id_info = parse_node_id(node_id)
    if node_id_info:
      node_kind, _, level, _ = node_id_info
      zero_indexed_level = level - 1
      if node_kind == 'S':
        max_seg_level = max(max_seg_level, zero_indexed_level)
        node_levels.setdefault(zero_indexed_level, []).append(node_id)
      elif node_kind == 'P': # collect the motif/pattern nodes and add to levels after the max seg level is determined
        motif_nodes.append(node_id)
  
  node_levels[max_seg_level + 1] = motif_nodes
  return node_levels

def create_partition_submatrices(A, idx_node_mapping, node_idx_mapping, levels_partition):
  sub_matrices = {}
  for level, node_ids in levels_partition.items():
    indices = [node_idx_mapping[node_id] for node_id in node_ids]
    sub_matrix = A[np.ix_(indices, indices)]
    sub_matrix_mapping = {i: idx_node_mapping[node_idx_mapping[node_id]] for i, node_id in enumerate(node_ids)}
    sub_matrices[level] = (sub_matrix, sub_matrix_mapping)
  return sub_matrices

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