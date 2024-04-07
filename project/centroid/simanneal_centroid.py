from curses import A_RIGHT
import networkx as nx
import numpy as np
import random
import re 
from simanneal import Annealer
import sys
import json 

import simanneal_centroid_tests as tests
import simanneal_centroid_helpers as helpers

sys.path.append("/Users/ilanashapiro/Documents/constraints_project/project")
import build_graph

'''
Simulated Annealing (SA) Combinatorial Optimization Approach
1. Use SA to find optimal alignments between current centroid and each graph in corpus
2. Use the resulting average difference adjacency matrix between centroid and corpus to select the best (valid) transform. 
3. Modify centroid and repeat until loss converges. Loss is sum of dist from centroid to seach graph in corpus
'''

def normalize(value, lower_bound, upper_bound):
  # Avoid division by zero if max_value == min_value
  if upper_bound == lower_bound:
    return 0
  return (value - lower_bound) / (upper_bound - lower_bound)

# current centroid g, list of alignments list_a to the graphs in the corpus list_G
# loss is the sum of the distances between current centroid g and each graph in corpus G,
  # based on the current alignments
# this is our objective we're trying to minimize
def loss(A_g, list_alignedA_G):
  distances = np.array([dist(A_g, A_G) for A_G in list_alignedA_G])
  distance = np.sum(distances)
  variance = np.var(distances)

  print("DIST", distance, "VAR", variance)
  return distance * variance

def align(a, A_G):
  return a.T @ A_G @ a 

# dist between g and G given alignment a
# i.e. reorder nodes of G according to alignment (i.e. permutation matrix) a
# ||A_g - a^t * A_G * a|| where ||.|| is the norm (using Frobenius norm)
def dist(A_g, A_G):
  return np.linalg.norm(A_g - A_G, 'fro')

# Generates random n x n permutation alignment matrixx
def random_alignment(n):
  perm_indices = np.random.permutation(n)
  identity = np.eye(n)
  return identity[perm_indices]

class GraphAlignmentAnnealer(Annealer):
  def __init__(self, initial_alignment, A_g, A_G, centroid_node_mapping):
    super(GraphAlignmentAnnealer, self).__init__(initial_alignment)
    self.A_g = A_g
    self.A_G = A_G
    self.centroid_node_mapping = centroid_node_mapping
    self.node_partitions = self.get_node_partitions()
  
  # this prevents us from printing out alignment annealing updates since this gets confusing when also doing centroid annealing
  # def default_update(self, step, T, E, acceptance, improvement):
  #   return 
  
  def get_node_info(self, node_id):
    if node_id.startswith('PrS'):
      return ('prototype_segmentation', None)
    if node_id.startswith('PrP'):
      return ('prototype_motif', None)
    match = re.match(r'S\d+L(\d+)N\d+', node_id)
    if match:
      return ('segmentation', int(match.group(1)))
    if 'O' in node_id and 'N' in node_id:
      return ('motif', None)
    return (None, None)
  
  def get_node_partitions(self):
    """Partition centroid_node_mapping into labeled sets."""
    partitions = {'prototype_segmentation': [], 'prototype_motif': [],
                  'segmentation': {}, 'motif': []}
    for index, node_id in self.centroid_node_mapping.items():
      kind, layer = self.get_node_info(node_id)
      if kind == 'segmentation':
        if layer not in partitions[kind]:
          partitions[kind][layer] = []
        partitions[kind][layer].append(index)
      elif kind:
        partitions[kind].append(index)
    return partitions

  def move(self):
    """Swaps two rows in the n x n permutation matrix by permuting within valid sets (protype node class or individual level)"""
    n = len(self.state)
    i = random.randint(0, n - 1)
    i_kind, i_layer = self.get_node_info(self.centroid_node_mapping[i])
    j_options = None 

    # Identify partition and find a random j within the same partition
    if i_kind == 'segmentation' and i_layer in self.node_partitions[i_kind]:
      j_options = self.node_partitions[i_kind][i_layer]
    elif i_kind:
      j_options = self.node_partitions[i_kind]

    # Ensure i is not equal to j
    if j_options and len(j_options):
      j = random.choice(j_options)
      while j == i and len(j_options) > 1: # if a partition has only 1 element we have infinite loop
        j = random.choice(j_options)
    else:
      # Fallback to random selection if no suitable j is found
      j = random.randint(0, n - 1)
      while j == i:
        j = random.randint(0, n - 1)

    self.state[[i, j], :] = self.state[[j, i], :]  # Swap rows i and j

  def energy(self): # i.e. cost, self.state represents the permutation/alignment matrix a
    return dist(self.A_g, align(self.state, self.A_G))

(g, layers, _) = build_graph.generate_graph('/Users/ilanashapiro/Documents/constraints_project/project/LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short_segments.txt', '/Users/ilanashapiro/Documents/constraints_project/project/LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short_motives.txt')
(G, layers1, _) = build_graph.generate_graph('/Users/ilanashapiro/Documents/constraints_project/project/LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short_segments.txt', '/Users/ilanashapiro/Documents/constraints_project/project/LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short_motives.txt')
# padded_matrices, centroid_node_mapping = helpers.pad_adj_matrices([tests.G1, tests.G2])
# A_G1, A_G2 = padded_matrices[0], padded_matrices[1]

# list_G = [tests.G1, tests.G2]
# listA_G, centroid_node_mapping = helpers.pad_adj_matrices(list_G)
# initial_centroid = listA_G[0]
# np.savetxt("centroid.txt", initial_centroid)

# A_g_c = np.loadtxt('centroid.txt')
# with open("centroid_node_mapping.txt", 'r') as file:
#   centroid_node_mapping = json.load(file)
#   centroid_node_mapping = {int(k): v for k, v in centroid_node_mapping.items()}
# # layers1 = build_graph.get_unsorted_layers_from_graph_by_index(tests.G1)
# # layers2 = build_graph.get_unsorted_layers_from_graph_by_index(tests.G2)
# g_c = helpers.adj_matrix_to_graph(A_g_c, centroid_node_mapping)
# layers_g_c = build_graph.get_unsorted_layers_from_graph_by_index(g_c)
# build_graph.visualize_p([g, G, g_c], [layers, layers1, layers_g_c])

# initial_state = np.eye(np.shape(A_G)[0])
# graph_aligner = GraphAlignmentAnnealer(initial_state, A_g, A_G, centroid_node_mapping)
# graph_aligner.Tmax = 1.25
# graph_aligner.Tmin = 0.01 
# graph_aligner.steps = 5000 

# alignment, cost = graph_aligner.anneal() # don't do auto scheduling, it does not appear to work at all

# print("Best cost1", cost)

# graph_aligner = GraphAlignmentAnnealer(alignment, A_g, A_G, centroid_node_mapping)
# graph_aligner.Tmax = 1.25
# graph_aligner.Tmin = 0.01 
# graph_aligner.steps = 5000 


# print("Best cost2", cost)

def get_alignments_to_centroid(A_g, listA_G, node_mapping, Tmax, Tmin, steps):
  alignments = []
  for i, A_G in enumerate(listA_G): # for each graph in the corpus, find its best alignment with current centroid
    initial_state = np.eye(np.shape(A_G)[0]) # initial state is identity means we're doing the alignment with whatever A_G currently is
    graph_aligner = GraphAlignmentAnnealer(initial_state, A_g, A_G, node_mapping)
    graph_aligner.Tmax = Tmax
    graph_aligner.Tmin = Tmin
    graph_aligner.steps = steps
    # each time we make the new alignment annealer at each step of the centroid annealer, we want to UPDATE THE TEMPERATURE PARAM (decrement it at each step)
    # and can try decreasing number of iterations each time as well
    alignment, cost = graph_aligner.anneal() # don't do auto scheduling, it does not appear to work at all
    print(f"ALIGNMENT COST{i}", cost, "|||")
    alignments.append(alignment)
  return alignments

class CentroidAnnealer(Annealer):
  def __init__(self, initial_centroid, listA_G, centroid_node_mapping):
    super(CentroidAnnealer, self).__init__(initial_centroid)
    self.listA_G = listA_G
    self.centroid_node_mapping = centroid_node_mapping
    self.step = 0

  def parse_node_name(self, node_name):
    # Prototype nodes of the form "PrS{n}" or "PrP{n}"
    proto_match = re.match(r"Pr([SP])(\d+)", node_name)
    if proto_match:
      return {
        "type": "prototype",
        "kind": proto_match.group(1),
        "n": int(proto_match.group(2)),
      }
    
    # Instance nodes of the form "S{n1}L{n2}N{n3}" or "P{n1}O{n2}N{n3}"
    instance_match = re.match(r"([SP])(\d+)L?O?(\d+)N(\d+)", node_name)
    if instance_match:
      return {
        "type": "instance",
        "kind": instance_match.group(1),
        "n1": int(instance_match.group(2)),
        "n2": int(instance_match.group(3)),
        "n3": int(instance_match.group(4)),
      }
    
    # If the node name does not match any known format
    return {
      "type": "unknown",
      "name": node_name
    }
  
  # i.e. this always makes the score worse, it's not an intermediate invalid state that could lead to a better valid state
  def is_valid_move(self, source_idx, sink_idx, node_mapping):
    # There would be a self-loop if we flip this coordinate
    if source_idx == sink_idx and self.state[source_idx, sink_idx] == 0:
      return False

    source_info = self.parse_node_name(node_mapping[source_idx])
    sink_info = self.parse_node_name(node_mapping[sink_idx])

    # The edge is from an instance to a prototype 
    if source_info['type'] == 'instance' and sink_info['type'] == 'prototype':
      return False
    
    # The edge is between two prototypes
    if source_info['type'] == 'prototype' and sink_info['type'] == 'prototype':
      return False
    
    # The edge is from the wrong prototype to an instance (i.e. PrP{n} to S{n1}L{n2}N{n3} or PrS{n} to P{n1}O{n2}N{n3})
    if source_info['type'] == 'prototype' and sink_info['type'] == 'instance' and source_info['kind'] != sink_info['kind']:
      return False
    
    # The edge is from a lower level to a higher level instance node (so either from P{n1}O{n2}N{n3} to S{n1}L{n2}N{n3}, or from S{n1}L{n2}N{n3} to S{n1'}L{n2'}N{n3'} where n2 > n2')
    if source_info['type'] == 'instance' and sink_info['type'] == 'instance':
      if source_info['kind'] == 'P' and sink_info['kind'] == 'S':
        return False
      if source_info['kind'] == 'S' and sink_info['kind'] == 'S' and source_info['n2'] > sink_info['n2']:
        return False

    return True
  
  def move(self):
    valid_move_found = False
    attempt_index = 0

    # Calculate the matrices only once for efficiency
    diff_matrices = np.array([self.state - A_G for A_G in self.listA_G])
    difference_matrix = np.mean(diff_matrices, axis=0)
    variance_matrix = np.var(diff_matrices, axis=0)
    score_matrix = np.abs(difference_matrix) * variance_matrix 
    
    # Flatten the score matrix to sort scores
    flat_indices_sorted_by_score = np.argsort(score_matrix, axis=None)[::-1]

    while not valid_move_found and attempt_index < len(flat_indices_sorted_by_score):
      flat_index = flat_indices_sorted_by_score[attempt_index]
      coord = np.unravel_index(flat_index, score_matrix.shape)
      source_idx, sink_idx = coord
      valid_move_found = self.is_valid_move(source_idx, sink_idx, self.centroid_node_mapping)
      if not valid_move_found:
        attempt_index += 1

    if valid_move_found:
      self.state[source_idx, sink_idx] = 1 - self.state[source_idx, sink_idx] 
      self.step += 1
    else:
      print("No valid move found.")

  def energy(self): # i.e. cost, self.state represents the permutation/alignment matrix a
    # Calculate the current temperature ratio of the centroid annealer
    current_temp_ratio = (self.T - self.Tmin) / (self.Tmax - self.Tmin)
    
    # Define params get_alignments_to_centroid
    initial_Tmax = 1
    final_Tmax = 0.05
    initial_steps = 50
    final_steps = 5
    
    # Alignment annealer params Tmax and steps are dynamic based on the current temperature ratio for the centroid
    # They get narrower as we get an increasingly more accurate centroid that's easier to align
    alignment_Tmax = initial_Tmax * current_temp_ratio + final_Tmax * (1 - current_temp_ratio)
    alignment_steps = int(initial_steps * current_temp_ratio + final_steps * (1 - current_temp_ratio))
    
    alignments = get_alignments_to_centroid(self.state, self.listA_G, self.centroid_node_mapping, alignment_Tmax, 0.01, alignment_steps)
    
    # Align the corpus to the current centroid
    self.listA_G = list(map(align, alignments, self.listA_G))
    l = loss(self.state, self.listA_G) 
    print("LOSS", l, "\n")
    return l

(g, _, _) = build_graph.generate_graph('/Users/ilanashapiro/Documents/constraints_project/project/LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short_segments.txt', '/Users/ilanashapiro/Documents/constraints_project/project/LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short_motives.txt')
(G, _, _) = build_graph.generate_graph('/Users/ilanashapiro/Documents/constraints_project/project/LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short_segments.txt', '/Users/ilanashapiro/Documents/constraints_project/project/LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short_motives.txt')
# list_G = [tests.G1, tests.G2]
list_G = [g, G]
listA_G, centroid_node_mapping = helpers.pad_adj_matrices(list_G)
initial_centroid = listA_G[0] #random.choice(listA_G) # initial centroid. random for now, can improve later
# alignments = get_alignments_to_centroid(initial_centroid, listA_G, centroid_node_mapping, 2.5, 0.01, 10000)

# for i, alignment in enumerate(alignments):
#   file_name = f'alignment_{i}.txt'
#   np.savetxt(file_name, alignment)
#   print(f'Saved: {file_name}')

alignments = [np.loadtxt('alignment_0.txt'), np.loadtxt('alignment_1.txt')]
aligned_listA_G = list(map(align, alignments, listA_G))

centroid_annealer = CentroidAnnealer(initial_centroid, aligned_listA_G, centroid_node_mapping)
centroid_annealer.Tmax = 2.5
centroid_annealer.Tmin = 0.05 
centroid_annealer.steps = 300
centroid, min_loss = centroid_annealer.anneal()

centroid, centroid_node_mapping = helpers.remove_dummy_nodes(centroid, centroid_node_mapping)

np.savetxt("centroid.txt", centroid)
print('Saved: centroid.txt')
with open("centroid_node_mapping.txt", 'w') as file:
  json.dump(centroid_node_mapping, file)
print('Saved: centroid_node_mapping.txt')
print("Best centroid", centroid)
print("Best loss", min_loss)
