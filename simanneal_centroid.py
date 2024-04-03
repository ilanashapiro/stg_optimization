import networkx as nx
import numpy as np
import random
import re 
from simanneal import Annealer
import sys

import simanneal_centroid_tests as tests
import simanneal_centroid_helpers as helpers
import build_graph

'''
Simulated Annealing (SA) Combinatorial Optimization Approach
1. Use SA to find optimal alignments between current centroid and each graph in corpus
2. Use the resulting average difference adjacency matrix between centroid and corpus to select the best (valid) transform. 
3. Modify centroid and repeat until loss converges. Loss is sum of dist from centroid to seach graph in corpus
'''

# current centroid g, list of alignments list_a to the graphs in the corpus list_G
# loss is the sum of the distances between current centroid g and each graph in corpus G,
  # based on the current alignments
# this is our objective we're trying to minimize
def loss(A_g, list_alignedA_G):
  return sum([dist(A_g, A_G) for A_G in list_alignedA_G])

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

g, G = tests.G1, tests.G2
# (g, layers, _) = build_graph.generate_graph('LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short_segments.txt', 'LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short_motives.txt')
# (G, _, _) = build_graph.generate_graph('LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short_segments.txt', 'LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short_motives.txt')
padded_matrices, centroid_node_mapping = helpers.pad_adj_matrices([g, G])
A_g, A_G = padded_matrices[0], padded_matrices[1]

A_g = np.array([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
 [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]])
layers = build_graph.get_unsorted_layers_from_graph_by_index(g)
g1 = helpers.adj_matrix_to_graph(A_g, centroid_node_mapping)
layers1 = build_graph.get_unsorted_layers_from_graph_by_index(g1)
build_graph.visualize_p([g, g1], [layers, layers1])

initial_state = A_G # random_alignment(np.shape(A_g)[0]) --> choosing strategic seed A_G gives better result than random or A_g
graph_aligner = GraphAlignmentAnnealer(initial_state, A_g, A_G, centroid_node_mapping)
graph_aligner.Tmax = 2.5
graph_aligner.Tmin = 0.01 
graph_aligner.steps = 10000 # ~19.5 energy on complete test graphs c. 1min 10sec, 5000 gives ~20.2 energy on complete test graphs but takes half the time

min_cost = np.inf
best_alignment = None
runs = 0
for _ in range(runs):
  alignment, cost = graph_aligner.anneal() # don't do auto scheduling, it does not appear to work at all
  if cost < min_cost:
    min_cost = cost
    best_alignment = alignment
    print("Best cost", cost)

# simanneal_runs is in order to ensure we're not stuck in local minima, tweak as needed
def get_alignments_to_centroid(A_g, listA_G, node_mapping, simanneal_runs=1):
  alignments = []
  for A_G in listA_G: # for each graph in the corpus, find its best alignment with current centroid
    initial_state = A_G # this is WRONG --> the rotated A_G is NOT a permutation matrix, but it encodes the permutation
    # so initial state is either the identity matrix, since A_G is always updated/aligned, or otherwise just maintain the alignments/permutations themselves
    graph_aligner = GraphAlignmentAnnealer(initial_state, A_g, A_G, node_mapping)
    # each time we make the new alignment annealer at each step of the centroid annealer, we want to UPDATE THE TEMPERATURE PARAM (decrement it at each step)
    # and can try decreasing number of iterations each time as well
    min_cost = np.inf
    best_alignment = None
    for _ in range(simanneal_runs):
      alignment, cost = graph_aligner.anneal() # don't do auto scheduling, it does not appear to work at all
      if cost < min_cost:
        min_cost = cost
        best_alignment = alignment
    alignments.append(best_alignment)
  return alignments

class CentroidAnnealer(Annealer):
  def __init__(self, initial_centroid, listA_G, centroid_node_mapping):
    super(CentroidAnnealer, self).__init__(initial_centroid)
    self.listA_G = listA_G
    self.centroid_node_mapping = centroid_node_mapping

  def move(self):
    # 1/n * sum_{i=1}^n A_{a_i}(G_i)
    # A_{a_i}(G_i) is the adjacency matrix for G_i given alignment a_i, and A_g is the adj matrix for centroid g
    avgA_G = np.sum(np.array(self.listA_G), axis=0) / len(self.listA_G)
    difference_matrix = self.state - avgA_G
    largest_diff = np.max(difference_matrix)
    indices_max = np.argwhere(difference_matrix == largest_diff)
    coord = random.choice(indices_max) # Randomly select a coordinate with a max value to do the transform on
    self.state[coord[0], coord[1]] = 1 - self.state[coord[0], coord[1]]

  def energy(self): # i.e. cost, self.state represents the permutation/alignment matrix a
    alignments = get_alignments_to_centroid(self.state, self.listA_G, self.centroid_node_mapping)
    # Align the corpus to the current centroid
    # Seems to perform better when we keep the corpus aligned to the prev centroid rather than starting
    # alignment from scratch each time
    self.listA_G = list(map(align, alignments, self.listA_G))
    return loss(self.state, self.listA_G) 

# list_G = [tests.G1, tests.G2]
# listA_G, centroid_node_mapping = helpers.pad_adj_matrices(list_G)
# initial_centroid = random.choice(listA_G) # initial centroid. random for now, can improve later
# centroid_annealer = CentroidAnnealer(initial_centroid, listA_G, centroid_node_mapping)
# centroid_annealer.Tmax = 2.5
# centroid_annealer.Tmin = 0.01 
# centroid_annealer.steps = 10
# centroid, min_loss = centroid_annealer.anneal()
# print("Best centroid", centroid)
# print("Best cost", min_loss)


