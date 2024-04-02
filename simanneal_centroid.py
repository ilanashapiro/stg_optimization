import networkx as nx
import numpy as np
from simanneal import Annealer
import random

import simanneal
import simanneal_centroid_tests as tests
import simanneal_centroid_helpers as helpers
import sys
import re 

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
  
  # def move(self): # naive implementation, more inefficient than pre-partitioning
  #   """Swaps two rows in the n x n permutation matrix, ensuring i != j."""
  #   n = len(self.state)
  #   i = random.randint(0, n - 1)
  #   i_kind, i_layer = self.get_node_info(self.centroid_node_mapping[i])

  #   # Try to find j options that match node type and layer (if it's a segmentation node)
  #   j_options = [index for index, node_id in self.centroid_node_mapping.items() 
  #                 if self.get_node_info(node_id)[0] == i_kind and
  #                   (self.get_node_info(node_id)[1] == i_layer or i_layer is None) and 
  #                   index != i]
  
  #   if j_options:
  #     j = random.choice(j_options) 
  #   else:
  #     j = i
  #     while j == i:
  #       j = random.randint(0, n - 1)
    
  #   self.state[[i, j], :] = self.state[[j, i], :] # Swap rows i and j

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
    if j_options:
      j_options = [option for option in j_options if option != i]
      if len(j_options) > 0:
        j = random.choice(j_options)
      else:
        # Fallback to random selection if no suitable j is found
        j = i
        while j == i:
          j = random.randint(0, n - 1)
    # if i_kind in ['prototype_segmentation', 'prototype_motif']:
    #   print(self.centroid_node_mapping[i], self.centroid_node_mapping[j])
    self.state[[i, j], :] = self.state[[j, i], :]  # Swap rows i and j

  def energy(self): # i.e. cost
    """Calculates the objective function of the current state."""
    # self.state represents the permutation/alignment matrix a
    return dist(self.A_g, align(self.state, self.A_G))

g, G = tests.G1, tests.G2
padded_matrices, centroid_node_mapping = helpers.pad_adj_matrices([g, G])
A_g, A_G = padded_matrices[0], padded_matrices[1]

initial_state = A_G # random_alignment(np.shape(A_g)[0]) --> choosing strategic seed A_G gives better result than random or A_g
graph_aligner = GraphAlignmentAnnealer(initial_state, A_g, A_G, centroid_node_mapping)
min_cost = np.inf
best_alignment = None
runs = 1
for _ in range(runs):
  alignment, cost = graph_aligner.anneal() # don't do auto scheduling, it does not appear to work at all
  if cost < min_cost:
    min_cost = cost
    best_alignment = alignment
    print("Best cost", cost)

# simanneal_runs is in order to ensure we're not stuck in local minima, tweak as needed
def get_alignments_to_centroid(A_g, listA_G, node_mapping, simanneal_runs=5):
  alignments = []
  for A_G in listA_G: # for each graph in the corpus, find its best alignment with current centroid
    initial_state = A_g # random_alignment(np.shape(A_g)[0]) --> choosing strategic seed A_G gives better result than random or A_g
    graph_aligner = GraphAlignmentAnnealer(initial_state, A_g, A_G, node_mapping)
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
    self.list_alignedA_G = None
    self.centroid_node_mapping = centroid_node_mapping

  def move(self):
    # 1/n * sum_{i=1}^n A_{a_i}(G_i)
    # A_{a_i}(G_i) is the adjacency matrix for G_i given alignment a_i, and A_g is the adj matrix for centroid g
    assert isinstance(self.list_alignedA_G, list)
    avgA_G = np.sum(np.array(self.list_alignedA_G), axis=0) / len(self.list_alignedA_G)
    difference_matrix = self.state - avgA_G
    largest_diff = np.max(difference_matrix)
    indices_max = np.argwhere(difference_matrix == largest_diff)
    coord = random.choice(indices_max) # Randomly select a coordinate with a max value to do the transform on
    self.state[coord[0], coord[1]] = 1 - self.state[coord[0], coord[1]]
    # print(difference_matrix)
    # Choose a random row and column index
    # i = random.randint(0, self.state.shape[0] - 1)
    # j = random.randint(0, self.state.shape[1] - 1)

    # # Flip the value at the selected position
    # self.state[i, j] = 1 - self.state[i, j]

  def energy(self): # i.e. cost
    """Calculates the objective function of the current state."""
    # self.state represents the permutation/alignment matrix a
    # def loss(A_g, aligned_listA_G):
    alignments = get_alignments_to_centroid(self.state, self.listA_G, self.centroid_node_mapping)
    self.list_alignedA_G = list(map(align, alignments, self.listA_G))
    x = loss(self.state, self.list_alignedA_G) 
    print("LOSS", x)
    return x
  
def greedy_centroid(list_G, threshold):
  listA_G, centroid_node_mapping = helpers.pad_adj_matrices(list_G)
  A_g = random.choice(listA_G) # initial centroid. random for now, can improve later
  loss = np.inf
  
  while (loss > threshold):
    alignments = get_alignments_to_centroid(A_g, listA_G, centroid_node_mapping)
    list_alignedA_G = list(map(align, alignments, listA_G))
    loss = improve(A_g, list_alignedA_G, centroid_node_mapping)

def improve(A_g, list_alignedA_G, centroid_node_mapping):
  # 1/n * sum_{i=1}^n A_{a_i}(G_i)
  # A_{a_i}(G_i) is the adjacency matrix for G_i given alignment a_i, and A_g is the adj matrix for centroid g
  avgA_G = np.sum(np.array(list_alignedA_G), axis=0) / len(list_alignedA_G)
  difference_matrix = A_g - avgA_G
  print(difference_matrix, avgA_G)
  sys.exit(0)

# list_G = [tests.G1, tests.G2]
# listA_G, centroid_node_mapping = helpers.pad_adj_matrices(list_G)
# initial_centroid = random.choice(listA_G) # initial centroid. random for now, can improve later
# centroid_annealer = CentroidAnnealer(initial_centroid, listA_G, centroid_node_mapping)
# centroid, min_loss = centroid_annealer.anneal()
# print("Best centroid", centroid)
# print("Best cost", min_loss)

# now, the alignments are fixed. 
# with these alignments, we want to execute the "proposal transform" t that minimizes the losses
# to choose t, look at the adj matrices of g and G1...Gn=list_G. Note that list_G has already 
  # been changed in improve_alignments to maximize alignment
# take the overall distance measure A_g - 1/2 * sum_{i=1}^n A_{a_i}(G_i) = A'
  # where A_{a_i}(G_i) is the adjacency matrix for G_i given alignment a_i, and A_g is the adj matrix for g
  # you can think of 1/n * sum_{i=1}^n A_{a_i}(G_i) as the "average" adj matrix of list_G
# then, take the abs val |A'| and pick from there the transform with the biggeset val/distance
  # and consult the original A' for the direction/size
# also, note that we have padded each A s.t. all A's are the same dimensions (i.e. size of max graph in G)
  # this means we can never add a node w/o a parent or child (need to refine so only can't add node with no parent)
# def improve_g(g, list_a, list_G):

