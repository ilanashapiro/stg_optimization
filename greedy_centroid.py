import networkx as nx
import numpy as np
from simanneal import Annealer
import random

class GraphAlignmentAnnealer(Annealer):
  def __init__(self, initial_state, A_g, A_G):
    super(GraphAlignmentAnnealer, self).__init__(initial_state)
    self.A_g = A_g
    self.A_G = A_G

  def move(self):
    """Swaps two nodes in the permutation."""
    a = random.randint(0, len(self.state) - 1)
    b = random.randint(0, len(self.state) - 1)
    self.state[a], self.state[b] = self.state[b], self.state[a]

  def energy(self): # i.e. cost
    """Calculates the objective function of the current state."""
    # self.state represents the permutation/alignment matrix a
    return dist(self.A_g, self.state, self.A_g)
    
# current centroid g, list of alignments list_a to the graphs in the corpus list_G
# loss is the sum of the distances between current centroid g and each graph in corpus G,
  # based on the current alignments
# this is our objective we're trying to minimize
def loss(g, list_a, list_G):
  loss = 0
  for a, G in zip(list_a, list_G):
    loss += dist(g, a, G)
  return loss

# dist between g and G given alignment a
# i.e. reorder nodes of G according to alignment (i.e. permutation matrix) a
# ||A_g - a^t * A_G * a|| where ||.|| is the norm (using Frobenius norm)
def dist(A_g, a, A_G):
  # a^T * A_G * a using numpy
  alignedA_G = a.T @ A_G @ a
  return np.linalg.norm(A_g - alignedA_G, 'fro')

def improve(g, list_a, list_G):
  while (True):
    improved_list_a = []
    for a, G in zip(list_a, list_G):
      a = improve_a(g, a, G)
      improved_list_a.append(a)
    g = improve_g(g, improved_list_a, list_G)
    list_a = improved_list_a
    print(loss(g, list_a, list_G))

# e.g. flip 2 edges that most improve dist for that alignment
# might need to iterate thru all edges
# this is where I say what's a valid transform, e.g. can't
  # add edge between non-adj levels
# highly flexible step 
def improve_a(g, a, G):

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
def improve_g(g, list_a, list_G):

