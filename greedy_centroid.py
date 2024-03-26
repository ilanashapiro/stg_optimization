import networkx as nx
import numpy as np

# current centroid g, list of alignments list_a to the graphs in the corpus list_G
# loss is the sum of the distances between current centroid g and each graph in corpus G,
  # based on the current alignments
# this is our objective we're trying to minimize
def loss(g, list_a, list_G):
  loss = 0
  for a_i, G_i in zip(list_a, list_G):
    loss += dist(g, a_i, G_i)
  return loss

# dist between g and G given alignment a
def dist(g, a, G):
  return np.sum(abs(g - reorder(G, a)))

# reorder nodes of G according to alignment a
def reorder(G, a):

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
  # you can think of 1/2 * sum_{i=1}^n A_{a_i}(G_i) as the "average" adj matrix of list_G
# then, take the abs val |A'| and pick from there the transform with the biggeset val/distance
  # and consult the original A' for the direction/size
# also, note that we have padded each A s.t. all A's are the same dimensions (i.e. size of max graph in G)
  # this means we can never add a node w/o a parent or child (need to refine so only can't add node with no parent)
def improve_g(g, list_a, list_G):

def improve(g, list_a, list_G):
