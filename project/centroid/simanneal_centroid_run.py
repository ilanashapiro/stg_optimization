import simanneal_centroid
import multiprocessing
import numpy as np
import simanneal_centroid

# find the graph in the corpus that has the overall minimum loss to all the other graphs in the corpus,
# along with its optimal alignments
def initial_centroid_and_alignments(listA_G, index_node_mapping):
  min_loss = np.inf
  min_loss_A_G = None
  optimal_alignments = []
  for A_g in listA_G:
    alignments = simanneal_centroid.get_alignments_to_centroid_parallel(A_g, listA_G, index_node_mapping, Tmax = 1.25, Tmin = 0.01, steps = 5000)
    aligned_graph_list = list(map(simanneal_centroid.align, alignments, listA_G))
    current_loss = simanneal_centroid.loss(A_g, aligned_graph_list)
    if current_loss < min_loss:
      min_loss = current_loss
      min_loss_A_G = A_g
      optimal_alignments = alignments
  return min_loss_A_G, min_loss, optimal_alignments
