import simanneal_centroid
import multiprocessing
import numpy as np
# import cupy as cp
import torch

def align_graph_pair(A_G1, A_G2, idx_node_mapping, node_metadata_dict, Tmax = 1.75, Tmin = 0.01, steps = 2000, device=None):
  if A_G1.shape != A_G2.shape:
    raise ValueError("Graphs must be of the same size to align.")
  # initial_state = np.eye(np.shape(A_G1)[0]) # or A_G2
  initial_state = torch.eye(A_G1.shape[0], dtype=torch.float64, device=device)
  graph_aligner = simanneal_centroid.GraphAlignmentAnnealer(initial_state, A_G1, A_G2, idx_node_mapping, node_metadata_dict, device=device)#, client, cluster)
  graph_aligner.Tmax = Tmax
  graph_aligner.Tmin = Tmin 
  graph_aligner.steps = steps 
  alignment, cost = graph_aligner.anneal()
  return alignment, cost

# find the graph in the corpus that has the overall minimum loss to all the other graphs in the corpus,
# along with its optimal alignments
def initial_centroid_and_alignments(list_A_G, index_node_mapping, node_metadata_dict, device=None):
  min_loss = np.inf
  min_loss_A_G = None
  optimal_alignments = []

  for i, A_g in enumerate(list_A_G):
    assert isinstance(A_g, torch.Tensor) # Ensure that A_g is a tensor (comment out if we're not doing multiprocess)
    current_loss, alignments = simanneal_centroid.get_alignments_to_centroid(A_g, list_A_G, index_node_mapping, node_metadata_dict, device, Tmax=2, Tmin=0.01, steps=2000)
    if current_loss < min_loss:
      min_loss = current_loss
      min_loss_A_G = A_g
      min_loss_A_G_list_index = i
      optimal_alignments = alignments

  return min_loss_A_G, min_loss_A_G_list_index, min_loss, optimal_alignments
