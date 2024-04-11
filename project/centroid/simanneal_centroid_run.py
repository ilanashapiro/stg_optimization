import simanneal_centroid
import numpy as np

def align_graph_pair(A_G1, A_G2, index_node_mapping, Tmax = 1.25, Tmin = 0.01, steps = 5000):
  if A_G1.shape != A_G2.shape:
    raise ValueError("Graphs must be of the same size to align.")
  initial_state = np.eye(np.shape(A_G1)[0]) # or A_G2
  graph_aligner = simanneal_centroid.GraphAlignmentAnnealer(initial_state, A_G1, A_G2, index_node_mapping)
  graph_aligner.Tmax = Tmax
  graph_aligner.Tmin = Tmin 
  graph_aligner.steps = steps 
  alignment, cost = graph_aligner.anneal()
  return alignment, cost

def optimal_alignments_for_graph_list(A_g, graph_list, index_node_mapping):
  aligned_graph_list = []
  list_alignments = []
  for A_G in graph_list:
    if np.array_equal(A_g, A_G):
      aligned_graph_list.append(A_g)
      list_alignments.append(np.eye(A_g.shape[0]))
      continue
    alignment, _ = align_graph_pair(A_g, A_G, index_node_mapping)
    aligned_A_G = simanneal_centroid.align(alignment, A_G)
    aligned_graph_list.append(aligned_A_G)
    list_alignments.append(alignment)
  loss = simanneal_centroid.loss(A_g, aligned_graph_list)
  return loss, list_alignments

# find the graph in the corpus that has the overall minimum loss to all the other graphs in the corpus,
# along with its optimal alignments
def initial_centroid_and_alignments(list_of_graphs, index_node_mapping):
  min_loss = np.inf
  min_loss_graph = None
  optimal_alignments = []
  for A_g in list_of_graphs:
    current_loss, alignments = optimal_alignments_for_graph_list(A_g, list_of_graphs, index_node_mapping)
    if current_loss < min_loss:
      min_loss = current_loss
      min_loss_graph = A_g
      optimal_alignments = alignments
  return min_loss_graph, min_loss, optimal_alignments
