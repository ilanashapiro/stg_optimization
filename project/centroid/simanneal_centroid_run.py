import simanneal_centroid
import multiprocessing
import numpy as np

def align_graph_pair(A_G1, A_G2, idx_node_mapping, node_metadata_dict, Tmax = 1.5, Tmin = 0.01, steps = 2000):
  if A_G1.shape != A_G2.shape:
    raise ValueError("Graphs must be of the same size to align.")
  initial_state = np.eye(np.shape(A_G1)[0]) # or A_G2
  graph_aligner = simanneal_centroid.GraphAlignmentAnnealer(initial_state, A_G1, A_G2, idx_node_mapping, node_metadata_dict)
  graph_aligner.Tmax = Tmax
  graph_aligner.Tmin = Tmin 
  graph_aligner.steps = steps 
  alignment, cost = graph_aligner.anneal()
  return alignment, cost

def align_graph_pair_wrapper(args):
  return align_graph_pair(*args)

# I think this function is probably doing the same thing as get_alignments_to_centroid in simanneal_centroid
# TODO: remove this function
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

def optimal_alignments_for_graph_list_parallel(A_g, graph_list, index_node_mapping):
  aligned_graph_list = []
  list_alignments = []
  tasks = [(A_g, A_G, index_node_mapping) for A_G in graph_list if not np.array_equal(A_g, A_G)]
  
  with multiprocessing.Pool() as pool:
    results = pool.map(align_graph_pair_wrapper, tasks)
  
  # Handle the graph that is equal to A_g separately
  for A_G in graph_list:
    if np.array_equal(A_g, A_G):
      aligned_graph_list.append(A_g)
      list_alignments.append(np.eye(A_g.shape[0]))
  
  for (alignment, _), A_G in zip(results, [x for x in graph_list if not np.array_equal(A_g, x)]):
    aligned_A_G = simanneal_centroid.align(alignment, A_G)
    aligned_graph_list.append(aligned_A_G)
    list_alignments.append(alignment)
  
  loss = simanneal_centroid.loss(A_g, aligned_graph_list)
  return loss, list_alignments

# find the graph in the corpus that has the overall minimum loss to all the other graphs in the corpus,
# along with its optimal alignments
def initial_centroid_and_alignments(list_A_G, index_node_mapping):
  min_loss = np.inf
  min_loss_A_G = None
  optimal_alignments = []
  for A_g in list_A_G:
    current_loss, alignments = optimal_alignments_for_graph_list(A_g, list_A_G, index_node_mapping)
    if current_loss < min_loss:
      min_loss = current_loss
      min_loss_A_G = A_g
      optimal_alignments = alignments
  return min_loss_A_G, min_loss, optimal_alignments
