import networkx as nx
import numpy as np
import random
import re 
from simanneal import Annealer
from centroid_anneal import CustomCentroidAnnealer
import sys
import json 
import multiprocessing
import pickle
from collections import defaultdict

import simanneal_centroid_tests as tests
import simanneal_centroid_helpers as helpers

EPSILON = 1e-8
DIRECTORY = '/home/ilshapiro/project'
DIRECTORY = '/Users/ilanashapiro/Documents/constraints_project/project'
sys.path.append(DIRECTORY)
import build_graph

'''
Simulated Annealing (SA) Combinatorial Optimization Approach
1. Use SA to find optimal alignments between current centroid and each graph in corpus
2. Use the resulting average difference adjacency matrix between centroid and corpus to select the best (valid) transform. 
3. Modify centroid and repeat until loss converges. Loss is sum of dist from centroid to seach graph in corpus
'''

def align(P, A_G):
	return P.T @ A_G @ P

# dist between g and G given alignment a
# i.e. reorder nodes of G according to alignment (i.e. permutation matrix) a
# ||A_g - a^t * A_G * a|| where ||.|| is the norm (using Frobenius norm)
def dist(A_g, A_G):
	return np.linalg.norm(A_g - A_G, 'fro')

class GraphAlignmentAnnealer(Annealer):
	def __init__(self, initial_alignment, A_g, A_G, centroid_idx_node_mapping, node_metadata_dict):
		super(GraphAlignmentAnnealer, self).__init__(initial_alignment)
		self.A_g = A_g
		self.A_G = A_G
		self.centroid_idx_node_mapping = centroid_idx_node_mapping
		self.node_metadata_dict = node_metadata_dict
		self.node_partitions = self.get_node_partitions()
		
	# this prevents us from printing out alignment annealing updates since this gets confusing when also doing centroid annealing
	# def default_update(self, step, T, E, acceptance, improvement):
	# 	return 
	
	# return: (partition name, sub-level (if any))
	# partition name is the layer for instance nodes
	def get_node_partition_info(self, node_id):
		def get_layer_id(node_id):
			for layer_id in ['S', 'P', 'K', 'C', 'M']:
				if node_id.startswith(layer_id):
					return layer_id
			raise Exception("Invalid node", node_id)
		
		if node_id.startswith('Pr'): # prototype nodes: one partition per prototype feature
			# EVEN THOUGH IT'S POSOSIBLE FOR THE BEST ALIGNMENT TO MIX ACROSS FEATURE SETS OF THE SAME LEVEL, FOR LARGER GRAPHS THIS IS HIGHLY UNLIKELY
			# EXPERIMENTAL RESULTS SHOW GREATER ACCURACY BY PARTITIONING PROTOS BY FEATURE SETS, INSTEAD OF MERGED SOURCE LAYER SETS, DUE TO THIS UNLIKELIHOOD AND THE INCREASED 
			# ANNEALING EFFICIENCY ACHIEVED BY THE SMALLER PARTITIONS
			feature = self.node_metadata_dict[node_id]['feature_name']
			feature = feature if 'filler' not in feature else get_layer_id(node_id)
			return ('proto_' + feature, None)
			# VERSION WITH MERGED FEATURES INTO SOURCE LAYER PARTITIONS
			# 	source_layer_kind = self.node_metadata_dict[node_id]['source_layer_kind']
			# 	return ('proto_' + source_layer_kind, None)
		else: # instance nodes: one partition per layer kind e.g. P or S or C etc (fillers of that layer are included)
			layer_id = get_layer_id(node_id)
			hierarchical_layers = ['S']
			if layer_id in hierarchical_layers:
				sublevel = re.search(r'L(\d+)', node_id).group(1)
				return ("inst_" + layer_id, sublevel)
			return ("inst_" + layer_id, None)
	
	def get_node_partitions(self):
		"""Partition centroid_idx_node_mapping into labeled sets."""
		partitions = {}
		for index, node_id in self.centroid_idx_node_mapping.items():
			partition_name, layer = self.get_node_partition_info(node_id)
			hierarchical_layers = ['inst_S']

			if partition_name not in partitions:
					partitions[partition_name] = {} if partition_name in hierarchical_layers else []
	
			if partition_name == 'inst_S': # segmentation possibly has a sub-hierarchy
				if layer not in partitions[partition_name]:
					partitions[partition_name][layer] = []
				partitions[partition_name][layer].append(index)
			else:
				partitions[partition_name].append(index)
		
		return partitions

	def move(self):
		"""Swaps two rows in the n x n permutation matrix by permuting within valid sets (protype node class or individual level)"""
		n = len(self.state)
		i = random.randint(0, n - 1)
		i_partition_name, i_sublevel = self.get_node_partition_info(self.centroid_idx_node_mapping[i])
		j_options = None 
		hierarchical_layers = ['inst_S']

		# Identify partition and find a random j within the same partition
		if i_partition_name in hierarchical_layers and i_sublevel in self.node_partitions[i_partition_name]:
			j_options = self.node_partitions[i_partition_name][i_sublevel]
		elif i_partition_name:
			j_options = self.node_partitions[i_partition_name]

		# Ensure i is not equal to j
		if j_options and len(j_options) > 1: # if a partition has only 1 element we have infinite loop
			j = random.choice(j_options)
			while j == i: 
				j = random.choice(j_options)
		else:
			# Fallback to random selection if no suitable j is found
			j = random.randint(0, n - 1)
			while j == i:
				j = random.randint(0, n - 1)

		self.state[[i, j], :] = self.state[[j, i], :]  # Swap rows i and j

	def energy(self): # i.e. cost, self.state represents the permutation/alignment matrix a
		return dist(self.A_g, align(self.state, self.A_G))

# ---------------------------------------- TEST CODE: --------------------------------------------------------------------------------
# fp1 = DIRECTORY + '/project/datasets/chopin/classical_piano_midi_db/chpn-p9/chpn-p9_augmented_graph_flat.pickle'
# fp2 = DIRECTORY + '/project/datasets/chopin/classical_piano_midi_db/chpn-p2/chpn-p2_augmented_graph_flat.pickle'

# with open(fp1, 'rb') as f:
# 	G1 = pickle.load(f)
# with open(fp2, 'rb') as f:
# 	G2 = pickle.load(f)

# G1, G2 = tests.G1_test, tests.G2_test
# padded_matrices, centroid_idx_node_mapping, node_metadata_dict = helpers.pad_adj_matrices([G1, G2])
# A_G1, A_G2 = padded_matrices[0], padded_matrices[1]
# print(A_G1, A_G2)
# print(centroid_idx_node_mapping)
# print()
# print(nodes_metadata_dict)

# list_G = [tests.G1, tests.G2]
# listA_G, centroid_idx_node_mapping, nodes_metadata_dict = helpers.pad_adj_matrices(list_G)
# initial_centroid = listA_G[0]
# np.savetxt("initial_centroid.txt", initial_centroid)

# A_g_c = np.loadtxt('centroid.txt')
# with open("centroid_idx_node_mapping.txt", 'r') as file:
#   centroid_idx_node_mapping = json.load(file)
#   centroid_idx_node_mapping = {int(k): v for k, v in centroid_idx_node_mapping.items()}
# layers1 = build_graph.get_unsorted_layers_from_graph_by_index(tests.G1)
# layers2 = build_graph.get_unsorted_layers_from_graph_by_index(tests.G2)
# g_c = helpers.adj_matrix_to_graph(A_g_c, centroid_idx_node_mapping)
# layers_g_c = build_graph.get_unsorted_layers_from_graph_by_index(g_c)
# build_graph.visualize_p([g_c], [layers_g_c])

# initial_state = np.eye(np.shape(A_G1)[0])
# graph_aligner = GraphAlignmentAnnealer(initial_state, A_G1, A_G2, centroid_idx_node_mapping, node_metadata_dict)
# graph_aligner.Tmax = 1.25
# graph_aligner.Tmin = 0.01 
# graph_aligner.steps = 2000 # 2000 
# alignment, cost1 = graph_aligner.anneal() # don't do auto scheduling, it does not appear to work at all

# print("Best cost1", cost1)
# sys.exit(0)
# graph_aligner = GraphAlignmentAnnealer(alignment, A_G1, A_G2, centroid_idx_node_mapping, node_metadata_dict)
# graph_aligner.Tmax = 0.5 #1.25
# graph_aligner.Tmin = 0.01 
# graph_aligner.steps = 2000 #2000 
# _, cost2 = graph_aligner.anneal()

# print("Best cost2", cost2)
# print("Difference", cost2 - cost1)
# sys.exit(0)
# ---------------------------------------- :TEST CODE --------------------------------------------------------------------------------

def get_alignments_to_centroid(A_g, listA_G, node_mapping, Tmax, Tmin, steps, node_metadata_dict):
	alignments = []
	for i, A_G in enumerate(listA_G): # for each graph in the corpus, find its best alignment with current centroid
		initial_state = np.eye(np.shape(A_G)[0]) # initial state is identity means we're doing the alignment with whatever A_G currently is
		graph_aligner = GraphAlignmentAnnealer(initial_state, A_g, A_G, node_mapping, node_metadata_dict)
		graph_aligner.Tmax = Tmax
		graph_aligner.Tmin = Tmin
		graph_aligner.steps = steps
		# each time we make the new alignment annealer at each step of the centroid annealer, we want to UPDATE THE TEMPERATURE PARAM (decrement it at each step)
		# and can try decreasing number of iterations each time as well
		alignment, cost = graph_aligner.anneal() # don't do auto scheduling, it does not appear to work at all
		# print(f"ALIGNMENT COST{i}", cost, "|||")
		alignments.append(alignment)
	return alignments

def align_single_graph(args):
	A_g, A_G, node_mapping, Tmax, Tmin, steps, node_metadata_dict = args
	initial_state = np.eye(np.shape(A_G)[0])  # Identity matrix as initial state
	if np.array_equal(A_g, A_G):
		return initial_state
	graph_aligner = GraphAlignmentAnnealer(initial_state, A_g, A_G, node_mapping, node_metadata_dict)
	graph_aligner.Tmax = Tmax
	graph_aligner.Tmin = Tmin
	graph_aligner.steps = steps
	alignment, _ = graph_aligner.anneal()
	return alignment

def get_alignments_to_centroid_parallel(A_g, listA_G, node_mapping, Tmax, Tmin, steps, node_metadata_dict):
	args = [(A_g, A_G, node_mapping, Tmax, Tmin, steps, node_metadata_dict) for A_G in listA_G]
	with multiprocessing.Pool() as pool:
		alignments = pool.map(align_single_graph, args)
	return alignments

# current centroid g, list of alignments list_a to the graphs in the corpus list_G
# loss is the sum of the distances between current centroid g and each graph in corpus G,
	# based on the current alignments
# this is our objective we're trying to minimize
def loss(A_g, list_alignedA_G):
	distances = np.array([dist(A_g, A_G) for A_G in list_alignedA_G])
	distance = np.mean(distances) # unit is distance
	std = np.std(distances) # unit is also distance (vs unit of variance would be distance^2)

	# We want to square the result to moderately amplify differences between losses/energies in different states
	# we want to amplify the larger differences, but not the smaller ones
	# A large energy positive difference means very low acceptance probability (i.e. we don't want to accept a very bad state)
	# vs small positive energy difference has higher acceptance probability
	# so make sure this distinction is possible via the scale of the loss function
	print("DIST", distance, "STD", std)
	
	# we add epsilon so that if std is zero (i.e. similarity is perfect), we can still be sensitive to changes in the distance
	return (distance * (std + EPSILON)) ** 2 

class CentroidAnnealer(CustomCentroidAnnealer):
	def __init__(self, initial_centroid, listA_G, centroid_idx_node_mapping, node_metadata_dict):
		super(CentroidAnnealer, self).__init__(initial_centroid) # i.e. set initial self.state = initial_centroid
		self.listA_G = listA_G
		self.centroid_idx_node_mapping = centroid_idx_node_mapping
		self.node_metadata_dict = node_metadata_dict
		self.step = 0
		self.rejected_moves_since_last_accept = []
		self.last_accepted_move = None
		self.prev_move = None

	# this prevents us from printing out alignment annealing updates since this gets confusing when also doing centroid annealing
	def default_update(self, step, T, E, acceptance, improvement):
		return 
	
	# i.e. the move always makes the score worse, it's not an intermediate invalid state that could lead to a better valid state
	def is_globlly_invalid_move(self, source_idx, sink_idx, node_mapping):
		# No self-loops (there would be a self-loop if we flip this coordinate)
		if source_idx == sink_idx and self.state[source_idx, sink_idx] == 0:
			return True
		
		source_node_id = node_mapping[source_idx]
		sink_node_id = node_mapping[sink_idx]

		def is_proto(node_id):
			return node_id.startswith('Pr')
		
		# The edge is from an instance to a prototype 
		if not is_proto(source_node_id) and is_proto(sink_node_id):
			return True
		
		# The edge is between two prototypes
		if is_proto(source_node_id) and is_proto(sink_node_id):
			return True
		
		# The edge is from a prototype to an instance level whose nodes don't have that prototype feature (i.e. PrAbs_interval -> segmentation)
		if is_proto(source_node_id) and not is_proto(sink_node_id):
			source_proto_feature = self.node_metadata_dict[source_node_id]['feature_name']
			sink_inst_features = self.node_metadata_dict[sink_node_id]['features_dict'].keys()
			if source_proto_feature not in sink_inst_features:
				return True
		
		# Source/sink are both instance
		if not is_proto(source_node_id) and not is_proto(sink_node_id):
			def rank_difference(rank1, rank2):
				primary_rank1, secondary_rank1 = rank1
				primary_rank2, secondary_rank2 = rank2
				if primary_rank1 == primary_rank2:
					return secondary_rank1 - secondary_rank2
				return primary_rank1 - primary_rank2
			
			# source level is NOT one level higher (i.e. 1 rank lower) or is NOT the same level than sink level
			# NOTE: this ONLY works for when we have a fixed number of levels in the graph. if sub-hierarchies are variable levels, then it's totally possible
			# to have an intermediate valid move of higher source->lower sink level that's not adjacent, like if we're in the process of deleting a level
			# but if all graphs have the same number of levels, like with flat segmentation or scluster, this won't happen, and hence we can add this important optimization
			# SO THIS MEANS WE DO NOT SUPPORT VARIABLE LEVEL SUB-HIERARCHIES
			# want difference = 0 (i.e. same source/sink level) or -1 (i.e. source level is one above sink level. higher level means lower rank value)
			source_rank = self.node_metadata_dict[source_node_id]['layer_rank']
			sink_rank = self.node_metadata_dict[sink_node_id]['layer_rank']
			if rank_difference(source_rank, sink_rank) not in [0, -1]:
				return True
		
		return False
	
	def move(self):
		diff_matrices = np.abs(np.array([self.state - A_G for A_G in self.listA_G]))
		difference_matrix = np.mean(diff_matrices, axis=0)
		std_matrix = np.std(diff_matrices, axis=0) 
		# we add epsilon so that if std is zero (i.e. similarity is perfect), we can still be sensitive to changes in the distance
		score_matrix = difference_matrix * (std_matrix + EPSILON) # don't need to square as we do in loss because this won't change the score ordering, it just would scale it, not necessary
		
		# Flatten the score matrix to sort scores highest->lowest
		flat_scores = score_matrix.flatten()
		flat_indices_sorted_by_score = np.argsort(flat_scores)[::-1]

		# Create a dictionary to group indices in the score matrix by score
		score_index_mapping = defaultdict(list)
		for flat_index in flat_indices_sorted_by_score:
			score = flat_scores[flat_index]
			score_index_mapping[score].append(flat_index)
		
		unique_scores_descending = sorted(score_index_mapping.keys(), reverse=True) # highest -> lowest score
		# randomly shuffle the indices for each score partition, so we're trying a more variable set of moves that equally/most contribute to the loss
		# this helps the annear be less stuck and explore a wider variety of equally possible moves
		flat_indices_sorted_by_score_and_shuffled = [] 
		for score in unique_scores_descending:
			indices = score_index_mapping[score]
			random.shuffle(indices) # Shuffle indices in the current score partition
			flat_indices_sorted_by_score_and_shuffled.extend(indices)

		valid_move_found = False
		attempt_index = 0
		while not valid_move_found and attempt_index < len(flat_indices_sorted_by_score_and_shuffled):
			flat_index = flat_indices_sorted_by_score_and_shuffled[attempt_index]
			coord = np.unravel_index(flat_index, score_matrix.shape)
			source_idx, sink_idx = coord
			move_not_globally_invalid = not self.is_globlly_invalid_move(source_idx, sink_idx, self.centroid_idx_node_mapping)
			have_not_already_tried_move = coord not in self.rejected_moves_since_last_accept
			is_not_undoing_last_accept = coord != self.last_accepted_move
			if is_not_undoing_last_accept and have_not_already_tried_move and move_not_globally_invalid:
				valid_move_found = True
			else:
				attempt_index += 1

		if valid_move_found:
			print("Coord", coord, "State at coord", self.state[source_idx, sink_idx])
			print("Rejected moves since last accept", self.rejected_moves_since_last_accept)
			print("Last accepted move", self.last_accepted_move)
			self.state[source_idx, sink_idx] = 1 - self.state[source_idx, sink_idx] 
			self.prev_move = coord
			self.step += 1
		else:
			print("No valid move found.")

	def energy(self): # i.e. cost, self.state represents the current centroid g
		current_temp_ratio = (self.T - self.Tmin) / (self.Tmax - self.Tmin)
		initial_Tmax = 1
		final_Tmax = 0.05
		initial_steps = 100
		final_steps = 5
		
		# Alignment annealer params Tmax and steps are dynamic based on the current temperature ratio for the centroid
		# They get narrower as we get an increasingly more accurate centroid that's easier to align
		alignment_Tmax = initial_Tmax * current_temp_ratio + final_Tmax * (1 - current_temp_ratio)
		alignment_steps = int(initial_steps * current_temp_ratio + final_steps * (1 - current_temp_ratio))
		alignments = get_alignments_to_centroid(self.state, self.listA_G, self.centroid_idx_node_mapping, alignment_Tmax, 0.01, alignment_steps, node_metadata_dict)
		
		# Align the corpus to the current centroid
		self.listA_G = list(map(align, alignments, self.listA_G))
		l = loss(self.state, self.listA_G) 
		print("LOSS", l)
		return l

if __name__ == "__main__":
	fp1 = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_461_(c)orlandi/biamonti_461_(c)orlandi_augmented_graph_flat.pickle'
	fp2 = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_811_(c)orlandi/biamonti_811_(c)orlandi_augmented_graph_flat.pickle'
	# fp1 = DIRECTORY + '/datasets/bach/kunstderfuge/bwv876frag/bwv876frag_augmented_graph_hier.pickle'
	# fp2 = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_459_(c)orlandi/biamonti_459_(c)orlandi_augmented_graph_hier.pickle'

	with open(fp1, 'rb') as f:
		G1 = pickle.load(f)
	with open(fp2, 'rb') as f:
		G2 = pickle.load(f)

	# list_G = [tests.G1_test, tests.G2_test]
	list_G = [G1, G2]
	listA_G, centroid_idx_node_mapping, node_metadata_dict = helpers.pad_adj_matrices(list_G)
	initial_centroid = listA_G[0] #random.choice(listA_G) # initial centroid. random for now, can improve later
	
	# alignments = get_alignments_to_centroid(initial_centroid, listA_G, centroid_idx_node_mapping, 2.5, 0.01, 10000, node_metadata_dict)
	# for i, alignment in enumerate(alignments):
	# 	file_name = f'alignment_{i}.txt'
	# 	np.savetxt(file_name, alignment.astype(int), fmt='%i', delimiter=",")
	# 	print(f'Saved: {file_name}')

	# alignments = [np.loadtxt('alignment_0.txt', dtype=int, delimiter=","), np.loadtxt('alignment_1.txt', dtype=int, delimiter=",")]
	# aligned_listA_G = list(map(align, alignments, listA_G))

	# centroid_annealer = CentroidAnnealer(initial_centroid, aligned_listA_G, centroid_idx_node_mapping, node_metadata_dict)
	# centroid_annealer.Tmax = 4.5
	# centroid_annealer.Tmin = 0.05 
	# centroid_annealer.steps = 500
	# centroid, min_loss = centroid_annealer.anneal()

	# centroid, centroid_idx_node_mapping = helpers.remove_dummy_nodes(centroid, centroid_idx_node_mapping, node_metadata_dict)
	# np.savetxt("approx_centroid_test.txt", centroid)
	# print('Saved: approx_centroid_test.txt')
	# with open("approx_centroid_idx_node_mapping_test.txt", 'w') as file:
	# 	json.dump(centroid_idx_node_mapping, file)
	# print('Saved: approx_centroid_idx_node_mapping_test.txt')
	# with open("approx_centroid_node_metadata_test.txt", 'w') as file:
	# 	json.dump(node_metadata_dict, file)
	# print('Saved: approx_centroid_node_metadata_test.txt')
	# print("Best centroid", centroid)
	# print("Best loss", min_loss)
	# sys.exit(0)

	centroid = np.loadtxt("approx_centroid_test.txt")
	with open("approx_centroid_idx_node_mapping_test.txt", 'r') as file:
		centroid_idx_node_mapping = {int(k): v for k, v in json.load(file).items()}
	
	g = helpers.adj_matrix_to_graph(centroid, centroid_idx_node_mapping, node_metadata_dict)
	
	layers_G1 = build_graph.get_unsorted_layers_from_graph_by_index(G1)
	layers_G2 = build_graph.get_unsorted_layers_from_graph_by_index(G2)
	layers_g = build_graph.get_unsorted_layers_from_graph_by_index(g)
	build_graph.visualize_p([G1, G2, g], [layers_G1, layers_G2, layers_g])