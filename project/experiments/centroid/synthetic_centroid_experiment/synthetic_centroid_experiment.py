import os, sys, pickle, json, re, random

from networkx import cost_of_flow
import test
import numpy as np
from multiprocessing import Pool, current_process
import networkx as nx
import torch, glob
from pulp import LpProblem, LpMinimize, LpVariable, lpSum
import matplotlib.pyplot as plt

DIRECTORY = "/home/ubuntu/project"
# DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
# DIRECTORY = "/home/ilshapiro/project"
TIME_PARAM = '50s'
ABLATION_LEVEL = None
EPSILON = 1e-8

sys.path.append(f"{DIRECTORY}/centroid")
import z3_matrix_projection_incremental as z3_repair
import simanneal_centroid_helpers, simanneal_centroid_run, simanneal_centroid
import build_graph

def solve_lower_bound(dist_matrix):
		"""
		Solves the lower bound LP for the generalized median graph problem.

		Parameters:
		- dist_matrix: a K x K matrix (list of lists) representing pairwise distances d(G_k, G_l)

		Returns:
		- The lower bound value from the LP solution
		- The values of x (approximated distances from the median)
		"""
		K = len(dist_matrix)  # Number of graphs
		
		# Define LP problem
		problem = LpProblem("Generalized_Median_Lower_Bound", LpMinimize)
		
		# Define variables x_i (non-negative)
		x = [LpVariable(f"x_{i}", lowBound=0) for i in range(K)]
		
		# Objective function: Minimize sum of all x_i
		problem += lpSum(x)
		
		# Add triangle inequality constraints
		for k in range(K):
				for l in range(K):
						if k != l:
								problem += x[k] + dist_matrix[k][l] >= x[l], f"Constraint_{k}_{l}_1"
								problem += x[l] + dist_matrix[k][l] >= x[k], f"Constraint_{k}_{l}_2"
								problem += x[k] + x[l] >= dist_matrix[k][l], f"Constraint_{k}_{l}_3"
		
		problem.solve()
		lower_bound_value = sum(x[i].varValue for i in range(K))
		# x_values = [x[i].varValue for i in range(K)]
		
		return lower_bound_value

def construct_distance_matrix(corpus_graphs):
		K = len(corpus_graphs)
		D = np.zeros((K, K))  # Initialize a KxK zero matrix

		for i in range(K):
				for j in range(i + 1, K):  # Only compute upper triangle to avoid redundancy
						d = struct_dist(corpus_graphs[i], corpus_graphs[j])  # Compute distance
						D[i][j] = D[j][i] = d  # Fill both symmetric entries

		return D

def plot_results():
	k_values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
	relative_errors = [
			0.009755985185740532,
			0.004949172262302578,
			0.029933014260994057,
			0.020073253229684212,
			0.017261706192896447,
			0.002681773959127958,
			0.0144659053685462,
			0.02991807085508924,
			0.0,
			0.008451775723758695,
			0.007012663638145545,
			0.008629118369398
	]
	
	plt.figure(figsize=(10, 6))
	plt.bar(k_values, relative_errors, color='skyblue', edgecolor='black')

	# Labels and title
	plt.xlabel('Corpus Size $k$', fontsize=20)
	plt.ylabel('Relative Error in Loss', fontsize=20)
	plt.tick_params(axis='both', which='major', labelsize=15)
	# plt.title('Relative Error for Different K Values')
	plt.xticks(k_values)  # Ensure all K values are shown on the x-axis

	# Show plot
	plt.show()

def load_test_centroid(test_centroid_path):
	with open(test_centroid_path, 'rb') as f:
		graph = pickle.load(f)
	return graph

def parse_node_name(node_name):
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

def is_approx_valid_move(source_node_name, sink_node_name):
		source_info = parse_node_name(source_node_name)
		sink_info = parse_node_name(sink_node_name)

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

def is_formally_valid_graph(new_STG, verbose=False):
	adj_matrix, idx_node_mapping, node_metadata_dict = simanneal_centroid_helpers.pad_adj_matrices([new_STG])
	z3_repair.initialize_globals(adj_matrix, idx_node_mapping, node_metadata_dict)
	return z3_repair.test_sat(verbose=verbose)

def add_noise_to_graph(graph, n_edits):
	# Create a copy of the original graph to avoid modifying it directly
	noisy_graph = graph.copy()
	nodes = list(noisy_graph.nodes)

	# Track changes to ensure noise doesn't decrease
	removed_edges = set()
	added_edges = set()
	
	noise_level = 0
	batch_changes = []  # Track last 5 changes

	while noise_level < n_edits:
		operation = random.choice(["add", "remove"])
		noisy_graph_original = noisy_graph.copy()
		
		if operation == "add":
			u, v = random.sample(nodes, 2)
			# don't want to re-add an edge that was already removed -- this undoes the noise addition
			if (u,v) not in removed_edges\
					and u != v \
					and not noisy_graph.has_edge(u, v) \
					and is_approx_valid_move(u,v):
				noisy_graph.add_edge(u, v)
				added_edges.add((u, v))
				batch_changes.append(("add", u, v))
			else:
				print(f"invalid attempt to add edge. currently at noise level {noise_level}")
				continue
		
		elif operation == "remove":
			if noisy_graph.number_of_edges() > 0:
				edge = random.choice(list(noisy_graph.edges))
				# don't want to re-remove an edge that was already added -- this undoes the noise addition
				if edge not in added_edges:
					noisy_graph.remove_edge(*edge)
					removed_edges.add(edge)
					batch_changes.append(("remove", *edge))
				else:
					print(f"invalid attempt to remove edge. currently at noise level {noise_level}")
					continue
			else:
				print(f"invalid attempt to remove edge. currently at noise level {noise_level}")
				continue
		
		noise_level += 1

		# Check validity every 5 edits OR when at final noise level
		if noise_level % 5 == 0 or noise_level == n_edits:
			if not is_formally_valid_graph(noisy_graph, verbose=False):
				rollback_count = min(len(batch_changes), noise_level)  # Handle last few edits case
				print(f"Invalid batch detected. Rolling back last {rollback_count} operations. Noise level was {noise_level}.")
				
				# Rollback only the necessary changes
				for _ in range(rollback_count):
					op, u, v = batch_changes.pop()
					if op == "add":
						noisy_graph.remove_edge(u, v)
						added_edges.remove((u, v))
					elif op == "remove":
						noisy_graph.add_edge(u, v)
						removed_edges.remove((u, v))
				
				# Rollback noise level
				noise_level -= rollback_count
			else:
					batch_changes.clear()  # Clear batch if valid
								
		# if is_formally_valid_graph(noisy_graph, verbose=False):
		# 	noise_level += 1
		# 	print(f"added noise. currently at noise level {noise_level}")
		# else:
		# 	print(f"invalid attempt. currently at noise level {noise_level}")
		# 	noisy_graph = noisy_graph_original
	print("done")
	return noisy_graph

def generate_noisy_corpus(test_centroid, save_dir, noisy_corpus_dirname, corpus_size, noise):
	noisy_graphs = [add_noise_to_graph(test_centroid, noise) for _ in range(corpus_size)]

	# Ensure the save directory exists
	os.makedirs(save_dir, exist_ok=True)

	# Save each noisy graph as a separate pickle file
	for i, noisy_graph in enumerate(noisy_graphs):
		file_path = os.path.join(save_dir, f"{noisy_corpus_dirname}_{i+1}.pickle")
		with open(file_path, 'wb') as f:
				pickle.dump(noisy_graph, f)

	print(f"Saved {corpus_size} noisy graphs to {save_dir}")

def load_noisy_corpus(noisy_corpus_dir):
	noisy_graphs = []
	for filename in os.listdir(noisy_corpus_dir):
		if filename.endswith(".pickle"):
				file_path = os.path.join(noisy_corpus_dir, filename)
				with open(file_path, 'rb') as f:
					graph = pickle.load(f)
					noisy_graphs.append(graph)
	return noisy_graphs

def struct_dist(G1, G2, noisy_corpus_dirname="", gpu_id=0):
	device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
	print(f"Process {current_process().name} running on GPU {device} for centroid cluster {noisy_corpus_dirname}")

	STG_augmented_list = [G1, G2]
	listA_G, idx_node_mapping, node_metadata_dict = simanneal_centroid_helpers.pad_adj_matrices(STG_augmented_list)
	# A_G1, A_G2 = cp.asarray(listA_G[0]), cp.asarray(listA_G[1]) 
	# A_G1, A_G2 = np.asarray(listA_G[0]), np.asarray(listA_G[1]) 
	A_G1, A_G2 = torch.from_numpy(listA_G[0]).to(device), torch.from_numpy(listA_G[1]).to(device)
	_, struct_dist = simanneal_centroid_run.align_graph_pair(A_G1, A_G2, idx_node_mapping, node_metadata_dict, device=device)
	return struct_dist.item()

def generate_initial_alignments(noisy_corpus_dirname, STG_augmented_list, gpu_id=0):
	device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
	print(f"Process {current_process().name} running on GPU {device} for centroid cluster {noisy_corpus_dirname}")

	listA_G, idx_node_mapping, node_metadata_dict = simanneal_centroid_helpers.pad_adj_matrices(STG_augmented_list)
	listA_G_tensors = [torch.tensor(matrix, device=device, dtype=torch.float64) for matrix in listA_G]
	min_loss_A_G, min_loss_A_G_list_index, min_loss, optimal_alignments = simanneal_centroid_run.initial_centroid_and_alignments(listA_G_tensors, idx_node_mapping, node_metadata_dict, device=device)

	# because these are tensors originally
	initial_centroid = min_loss_A_G.cpu().numpy() 
	initial_alignments = [alignment.cpu().numpy() for alignment in optimal_alignments]
	alignments_dir = f"{DIRECTORY}/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}/initial_alignments"
	print("ALIGNMENTS DIR", alignments_dir)

	initial_alignment_files = []
	pieces_fp = [f for f in os.listdir(noisy_corpus_dirname) if os.path.isfile(os.path.join(noisy_corpus_dirname, f))]
	for i in range(len(STG_augmented_list)):
		piece_name = os.path.splitext(os.path.basename(pieces_fp[i]))[0]
		initial_alignment_files.append(os.path.join(alignments_dir, f'initial_alignment_{piece_name}.txt'))
	
	intial_centroid_piece_name = os.path.splitext(os.path.basename(pieces_fp[min_loss_A_G_list_index]))[0]
	initial_centroid_file = os.path.join(alignments_dir, f'initial_centroid_{intial_centroid_piece_name}.txt')
	print(initial_centroid_file)
	sys.exit(0)
	if not os.path.exists(alignments_dir):
		os.makedirs(alignments_dir)
	print(f"Created directory {alignments_dir}")
	
	np.savetxt(initial_centroid_file, initial_centroid, fmt='%d', delimiter=' ') # since min_loss_A_G is a tensor
	print(f'Saved: {initial_centroid_file}')
	
	for i, alignment in enumerate(initial_alignments):
		file_name = initial_alignment_files[i]
		np.savetxt(file_name, alignment, fmt='%d', delimiter=' ')
		print(f'Saved: {file_name}')

def get_saved_initial_alignments_and_centroid(noisy_corpus_dirname):
	alignments_dir = f"{DIRECTORY}/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}/initial_alignments"

	initial_alignment_files = []
	pieces_fp = [f for f in os.listdir(noisy_corpus_dirname) if os.path.isfile(os.path.join(noisy_corpus_dirname, f))]
	for file_path in pieces_fp:
		piece_name = os.path.splitext(os.path.basename(file_path))[0]
		initial_alignment_files.append(os.path.join(alignments_dir, f'initial_alignment_{piece_name}.txt'))
	initial_centroid_file = glob.glob(os.path.join(alignments_dir, '*initial_centroid*'))[0]
	
	alignments = [np.loadtxt(f) for f in initial_alignment_files]
	initial_centroid = np.loadtxt(initial_centroid_file)
	print(f'Loaded existing centroid and alignment files from {alignments_dir}')
	return initial_centroid, alignments, initial_centroid_file

def generate_approx_centroid(noisy_corpus_dirname, noisy_corpus_graphs, gpu_id=0):
	device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
	print(f"Process {current_process().name} running on GPU {device} for centroid cluster {noisy_corpus_dirname}")

	initial_centroid, initial_alignments, _ = get_saved_initial_alignments_and_centroid(noisy_corpus_dirname)
	listA_G, idx_node_mapping, node_metadata_dict = simanneal_centroid_helpers.pad_adj_matrices(noisy_corpus_graphs)

	# bc these are originally numpy and we can't convert to tensor till we get the device
	listA_G = [torch.tensor(A_G, device=device, dtype=torch.float64) for A_G in listA_G]
	initial_alignments = [torch.tensor(alignment, device=device, dtype=torch.float64) for alignment in initial_alignments]
	initial_centroid = torch.tensor(initial_centroid, device=device, dtype=torch.float64)
	aligned_listA_G = list(map(simanneal_centroid.align_torch, initial_alignments, listA_G))

	centroid_annealer = simanneal_centroid.CentroidAnnealer(initial_centroid, aligned_listA_G, idx_node_mapping, node_metadata_dict, device=device)
	# centroid_annealer.Tmax = 1.5
	# centroid_annealer.Tmin = 0.001
	# centroid_annealer.steps = 3000

	centroid_annealer.Tmax = 2.5
	centroid_annealer.Tmin = 0.05 
	centroid_annealer.steps = 1000


	approx_centroid, loss = centroid_annealer.anneal()
	approx_centroid = approx_centroid.cpu().numpy() # convert from tensor -> numpy
	loss = loss.item() # convert from tensor -> numpy
	
	approx_centroid, final_idx_node_mapping = simanneal_centroid_helpers.remove_unnecessary_dummy_nodes(approx_centroid, idx_node_mapping, node_metadata_dict)
	approx_centroid_dir = f"{DIRECTORY}/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}/approx_centroid"
	if not os.path.exists(approx_centroid_dir):
		os.makedirs(approx_centroid_dir)

	approx_centroid_path = os.path.join(approx_centroid_dir, "centroid.txt")
	np.savetxt(approx_centroid_path, approx_centroid, fmt='%d', delimiter=' ')
	print(f'Saved: {approx_centroid_path}')

	approx_centroid_idx_node_mapping_path = os.path.join(approx_centroid_dir, "idx_node_mapping.txt")
	with open(approx_centroid_idx_node_mapping_path, 'w') as file:
		json.dump(final_idx_node_mapping, file)
	print(f'Saved: {approx_centroid_idx_node_mapping_path}')

	approx_centroid_node_metadata_dict_path = os.path.join(approx_centroid_dir, "node_metadata_dict.txt")
	with open(approx_centroid_node_metadata_dict_path, 'w') as file:
		json.dump(node_metadata_dict, file)
	print(f'Saved: {approx_centroid_node_metadata_dict_path}')
	
	approx_centroid_loss_path = os.path.join(approx_centroid_dir, "loss.txt")
	np.savetxt(approx_centroid_loss_path, [loss], fmt='%d')
	print(f'Saved: {approx_centroid_loss_path}')

def repair_centroid(noisy_corpus_dirname):
	approx_centroid_dir = f"{DIRECTORY}/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}/approx_centroid"

	approx_centroid_path = os.path.join(approx_centroid_dir, "centroid.txt")
	approx_centroid = np.loadtxt(approx_centroid_path)
	print(f'Loaded: {approx_centroid_path}')

	approx_centroid_idx_node_mapping_path = os.path.join(approx_centroid_dir, "idx_node_mapping.txt")
	with open(approx_centroid_idx_node_mapping_path, 'r') as file:
		idx_node_mapping = json.load(file)
		idx_node_mapping = {int(k): v for k, v in idx_node_mapping.items()}
	print(f'Loaded: {approx_centroid_idx_node_mapping_path}')

	approx_centroid_node_metadata_dict_path = os.path.join(approx_centroid_dir, "node_metadata_dict.txt")
	with open(approx_centroid_node_metadata_dict_path, 'r') as file:
		node_metadata_dict = json.load(file)
	print(f'Loaded: {approx_centroid_node_metadata_dict_path}')

	z3_repair.initialize_globals(approx_centroid, idx_node_mapping, node_metadata_dict)

	final_centroid_dir = f"{DIRECTORY}/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}/final_centroid"
	if not os.path.exists(final_centroid_dir):
		os.makedirs(final_centroid_dir)
	final_centroid_filename = f'{final_centroid_dir}/final_centroid.txt'
	final_idx_node_mapping_filename = f'{final_centroid_dir}/final_idx_node_mapping.txt'

	z3_repair.run(final_centroid_filename, final_idx_node_mapping_filename)

def get_distances_from_centroid_to_corpus(noisy_corpus_graphs, centroid, gpu_id):
	device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
	listA_G, centroid_idx_node_mapping, node_metadata_dict = simanneal_centroid_helpers.pad_adj_matrices(noisy_corpus_graphs + [centroid])
	A_g, listA_G = torch.from_numpy(listA_G[-1]).to(device), [torch.from_numpy(A_G).to(device) for A_G in listA_G[:-1]]
	alignments = simanneal_centroid.get_alignments_to_centroid(A_g, listA_G, centroid_idx_node_mapping, node_metadata_dict, device=device)
	list_alignedA_G = list(map(simanneal_centroid.align_torch, alignments, listA_G)) # align to centroid A_g
	return np.array([simanneal_centroid.dist_torch(A_g, A_G).item() for A_G in list_alignedA_G])

if __name__ == "__main__":
	# plot_results()
	# sys.exit(0)
	
	test_centroid_path = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_461_(c)orlandi/biamonti_461_(c)orlandi_augmented_graph_flat.pickle'
	# test_centroid_path = DIRECTORY + '/datasets/bach/kunstderfuge/bwv876frag/bwv876frag_augmented_graph_flat.pickle'
	# test_centroid_path = DIRECTORY +'/datasets/beethoven/kunstderfuge/biamonti_811_(c)orlandi/biamonti_811_(c)orlandi_augmented_graph_flat.pickle'
	# test_centroid_path = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_317_(c)orlandi/biamonti_317_(c)orlandi_augmented_graph_flat.pickle'
	# test_centroid_path = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_360_(c)orlandi/biamonti_360_(c)orlandi_augmented_graph_flat.pickle'
	
	# make this more robust by doing different noise for each corpus?
	# come up with error bounds for the loss
	
	K = list(range(3,15))
	gpu_id = 1
	for k in K:
		print("K", k)
		# TEST extension is for HP1 params
		noisy_corpus_dirname = "noisy_corpus_" + os.path.basename(os.path.dirname(test_centroid_path)) + f"_no_std_size{k}"
		test_centroid = load_test_centroid(test_centroid_path)

		noisy_corpus_save_dir = DIRECTORY + f'/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}'
		noise = int(np.ceil(test_centroid.size()/2))
		# generate_noisy_corpus(test_centroid, noisy_corpus_save_dir, noisy_corpus_dirname, corpus_size=k, noise=noise)
		
		noisy_corpus_graphs = load_noisy_corpus(noisy_corpus_save_dir)
		print("NUM CORPUS GRAPHS", len(noisy_corpus_graphs))
		# for G in noisy_corpus_graphs:
		# 	print(struct_dist(test_centroid, G, noisy_corpus_dirname, gpu_id=gpu_id))
	
		# generate_initial_alignments(noisy_corpus_dirname, noisy_corpus_graphs, gpu_id=gpu_id)
		# generate_approx_centroid(noisy_corpus_dirname, noisy_corpus_graphs, gpu_id=gpu_id)
		# repair_centroid(noisy_corpus_dirname)
		# continue
		derived_centroid_A = np.loadtxt(f"{DIRECTORY}/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}/final_centroid/final_centroid.txt")
		approx_centroid_A = np.loadtxt(f"{DIRECTORY}/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}/approx_centroid/centroid.txt")
		
		node_metadata_dict_path = f"{DIRECTORY}/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}/approx_centroid/node_metadata_dict.txt"
		with open(node_metadata_dict_path, 'r') as file:
			node_metadata_dict = json.load(file)
		
		idx_node_mapping_path = f"{DIRECTORY}/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}/final_centroid/final_idx_node_mapping.txt"
		with open(idx_node_mapping_path, 'r') as file:
			idx_node_mapping = json.load(file)
			idx_node_mapping = {int(k): v for k, v in idx_node_mapping.items()}

		approx_idx_node_mapping_path = f"{DIRECTORY}/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}/approx_centroid/idx_node_mapping.txt"
		with open(approx_idx_node_mapping_path, 'r') as file:
			approx_idx_node_mapping = json.load(file)
			approx_idx_node_mapping = {int(k): v for k, v in approx_idx_node_mapping.items()}
		
		derived_centroid = simanneal_centroid_helpers.adj_matrix_to_graph(derived_centroid_A, idx_node_mapping, node_metadata_dict)
		approx_centroid = simanneal_centroid_helpers.adj_matrix_to_graph(approx_centroid_A, approx_idx_node_mapping, node_metadata_dict)

		# layers_test_centroid = build_graph.get_unsorted_layers_from_graph_by_index(test_centroid)
		# layers_derived_centroid = build_graph.get_unsorted_layers_from_graph_by_index(derived_centroid)
		# build_graph.visualize([test_centroid, derived_centroid], [layers_test_centroid, layers_derived_centroid])

		# print("SYNTHETIC TO DERIVED CENTROID DIST", struct_dist(test_centroid, derived_centroid, noisy_corpus_dirname, gpu_id=device))

		# lower_bound_value = solve_lower_bound(construct_distance_matrix(noisy_corpus_graphs))
		# print("LOWER BOUND:", lower_bound_value)

		distances_derived = get_distances_from_centroid_to_corpus(noisy_corpus_graphs, derived_centroid, gpu_id)
		derived_loss = np.mean(distances_derived) # unit is distance
		print("DERIVED LOSS", derived_loss)
		print("DISTS DERIVED", distances_derived)

		synthetic_loss = np.sqrt(noise)
		print("SYNTHETIC LOSS", synthetic_loss)
		print("RELATIVE ERROR:", np.abs(synthetic_loss - derived_loss) / synthetic_loss)
		print("ABSOLUTE ERROR:", np.abs(synthetic_loss - derived_loss))
		
		# for G in noisy_corpus_graphs:
		# 	print(struct_dist(G, test_centroid, noisy_corpus_dirname, gpu_id=device))

# K 3
# RELATIVE ERROR: 0.009755985185740532
# K 4
# RELATIVE ERROR: 0.004949172262302578
# K 5
# RELATIVE ERROR: 0.029933014260994057
# K 6
# RELATIVE ERROR: 0.020073253229684212
# K 7
# RELATIVE ERROR: 0.017261706192896447
# K 8
# RELATIVE ERROR: 0.002681773959127958
# K 9
# RELATIVE ERROR: 0.0144659053685462
# K 10
# RELATIVE ERROR: 0.02991807085508924
# K 11
# RELATIVE ERROR: 0.0
# K 12
# RELATIVE ERROR: 0.008451775723758695
# K 13
# RELATIVE ERROR: 0.007012663638145545
# K 14
# RELATIVE ERROR: 0.008629118369398
