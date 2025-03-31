import os, sys, pickle, json, re, random
import numpy as np
import networkx as nx
import torch, glob
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum
from multiprocessing import Pool, current_process

# DIRECTORY = "/home/ubuntu/project"
DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
# DIRECTORY = "/home/ilshapiro/project"

sys.path.append(f"{DIRECTORY}/centroid")
import z3_matrix_projection_incremental as z3_repair
import simanneal_centroid_helpers, simanneal_centroid_run, simanneal_centroid
# import build_graph

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
	k_values = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
	relative_errors_derived = np.array([
			0.009755985185740532, 0.004949172262302578, 0.029933014260994057, 
			0.020073253229684212, 0.017261706192896447, 0.002681773959127958, 
			0.0144659053685462, 0.0, 0.0, 0.008451775723758695, 
			0.007012663638145545, 0.008629118369398
	])
	relative_errors_naive = np.array([
			0.11169285642178257, 0.01816039639319403, 0.08063778055995689, 
			0.12669268411796772, 0.13979314423458802, 0.1454582326611979, 
			0.10859480915733677, 0.1734060626852591, 0.20135027442563594, 
			0.20773524150868308, 0.20947096849054983, 0.2077036544127451
	])

	proportions = np.where(relative_errors_derived != 0, relative_errors_naive / relative_errors_derived, np.inf)

	finite_proportions = proportions[np.isfinite(proportions)]
	average_proportion = np.mean(finite_proportions)
	min_proportion = np.min(finite_proportions)
	max_proportion = np.max(finite_proportions)

	# Find indices of min and max proportions
	min_index = np.argmin(finite_proportions)
	max_index = np.argmax(finite_proportions)

	# Get the original indices in the array
	finite_indices = np.where(np.isfinite(proportions))[0]
	min_index_original = finite_indices[min_index]
	max_index_original = finite_indices[max_index]

	k_start = 3
	print(f"Average proportion: {average_proportion}")
	print(f"Minimum proportion: {min_proportion} (K: {k_start + min_index_original})")
	print(f"Maximum proportion: {max_proportion} (K: {k_start + max_index_original})")

	width = 0.3  # Width of the bars
	x = np.arange(len(k_values))  # X locations for the groups

	plt.figure(figsize=(10, 6))
	
	# Plot the two sets of bars side by side
	plt.bar(x - width/2, relative_errors_derived, width=width, color='#eb3223', edgecolor='black', label='Derived vs Ground Truth')
	plt.bar(x + width/2, relative_errors_naive, width=width, color='#4cafeb', edgecolor='black', label='Naive vs Ground Truth')

	plt.xlabel('Corpus Size $k$', fontsize=20)
	plt.ylabel('Relative Error in Loss', fontsize=20)
	plt.xticks(x, k_values)  # Ensure all k values are shown on the x-axis
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.legend(fontsize=15)

	plt.show()

def load_STG(stg_path):
	with open(stg_path, 'rb') as f:
		graph = pickle.load(f)
	return graph

# modified from simanneal_centroid.py
def is_approx_valid_move(G, source_node_id, sink_node_id):
	def is_proto(node_id):
		return node_id.startswith('Pr')
	def is_instance(node_id):
		return not is_proto(node_id)
	
	# The edge is between two prototypes
	if is_proto(source_node_id) and is_proto(sink_node_id):
		return False
	
	# The edge is between a prototype and instance level whose nodes don't have that prototype feature (i.e. PrAbs_interval -> segmentation)
	if is_proto(source_node_id) and is_instance(sink_node_id) or is_proto(sink_node_id) and is_instance(source_node_id):
		proto_feature_name = nx.get_node_attributes(G, "feature_name")[source_node_id if is_proto(source_node_id) else sink_node_id]
		inst_features = nx.get_node_attributes(G, "features_dict")[source_node_id if is_instance(source_node_id) else sink_node_id].keys()
		if proto_feature_name not in inst_features:
			return False
	
	# Source/sink are both instance
	if is_instance(source_node_id) and is_instance(sink_node_id):
		def rank_difference(rank1, rank2):
			primary_rank1, secondary_rank1 = rank1
			primary_rank2, secondary_rank2 = rank2
			if primary_rank1 == primary_rank2:
				return secondary_rank1 - secondary_rank2
			return primary_rank1 - primary_rank2
		
		# levels are NOT adjacent (i.e. 1 rank lower or higher, since this is the undirected version), or levels are NOT the same level (for intra level chain edge)
		layer_ranks = nx.get_node_attributes(G, "layer_rank") 
		source_rank = layer_ranks[source_node_id]
		sink_rank = layer_ranks[source_node_id]
		if np.abs(rank_difference(source_rank, sink_rank)) not in [0, 1]:
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
		
		if operation == "add":
			u, v = random.sample(nodes, 2)
			# don't want to re-add an edge that was already removed -- this undoes the noise addition
			if (u,v) not in removed_edges\
					and u != v \
					and not noisy_graph.has_edge(u, v) \
					and is_approx_valid_move(noisy_graph, u,v):
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
		if noise_level % 20 == 0 or noise_level == n_edits:
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

def generate_noisy_corpus(base_graph, save_dir, noisy_corpus_dirname, corpus_size, noise):
	noisy_graphs = [add_noise_to_graph(base_graph, noise) for _ in range(corpus_size)]

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
	initial_centroid_filepath = glob.glob(os.path.join(alignments_dir, '*initial_centroid*'))[0]
	
	alignments = [np.loadtxt(f) for f in initial_alignment_files]
	initial_centroid = np.loadtxt(initial_centroid_filepath)
	print(f'Loaded existing centroid and alignment files from {alignments_dir}')
	return initial_centroid, alignments, initial_centroid_filepath

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
	# centroid_annealer.Tmax = 1.25
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
	plot_results()
	# sys.exit(0)
	
	base_graph_path = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_461_(c)orlandi/biamonti_461_(c)orlandi_augmented_graph_flat.pickle'
	# base_graph_path = DIRECTORY + '/datasets/bach/kunstderfuge/bwv876frag/bwv876frag_augmented_graph_flat.pickle'
	# base_graph_path = DIRECTORY +'/datasets/beethoven/kunstderfuge/biamonti_811_(c)orlandi/biamonti_811_(c)orlandi_augmented_graph_flat.pickle'
	# base_graph_path = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_317_(c)orlandi/biamonti_317_(c)orlandi_augmented_graph_flat.pickle'
	# base_graph_path = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_360_(c)orlandi/biamonti_360_(c)orlandi_augmented_graph_flat.pickle'
	
	K = list(range(3,15))
	gpu_id = 3
	for k in K:
		print("K", k)
		noisy_corpus_dirname = "noisy_corpus_" + os.path.basename(os.path.dirname(base_graph_path)) + f"_size{k}_TESTT"
		base_graph = load_STG(base_graph_path)

		noisy_corpus_save_dir = DIRECTORY + f'/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}'
		noise = int(np.ceil(base_graph.size()/2))

		noisy_corpus_graphs = load_noisy_corpus(noisy_corpus_save_dir)
		print("NUM CORPUS GRAPHS", len(noisy_corpus_graphs))
		# for G in noisy_corpus_graphs:
		# 	print(struct_dist(base_graph, G, noisy_corpus_dirname, gpu_id=gpu_id))
	
		# generate_initial_alignments(noisy_corpus_dirname, noisy_corpus_graphs, gpu_id=gpu_id)
		# generate_approx_centroid(noisy_corpus_dirname, noisy_corpus_graphs, gpu_id=gpu_id)
		# repair_centroid(noisy_corpus_dirname)
		# continue
		
		derived_centroid_A = np.loadtxt(f"{DIRECTORY}/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}/final_centroid/final_centroid.txt")
		approx_centroid_A = np.loadtxt(f"{DIRECTORY}/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}/approx_centroid/centroid.txt")
		
		_, _, initial_centroid_filepath = get_saved_initial_alignments_and_centroid(noisy_corpus_dirname)
		initial_centroid_filename = os.path.basename(initial_centroid_filepath)[:-4].replace('initial_centroid_', '')
		naive_centroid_filepath = f"{DIRECTORY}/experiments/centroid/synthetic_centroid_experiment/{noisy_corpus_dirname}/{initial_centroid_filename}.pickle"
		naive_centroid = load_STG(naive_centroid_filepath)

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

		# layers_base_graph = build_graph.get_unsorted_layers_from_graph_by_index(base_graph)
		# layers_derived_centroid = build_graph.get_unsorted_layers_from_graph_by_index(derived_centroid)
		# build_graph.visualize([base_graph, derived_centroid], [layers_base_graph, layers_derived_centroid])

		# print("SYNTHETIC TO DERIVED CENTROID DIST", struct_dist(base_graph, derived_centroid, noisy_corpus_dirname, gpu_id=device))

		# lower_bound_value = solve_lower_bound(construct_distance_matrix(noisy_corpus_graphs))
		# print("LOWER BOUND:", lower_bound_value)

		distances_naive = get_distances_from_centroid_to_corpus(noisy_corpus_graphs, naive_centroid, gpu_id)
		naive_loss = np.mean(distances_naive) # unit is distance
		# print("NAIVE LOSS", naive_loss)

		distances_derived = get_distances_from_centroid_to_corpus(noisy_corpus_graphs, derived_centroid, gpu_id)
		derived_loss = np.mean(distances_derived) # unit is distance
		# print("DERIVED LOSS", derived_loss)
		# print("DISTS DERIVED", distances_derived)

		synthetic_loss = np.sqrt(noise)
		# print("SYNTHETIC LOSS", synthetic_loss)
		print("RELATIVE ERROR DERIVED VS SYNTHETIC:", np.abs(synthetic_loss - derived_loss) / synthetic_loss)
		# print("ABSOLUTE ERROR DERIVED VS SYNTHETIC:", np.abs(synthetic_loss - derived_loss))

		print("RELATIVE ERROR NAIVE VS SYNTHETIC:", np.abs(synthetic_loss - naive_loss) / synthetic_loss)
		# print("ABSOLUTE ERROR NAIVE VS SYNTHETIC:", np.abs(synthetic_loss - derived_loss))
		
		# for G in noisy_corpus_graphs:
		# 	print(struct_dist(G, base_graph, noisy_corpus_dirname, gpu_id=device))

# K 3
# RELATIVE ERROR DERIVED VS SYNTHETIC: 0.009755985185740532
# RELATIVE ERROR NAIVE VS SYNTHETIC: 0.11169285642178257
# K 4
# RELATIVE ERROR DERIVED VS SYNTHETIC: 0.004949172262302578
# RELATIVE ERROR NAIVE VS SYNTHETIC: 0.01816039639319403
# K 5
# RELATIVE ERROR DERIVED VS SYNTHETIC: 0.029933014260994057
# RELATIVE ERROR NAIVE VS SYNTHETIC: 0.08063778055995689
# K 6
# RELATIVE ERROR DERIVED VS SYNTHETIC: 0.020073253229684212
# RELATIVE ERROR NAIVE VS SYNTHETIC: 0.12669268411796772
# K 7
# RELATIVE ERROR DERIVED VS SYNTHETIC: 0.017261706192896447
# RELATIVE ERROR NAIVE VS SYNTHETIC: 0.13979314423458802
# K 8
# RELATIVE ERROR DERIVED VS SYNTHETIC: 0.002681773959127958
# RELATIVE ERROR NAIVE VS SYNTHETIC: 0.1454582326611979
# K 9
# RELATIVE ERROR DERIVED VS SYNTHETIC: 0.0144659053685462
# RELATIVE ERROR NAIVE VS SYNTHETIC: 0.10859480915733677
# K 10
# RELATIVE ERROR DERIVED VS SYNTHETIC: 0.0
# RELATIVE ERROR NAIVE VS SYNTHETIC: 0.1734060626852591
# K 11
# RELATIVE ERROR DERIVED VS SYNTHETIC: 0.0
# RELATIVE ERROR NAIVE VS SYNTHETIC: 0.20135027442563594
# K 12
# RELATIVE ERROR DERIVED VS SYNTHETIC: 0.008451775723758695
# RELATIVE ERROR NAIVE VS SYNTHETIC: 0.20773524150868308
# K 13
# RELATIVE ERROR DERIVED VS SYNTHETIC: 0.007012663638145545
# RELATIVE ERROR NAIVE VS SYNTHETIC: 0.20947096849054983
# K 14
# RELATIVE ERROR DERIVED VS SYNTHETIC: 0.008629118369398
# RELATIVE ERROR NAIVE VS SYNTHETIC: 0.2077036544127451