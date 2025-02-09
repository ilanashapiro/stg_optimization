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

sys.path.append(f"{DIRECTORY}/centroid")
import z3_matrix_projection_incremental as z3_repair
import simanneal_centroid_helpers, simanneal_centroid_run
# import build_graph

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
	plt.bar(x - width/2, relative_errors_derived, width=width, color='salmon', edgecolor='black', label='Derived vs Ground Truth')
	plt.bar(x + width/2, relative_errors_naive, width=width, color='skyblue', edgecolor='black', label='Naive vs Ground Truth')

	plt.xlabel('Corpus Size $k$', fontsize=20)
	plt.ylabel('Relative Error in Loss', fontsize=20)
	plt.xticks(x, k_values)  # Ensure all k values are shown on the x-axis
	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.legend(fontsize=15)

	plt.show()

def load_STG(stg_path):
	with open(stg_path, 'rb') as f:
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
	batch_changes = []
	batch_size = 20 # how much noise to add before checking validity, so we possibly need to rollback this many changes if it isn't valid

	while noise_level < n_edits:
		operation = random.choice(["add", "remove"])
		
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

		if noise_level % batch_size == 0 or noise_level == n_edits:
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

	print("done")
	return noisy_graph

def load_noisy_STGs(noisy_STGs_dir):
	pickle_files = [f for f in os.listdir(noisy_STGs_dir) if f.endswith(".pickle")]

	def extract_number(filename):
			match = re.search(r"_(\d+)\.pickle$", filename)
			return int(match.group(1)) if match else float('inf')  # Default to inf if no number found

	pickle_files.sort(key=extract_number)
	noisy_graphs = []
	for filename in pickle_files:
			file_path = os.path.join(noisy_STGs_dir, filename)
			with open(file_path, 'rb') as f:
					noisy_graphs.append(pickle.load(f))
	
	return noisy_graphs

def structural_distance(G1, G2, noisy_corpus_dirname="", gpu_id=0):
	device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
	# print(f"Process {current_process().name} running on GPU {device} for STGs derived from {noisy_corpus_dirname}")

	STG_augmented_list = [G1, G2]
	listA_G, idx_node_mapping, node_metadata_dict = simanneal_centroid_helpers.pad_adj_matrices(STG_augmented_list)
	# A_G1, A_G2 = cp.asarray(listA_G[0]), cp.asarray(listA_G[1]) 
	# A_G1, A_G2 = np.asarray(listA_G[0]), np.asarray(listA_G[1]) 
	A_G1, A_G2 = torch.from_numpy(listA_G[0]).to(device), torch.from_numpy(listA_G[1]).to(device)
	_, structural_distance = simanneal_centroid_run.align_graph_pair(A_G1, A_G2, idx_node_mapping, node_metadata_dict, device=device)
	return structural_distance.item()

def generate_noisy_STGs(base_graph, percent_noise_max, percent_noise_increment, noisy_STGs_save_dir):
	noisy_graphs = []
	noise_percent_level = percent_noise_increment
	while noise_percent_level <= percent_noise_max:
		noise = int(np.ceil(base_graph.size()*noise_percent_level))
		noisy_graphs.append(add_noise_to_graph(base_graph, noise))
		print("Finished generating STGs with noise percent", noise_percent_level, "and noise", noise)
		noise_percent_level += percent_noise_increment

	os.makedirs(noisy_STGs_save_dir, exist_ok=True)
	for i, noisy_graph in enumerate(noisy_graphs):
		file_path = os.path.join(noisy_STGs_save_dir, f"{noisy_STGs_dirname}_{i+1}.pickle")
		with open(file_path, 'wb') as f:
				pickle.dump(noisy_graph, f)

	print(f"Saved graphs with noise up to {percent_noise_max*100}% percent in increments of {percent_noise_increment*100}% to {noisy_STGs_save_dir}")

if __name__ == "__main__":
	# plot_results()
	# sys.exit(0)
	
	# base_graph_path = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_461_(c)orlandi/biamonti_461_(c)orlandi_augmented_graph_flat.pickle'
	base_graph_path = DIRECTORY + '/datasets/bach/kunstderfuge/bwv876frag/bwv876frag_augmented_graph_flat.pickle'
	# base_graph_path = DIRECTORY +'/datasets/beethoven/kunstderfuge/biamonti_811_(c)orlandi/biamonti_811_(c)orlandi_augmented_graph_flat.pickle'
	# base_graph_path = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_317_(c)orlandi/biamonti_317_(c)orlandi_augmented_graph_flat.pickle'
	# base_graph_path = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_360_(c)orlandi/biamonti_360_(c)orlandi_augmented_graph_flat.pickle'
	
	base_graph = load_STG(base_graph_path)
	percent_noise_max = 2 # this is 50%
	percent_noise_increment = 0.05 # 5% increments in noise 
	gpu_id = 7
	
	noisy_STGs_dirname = "noisy_STGs_" + os.path.basename(os.path.dirname(base_graph_path)) + f"max_percent{percent_noise_max*100}_increment{percent_noise_increment*100}"
	noisy_STGs_save_dir = DIRECTORY + f'/experiments/structural_distance/structural_distance_synthetic/{noisy_STGs_dirname}'
	if not os.path.exists(noisy_STGs_save_dir):
		generate_noisy_STGs(base_graph, percent_noise_max, percent_noise_increment, noisy_STGs_save_dir)
	noisy_STGs = load_noisy_STGs(noisy_STGs_save_dir)
		
	for i, G in enumerate(noisy_STGs):
		noise_percent = percent_noise_increment * (i+1)
		noise = int(np.ceil(base_graph.size()*noise_percent))
		ground_truth = np.sqrt(noise)
		struct_dist = structural_distance(base_graph, G, noisy_STGs_dirname, gpu_id=gpu_id)
		print(f"AT NOISE PERCENT {noise_percent}, STRUCT DIST ERROR:",  np.abs(struct_dist-ground_truth)/ground_truth)
		print("NOSIE", noise, "STRUCT DIST", struct_dist, "GROUND TRUTH", ground_truth)
		print()

# AT NOISE PERCENT 0.05, STRUCT DIST ERROR: 0.0
# STRUCT DIST 2.6457513110645907 GROUND TRUTH 2.6457513110645907

# AT NOISE PERCENT 0.1, STRUCT DIST ERROR: 0.0
# STRUCT DIST 3.605551275463989 GROUND TRUTH 3.605551275463989

# AT NOISE PERCENT 0.15000000000000002, STRUCT DIST ERROR: 0.0
# STRUCT DIST 4.358898943540674 GROUND TRUTH 4.358898943540674

# AT NOISE PERCENT 0.2, STRUCT DIST ERROR: 0.0
# STRUCT DIST 5.0 GROUND TRUTH 5.0

# AT NOISE PERCENT 0.25, STRUCT DIST ERROR: 0.0
# STRUCT DIST 5.5677643628300215 GROUND TRUTH 5.5677643628300215

# AT NOISE PERCENT 0.30000000000000004, STRUCT DIST ERROR: 0.04318516770401161
# STRUCT DIST 6.082762530298219 GROUND TRUTH 5.830951894845301

# AT NOISE PERCENT 0.35000000000000003, STRUCT DIST ERROR: 0.05351527569995439
# STRUCT DIST 6.557438524302 GROUND TRUTH 6.928203230275509

# AT NOISE PERCENT 0.4, STRUCT DIST ERROR: 0.03209369308427994
# STRUCT DIST 7.0 GROUND TRUTH 6.782329983125268

# AT NOISE PERCENT 0.45, STRUCT DIST ERROR: 0.018693206542874638
# STRUCT DIST 7.416198487095663 GROUND TRUTH 7.280109889280518

# AT NOISE PERCENT 0.5, STRUCT DIST ERROR: 0.008298897483611496
# STRUCT DIST 7.810249675906654 GROUND TRUTH 7.745966692414834

# AT NOISE PERCENT 0.05, STRUCT DIST ERROR: 0.0
# STRUCT DIST 2.6457513110645907 GROUND TRUTH 2.6457513110645907

# AT NOISE PERCENT 0.1, STRUCT DIST ERROR: 0.0
# STRUCT DIST 3.605551275463989 GROUND TRUTH 3.605551275463989

# AT NOISE PERCENT 0.15000000000000002, STRUCT DIST ERROR: 0.02740233382816293
# STRUCT DIST 4.358898943540674 GROUND TRUTH 4.242640687119285

# AT NOISE PERCENT 0.2, STRUCT DIST ERROR: 0.0
# STRUCT DIST 5.0 GROUND TRUTH 5.0

# AT NOISE PERCENT 0.25, STRUCT DIST ERROR: 0.016530045465127
# STRUCT DIST 5.5677643628300215 GROUND TRUTH 5.477225575051661

# AT NOISE PERCENT 0.30000000000000004, STRUCT DIST ERROR: 0.0
# STRUCT DIST 6.082762530298219 GROUND TRUTH 6.082762530298219

# AT NOISE PERCENT 0.35000000000000003, STRUCT DIST ERROR: 0.0331584366114311
# STRUCT DIST 6.557438524302 GROUND TRUTH 6.782329983125268

# AT NOISE PERCENT 0.4, STRUCT DIST ERROR: 0.04742065558431966
# STRUCT DIST 7.0 GROUND TRUTH 7.3484692283495345

# AT NOISE PERCENT 0.45, STRUCT DIST ERROR: 0.018693206542874638
# STRUCT DIST 7.416198487095663 GROUND TRUTH 7.280109889280518

# AT NOISE PERCENT 0.5, STRUCT DIST ERROR: 0.023718790511668253
# STRUCT DIST 7.810249675906654 GROUND TRUTH 8.0

# AT NOISE PERCENT 0.55, STRUCT DIST ERROR: 0.0
# STRUCT DIST 8.246211251235321 GROUND TRUTH 8.246211251235321

# AT NOISE PERCENT 0.6000000000000001, STRUCT DIST ERROR: 0.050032092968270984
# STRUCT DIST 8.602325267042627 GROUND TRUTH 9.055385138137417

# AT NOISE PERCENT 0.65, STRUCT DIST ERROR: 0.046537410754407676
# STRUCT DIST 8.94427190999916 GROUND TRUTH 9.38083151964686

# AT NOISE PERCENT 0.7000000000000001, STRUCT DIST ERROR: 0.0434992854047225
# STRUCT DIST 9.273618495495704 GROUND TRUTH 9.695359714832659

# AT NOISE PERCENT 0.75, STRUCT DIST ERROR: 0.05282179638329998
# STRUCT DIST 9.591663046625438 GROUND TRUTH 9.1104335791443

# AT NOISE PERCENT 0.8, STRUCT DIST ERROR: 0.07282735005446933
# STRUCT DIST 9.899494936611665 GROUND TRUTH 10.677078252031311

# AT NOISE PERCENT 0.8500000000000001, STRUCT DIST ERROR: 0.08785965992068973
# STRUCT DIST 10.198039027185569 GROUND TRUTH 11.180339887498949

# AT NOISE PERCENT 0.9, STRUCT DIST ERROR: 0.030375876862100745
# STRUCT DIST 10.488088481701515 GROUND TRUTH 10.816653826391969

# AT NOISE PERCENT 0.9500000000000001, STRUCT DIST ERROR: 0.06256313343890794
# STRUCT DIST 10.770329614269007 GROUND TRUTH 11.489125293076057

# AT NOISE PERCENT 1.0, STRUCT DIST ERROR: 0.0
# STRUCT DIST 11.090536506409418 GROUND TRUTH 11.090536506409418

# AT NOISE PERCENT 1.05, STRUCT DIST ERROR: 0.01587400793602348
# STRUCT DIST 11.357816691600547 GROUND TRUTH 11.180339887498949

# AT NOISE PERCENT 1.1, STRUCT DIST ERROR: 0.08144134645630831
# STRUCT DIST 11.61895003862225 GROUND TRUTH 12.649110640673518

# AT NOISE PERCENT 1.1500000000000001, STRUCT DIST ERROR: 0.010471492746840264
# STRUCT DIST 11.874342087037917 GROUND TRUTH 12.0

# AT NOISE PERCENT 1.2000000000000002, STRUCT DIST ERROR: 0.06178919162158244
# STRUCT DIST 12.12435565298214 GROUND TRUTH 12.922847983320086

# AT NOISE PERCENT 1.25, STRUCT DIST ERROR: 0.07026521004051005
# STRUCT DIST 12.36931687685298 GROUND TRUTH 13.30413469565007

# AT NOISE PERCENT 1.3, STRUCT DIST ERROR: 0.10613481168893717
# STRUCT DIST 12.609520212918492 GROUND TRUTH 14.106735979665885

# AT NOISE PERCENT 1.35, STRUCT DIST ERROR: 0.034493195347382044
# STRUCT DIST 12.84523257866513 GROUND TRUTH 13.30413469565007

# AT NOISE PERCENT 1.4000000000000001, STRUCT DIST ERROR: 0.13586755018128482
# STRUCT DIST 13.076696830622021 GROUND TRUTH 15.132745950421556

# AT NOISE PERCENT 1.4500000000000002, STRUCT DIST ERROR: 0.09685648827831521
# STRUCT DIST 13.30413469565007 GROUND TRUTH 14.730919862656235

# AT NOISE PERCENT 1.5, STRUCT DIST ERROR: 0.13986198876344053
# STRUCT DIST 13.490737563232042 GROUND TRUTH 15.684387141358123

# AT NOISE PERCENT 1.55, STRUCT DIST ERROR: 0.14678880184387877
# STRUCT DIST 13.784048752090222 GROUND TRUTH 16.15549442140351

# AT NOISE PERCENT 1.6, STRUCT DIST ERROR: 0.16184192387505053
# STRUCT DIST 14.0 GROUND TRUTH 16.703293088490067

# AT NOISE PERCENT 1.6500000000000001, STRUCT DIST ERROR: 0.17529721709292134
# STRUCT DIST 14.212670403551895 GROUND TRUTH 17.233687939614086

# AT NOISE PERCENT 1.7000000000000002, STRUCT DIST ERROR: 0.1617263557150907
# STRUCT DIST 14.422205101855956 GROUND TRUTH 17.204650534085253

# AT NOISE PERCENT 1.75, STRUCT DIST ERROR: 0.18978909055023765
# STRUCT DIST 14.628738838327793 GROUND TRUTH 18.05547008526779

# AT NOISE PERCENT 1.8, STRUCT DIST ERROR: 0.16295941396180835
# STRUCT DIST 14.832396974191326 GROUND TRUTH 17.72004514666935

# AT NOISE PERCENT 1.85, STRUCT DIST ERROR: 0.1810841646416932
# STRUCT DIST 15.033296378372908 GROUND TRUTH 18.35755975068582

# AT NOISE PERCENT 1.9000000000000001, STRUCT DIST ERROR: 0.23746892910176234
# STRUCT DIST 15.231546211727817 GROUND TRUTH 19.974984355438178

# AT NOISE PERCENT 1.9500000000000002, STRUCT DIST ERROR: 0.1947041770890525
# STRUCT DIST 15.427248620541512 GROUND TRUTH 19.157244060668017



# done
# Finished generating STGs with noise percent 0.05 and noise 7
# invalid attempt to add edge. currently at noise level 2
# done
# Finished generating STGs with noise percent 0.1 and noise 13
# invalid attempt to add edge. currently at noise level 17
# done
# Finished generating STGs with noise percent 0.15000000000000002 and noise 19
# invalid attempt to add edge. currently at noise level 16
# done
# Finished generating STGs with noise percent 0.2 and noise 25
# invalid attempt to remove edge. currently at noise level 25
# done
# Finished generating STGs with noise percent 0.25 and noise 31
# invalid attempt to remove edge. currently at noise level 22
# done
# Finished generating STGs with noise percent 0.3 and noise 37
# invalid attempt to add edge. currently at noise level 21
# invalid attempt to remove edge. currently at noise level 21
# invalid attempt to remove edge. currently at noise level 26
# done
# Finished generating STGs with noise percent 0.35 and noise 43
# invalid attempt to add edge. currently at noise level 4
# invalid attempt to add edge. currently at noise level 39
# done
# Finished generating STGs with noise percent 0.39999999999999997 and noise 49
# invalid attempt to remove edge. currently at noise level 21
# invalid attempt to add edge. currently at noise level 29
# invalid attempt to remove edge. currently at noise level 29
# invalid attempt to remove edge. currently at noise level 30
# invalid attempt to remove edge. currently at noise level 51
# done
# Finished generating STGs with noise percent 0.44999999999999996 and noise 55
# invalid attempt to remove edge. currently at noise level 15
# invalid attempt to remove edge. currently at noise level 38
# invalid attempt to remove edge. currently at noise level 49
# invalid attempt to remove edge. currently at noise level 57
# done