import glob
import os
import sys
from multiprocessing import Pool
import simanneal

dir_prefix = "/home/jonsuss/Ilana_Shapiro/constraints"
# dir_prefix = "/Users/ilanashapiro/Documents/constraints_project/project"

sys.path.append(dir_prefix)
sys.path.append(f"{dir_prefix}/centroid")

import build_graph
import simanneal_centroid_run, simanneal_centroid_helpers

dirs = [f"{dir_prefix}/classical_piano_midi_db/chopin/chpn-p7", # 167 -- NUM NOTES
				f"{dir_prefix}/classical_piano_midi_db/chopin/chpn-p20", # 286
				f"{dir_prefix}/classical_piano_midi_db/schumann/scn15_4", # 220
				f"{dir_prefix}/classical_piano_midi_db/schumann/scn15_13", # 236
				f"{dir_prefix}/classical_piano_midi_db/clementi/clementi_opus36_2_2", # 253
				f"{dir_prefix}/classical_piano_midi_db/clementi/clementi_opus36_3_2", # 274
				f"{dir_prefix}/classical_piano_midi_db/haydn/haydn_8_2", # 290
				f"{dir_prefix}/classical_piano_midi_db/haydn/haydn_8_3"] # 344

STG_list = []
STG_augmented_list = []
STG_padded_adj_matrix_list = []
for dir in dirs:
	for f in os.listdir(dir):
		segment_filepath = glob.glob(os.path.join(dir, "*_scluster_scluster_segments.txt"))[0]
		motive_filepath = glob.glob(os.path.join(dir, "*_motives1.txt"))[0]
		G, _ = build_graph.generate_graph(segment_filepath, motive_filepath)
		STG_list.append(G)
		build_graph.augment_graph(G)
		STG_augmented_list.append(G)

padded_A_G_list, idx_node_mapping = simanneal_centroid_helpers.pad_adj_matrices(STG_augmented_list)
chopin1_A_G = padded_A_G_list[0]
chopin2_A_G = padded_A_G_list[1]
schumann1_A_G = padded_A_G_list[2]
schumann2_A_G = padded_A_G_list[3]
clementi1_A_G = padded_A_G_list[4]
clementi2_A_G = padded_A_G_list[5]
haydn1_A_G = padded_A_G_list[6]
haydn2_A_G = padded_A_G_list[7]

# _, chopin_chopin_cost = simanneal_centroid_run.align_graph_pair(chopin1_A_G, chopin2_A_G, idx_node_mapping)
# _, schumann_schumann_cost = simanneal_centroid_run.align_graph_pair(schumann1_A_G, schumann2_A_G, idx_node_mapping)
# _, clementi_clementi_cost = simanneal_centroid_run.align_graph_pair(clementi1_A_G, clementi2_A_G, idx_node_mapping)
# _, haydn_haydn_cost = simanneal_centroid_run.align_graph_pair(haydn1_A_G, haydn2_A_G, idx_node_mapping)

# _, chopin_schumann_cost = simanneal_centroid_run.align_graph_pair(chopin1_A_G, schumann1_A_G, idx_node_mapping)
# _, clementi_haydn_cost = simanneal_centroid_run.align_graph_pair(clementi1_A_G, haydn1_A_G, idx_node_mapping)

# _, chopin_haydn_cost = simanneal_centroid_run.align_graph_pair(chopin1_A_G, haydn1_A_G, idx_node_mapping)
# _, chopin_clementi_cost = simanneal_centroid_run.align_graph_pair(chopin1_A_G, clementi1_A_G, idx_node_mapping)
# _, schumann_haydn_cost = simanneal_centroid_run.align_graph_pair(schumann1_A_G, haydn1_A_G, idx_node_mapping)
# _, schumann_clementi_cost = simanneal_centroid_run.align_graph_pair(schumann1_A_G, clementi1_A_G, idx_node_mapping)

# for var_name in variable_names:
# 	print(f"{var_name}: {locals()[var_name]}")

def align_graph_pair_wrapper(args):
    print("PROCESSING", args[0])
    return simanneal_centroid_run.align_graph_pair(*args[1])

variable_names = [
			"chopin_chopin_cost",
			"schumann_schumann_cost",
			"clementi_clementi_cost",
			"haydn_haydn_cost",
			"chopin_schumann_cost",
			"clementi_haydn_cost",
			"chopin_haydn_cost",
			"chopin_clementi_cost",
			"schumann_haydn_cost",
			"schumann_clementi_cost"
	]

# Define tasks as tuples to match the expected arguments for `align_graph_pair`
tasks = [
		("chopin1_A_G, chopin2_A_G", (chopin1_A_G, chopin2_A_G, idx_node_mapping)),
		("schumann1_A_G, schumann2_A_G", (schumann1_A_G, schumann2_A_G, idx_node_mapping)),
		("clementi1_A_G, clementi2_A_G", (clementi1_A_G, clementi2_A_G, idx_node_mapping)),
		("haydn1_A_G, haydn2_A_G", (haydn1_A_G, haydn2_A_G, idx_node_mapping)),
		("chopin1_A_G, schumann1_A_G", (chopin1_A_G, schumann1_A_G, idx_node_mapping)),
		("clementi1_A_G, haydn1_A_G", (clementi1_A_G, haydn1_A_G, idx_node_mapping)),
		("chopin1_A_G, haydn1_A_G", (chopin1_A_G, haydn1_A_G, idx_node_mapping)),
		("chopin1_A_G, clementi1_A_G", (chopin1_A_G, clementi1_A_G, idx_node_mapping)),
		("schumann1_A_G, haydn1_A_G", (schumann1_A_G, haydn1_A_G, idx_node_mapping)),
		("schumann1_A_G, clementi1_A_G", (schumann1_A_G, clementi1_A_G, idx_node_mapping))
]

# A dictionary to hold the results
costs = {}

if __name__ == '__main__':
	with Pool() as pool:
		results = pool.map(align_graph_pair_wrapper, tasks)

	results_dict = dict(zip(variable_names, results))

	for name, cost in results_dict.items():
			print(f"{name}: {cost}")