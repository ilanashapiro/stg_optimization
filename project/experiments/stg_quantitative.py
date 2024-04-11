import glob
import os
import sys

import simanneal

dir_prefix = "/home/ilshapiro/project"
# dir_prefix = "/home/jonsuss/Ilana_Shapiro/constraints"
# dir_prefix = "/Users/ilanashapiro/Documents/constraints_project/project"

sys.path.append(dir_prefix)
sys.path.append(f"{dir_prefix}/centroid")

import build_graph
import simanneal_centroid_run, simanneal_centroid_helpers

# available_dirs = ['classical_piano_midi_db/chopin/chpn-p7', # 167 -- NUM NOTES
# 				'classical_piano_midi_db/chopin/chpn-p20', # 286
# 				'classical_piano_midi_db/chopin/chpn-p4', 
# 				'classical_piano_midi_db/chopin/chpn-p6',
# 				'classical_piano_midi_db/schumann/scn15_4', # 220
# 				'classical_piano_midi_db/schumann/scn15_1',
# 				'classical_piano_midi_db/schumann/scn15_7',
# 				'classical_piano_midi_db/schumann/scn15_13', # 236
# 				'classical_piano_midi_db/clementi/clementi_opus36_2_2', # 253
# 				'classical_piano_midi_db/clementi/clementi_opus36_3_2', # 274
# 				'classical_piano_midi_db/clementi/clementi_opus36_1_2',
# 				'classical_piano_midi_db/clementi/clementi_opus36_1_1',
# 				'classical_piano_midi_db/clementi/clementi_opus36_2_1',
# 				'classical_piano_midi_db/clementi/clementi_opus36_4_2',
# 				'classical_piano_midi_db/clementi/clementi_opus36_1_3',
# 				'classical_piano_midi_db/haydn/haydn_8_2', # 290
# 				'classical_piano_midi_db/haydn/haydn_8_3', # 344
# 				'classical_piano_midi_db/haydn/haydn_7_2'
# 				]

dirs = [f"{dir_prefix}/classical_piano_midi_db/chopin/chpn-p7",
				f"{dir_prefix}/classical_piano_midi_db/chopin/chpn-p20",
				f"{dir_prefix}/classical_piano_midi_db/schumann/scn15_4", 
				f"{dir_prefix}/classical_piano_midi_db/schumann/scn15_13",
				f"{dir_prefix}/classical_piano_midi_db/clementi/clementi_opus36_1_3",
				f"{dir_prefix}/classical_piano_midi_db/clementi/clementi_opus36_3_2",
				f"{dir_prefix}/classical_piano_midi_db/haydn/haydn_8_3",
				f"{dir_prefix}/classical_piano_midi_db/haydn/haydn_7_2"]

STG_list = []
STG_augmented_list = []
STG_padded_adj_matrix_list = []
for dir in dirs:
	segment_filepath = glob.glob(os.path.join(dir, "*_sf_fmc2d_segments.txt"))[0]
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

_, chopin_schumann_cost = simanneal_centroid_run.align_graph_pair(chopin1_A_G, schumann1_A_G, idx_node_mapping)
# _, clementi_haydn_cost = simanneal_centroid_run.align_graph_pair(clementi1_A_G, haydn1_A_G, idx_node_mapping)

_, chopin_haydn_cost = simanneal_centroid_run.align_graph_pair(chopin1_A_G, haydn1_A_G, idx_node_mapping)
_, chopin_clementi_cost = simanneal_centroid_run.align_graph_pair(chopin1_A_G, clementi1_A_G, idx_node_mapping)
_, schumann_haydn_cost = simanneal_centroid_run.align_graph_pair(schumann1_A_G, haydn1_A_G, idx_node_mapping)
_, schumann_clementi_cost = simanneal_centroid_run.align_graph_pair(schumann1_A_G, clementi1_A_G, idx_node_mapping)

variable_names = [
		# "chopin_chopin_cost",
		# "schumann_schumann_cost",
		# "clementi_clementi_cost",
		# "haydn_haydn_cost",
		"chopin_schumann_cost",
		# "clementi_haydn_cost",
		"chopin_haydn_cost",
		"chopin_clementi_cost",
		"schumann_haydn_cost",
		"schumann_clementi_cost"
]

for var_name in variable_names:
	print(f"{var_name}: {locals()[var_name]}")

# with Tmax = 1.25, Tmin = 0.01, steps = 500
# results: scluster + scluster
# chopin_schumann_cost: 22.67156809750927
# chopin_haydn_cost: 28.26658805020514
# chopin_clementi_cost: 25.436194683953808
# schumann_haydn_cost: 29.34280150224242
# schumann_clementi_cost: 25.632011235952593
# sf + fmc2d
# chopin_schumann_cost: 7.0710678118654755
# chopin_haydn_cost: 9.695359714832659
# chopin_clementi_cost: 16.492422502470642
# schumann_haydn_cost: 9.899494936611665
# schumann_clementi_cost: 16.55294535724685