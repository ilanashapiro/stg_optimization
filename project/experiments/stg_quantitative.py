import glob
import os
import sys

import simanneal

sys.path.append("/Users/ilanashapiro/Documents/constraints_project/project")
sys.path.append("/Users/ilanashapiro/Documents/constraints_project/project/centroid")
import build_graph
import simanneal_centroid_run, simanneal_centroid_helpers

dirs = ['/Users/ilanashapiro/Documents/constraints_project/project/classical_piano_midi_db/chopin/chpn-p7', # 167 -- NUM NOTES
				'/Users/ilanashapiro/Documents/constraints_project/project/classical_piano_midi_db/chopin/chpn-p20', # 286
				'/Users/ilanashapiro/Documents/constraints_project/project/classical_piano_midi_db/schumann/scn15_4', # 220
				'/Users/ilanashapiro/Documents/constraints_project/project/classical_piano_midi_db/schumann/scn15_13', # 236
				'/Users/ilanashapiro/Documents/constraints_project/project/classical_piano_midi_db/clementi/clementi_opus36_2_2', # 253
				'/Users/ilanashapiro/Documents/constraints_project/project/classical_piano_midi_db/clementi/clementi_opus36_3_2', # 274
				'/Users/ilanashapiro/Documents/constraints_project/project/classical_piano_midi_db/haydn/haydn_8_2', # 290
				'/Users/ilanashapiro/Documents/constraints_project/project/classical_piano_midi_db/haydn/haydn_8_3'] # 344

STG_list = []
STG_augmented_list = []
STG_padded_adj_matrix_list = []
for dir in dirs:
	for f in os.listdir(dir):
		f_path = os.path.join(dir, f)
		segment_filepath = glob.glob(os.path.join(dir, "*_scluster_scluster_segments.txt"))[0]
		motive_filepath = glob.glob(os.path.join(dir, "*_motives4.txt"))[0]
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

_, chopin_chopin_cost = simanneal_centroid_run.align_graph_pair(chopin1_A_G, chopin2_A_G, idx_node_mapping)
_, schumann_schumann_cost = simanneal_centroid_run.align_graph_pair(schumann1_A_G, schumann2_A_G, idx_node_mapping)
_, clementi_clementi_cost = simanneal_centroid_run.align_graph_pair(clementi1_A_G, clementi2_A_G, idx_node_mapping)
_, haydn_haydn_cost = simanneal_centroid_run.align_graph_pair(haydn1_A_G, haydn2_A_G, idx_node_mapping)

_, chopin_schumann_cost = simanneal_centroid_run.align_graph_pair(chopin1_A_G, schumann1_A_G, idx_node_mapping)
_, clementi_haydn_cost = simanneal_centroid_run.align_graph_pair(clementi1_A_G, haydn1_A_G, idx_node_mapping)
_, chopin_haydn_cost = simanneal_centroid_run.align_graph_pair(chopin1_A_G, chopin2_A_G, idx_node_mapping)
_, chopin_clementi_cost = simanneal_centroid_run.align_graph_pair(chopin1_A_G, chopin2_A_G, idx_node_mapping)
print(align_graph_pair())