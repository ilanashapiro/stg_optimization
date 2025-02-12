import pickle
import os, sys
import networkx as nx
import ast, math
import pandas as pd

# DIRECTORY = '/home/ilshapiro/project'
DIRECTORY = '/home/ubuntu/project'
# DIRECTORY = '/Users/ilanashapiro/Documents/constraints_project/project'
sys.path.append(DIRECTORY)
import build_graph

def get_approx_end_time(csv_path):
	df = pd.read_csv(csv_path)
	if 'onset_seconds' in df.columns:
		return df['onset_seconds'].max()
	else:
		raise ValueError(f"'onset_seconds' column not found in {csv_path}")
		
def find_n_smallest_pickles(n=2):
	files_with_sizes = []
	for root, _, files in os.walk(DIRECTORY + "/datasets"):
			for file in files:
				if file.endswith('_augmented_graph_flat.pickle'):
					file_path = os.path.join(root, file)
					file_size = os.path.getsize(file_path)
					csv_path = file_path.replace("_augmented_graph_flat.pickle", ".csv")
					duration = get_approx_end_time(csv_path)
					files_with_sizes.append((file_path, file_size, duration))
	
	# Sort the list by file size (ascending)
	files_with_sizes.sort(key=lambda x: x[1])
	return files_with_sizes[:n]

fp1 = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_461_(c)orlandi/biamonti_461_(c)orlandi_augmented_graph_flat.pickle'
fp2 = DIRECTORY + '/datasets/beethoven/kunstderfuge/biamonti_811_(c)orlandi/biamonti_811_(c)orlandi_augmented_graph_flat.pickle'

with open(fp1, 'rb') as f:
	G1 = pickle.load(f)
with open(fp2, 'rb') as f:
	G2 = pickle.load(f)

def find_n_pickles_within_size_by_composer(min_n=4, max_n=14, max_file_size=math.inf):
	composer_files = {}
	for root, _, files in os.walk(DIRECTORY):
		for file in files:
			if file.endswith('_augmented_graph_flat.pickle'):
				path_parts = root.split(os.sep)
				composer = path_parts[-3]  # assuming the structure is: /home/ubuntu/project/datasets/{COMPOSER}/...
				
				file_path = os.path.join(root, file)
				file_size = os.path.getsize(file_path)
				csv_path = file_path.replace("_augmented_graph_flat.pickle", ".csv")
				duration = get_approx_end_time(csv_path)
				
				if composer not in composer_files:
					composer_files[composer] = []
				
				if file_size <= max_file_size:
					composer_files[composer].append((file_path, file_size, duration))
	
	# For each composer, sort the files by size and select the n smallest
	smallest_files_by_composer = {}
	for composer, files in composer_files.items():
		if len(files) >= min_n:
			files.sort(key=lambda x: x[1])
			smallest_files_by_composer[composer] = files[:max_n]
	
	return smallest_files_by_composer


G1_nodes = '''
('S0L1N1', {'label': 'S0L1', 'index': 1, 'features_dict': {'section_num': 0}, 'layer_rank': (0, 1)})
('P0O1N1', {'label': 'P0O1', 'index': 1, 'features_dict': {'pattern_num': 0}, 'layer_rank': (1, 0)})
('P0O2N2', {'label': 'P0O2', 'index': 2, 'features_dict': {'pattern_num': 0}, 'layer_rank': (1, 0)})
('PfillerN2.5', {'label': 'Pfiller', 'index': 2.5, 'features_dict': {}})
('K0QMN1', {'label': 'K0QM', 'index': 1, 'features_dict': {'relative_key_num': 0, 'key_quality': 'M'}, 'layer_rank': (2, 0)})
('C1,1QMN1', {'label': 'I', 'index': 1, 'features_dict': {'degree1': '1', 'degree2': '1', 'chord_quality': 'M'}, 'layer_rank': (3, 0)})
('C1,5QD7N2', {'label': 'V7', 'index': 2, 'features_dict': {'degree1': '1', 'degree2': '5', 'chord_quality': 'D7'}, 'layer_rank': (3, 0)})
('C1,1QMN3', {'label': 'I', 'index': 3, 'features_dict': {'degree1': '1', 'degree2': '1', 'chord_quality': 'M'}, 'layer_rank': (3, 0)})
('C1,5QD7N4', {'label': 'V7', 'index': 4, 'features_dict': {'degree1': '1', 'degree2': '5', 'chord_quality': 'D7'}, 'layer_rank': (3, 0)})
('M-2N1', {'label': 'M-2', 'index': 1, 'features_dict': {'abs_interval': 2, 'interval_sign': '-'}, 'layer_rank': (4, 0)})
('M-6N2', {'label': 'M-6', 'index': 2, 'features_dict': {'abs_interval': 6, 'interval_sign': '-'}, 'layer_rank': (4, 0)})
('M1N3', {'label': 'M1', 'index': 3, 'features_dict': {'abs_interval': 1, 'interval_sign': '+'}, 'layer_rank': (4, 0)})
('M1N4', {'label': 'M1', 'index': 4, 'features_dict': {'abs_interval': 1, 'interval_sign': '+'}, 'layer_rank': (4, 0)})
('PrSection_num:0', {'label': 'PrSection_num:0', 'layer_rank': (0, 1), 'feature_name': 'section_num', 'source_layer_kind': 'S'})
('PrPattern_num:0', {'label': 'PrPattern_num:0', 'layer_rank': (1, 0), 'feature_name': 'pattern_num', 'source_layer_kind': 'P'})
('PrPfiller:0', {'label': 'PrPfiller:0', 'layer_rank': (1, 0), 'feature_name': 'Pfiller', 'source_layer_kind': 'P'})
('PrRelative_key_num:0', {'label': 'PrRelative_key_num:0', 'layer_rank': (2, 0), 'feature_name': 'relative_key_num', 'source_layer_kind': 'K'})
('PrKey_quality:M', {'label': 'PrKey_quality:M', 'layer_rank': (2, 0), 'feature_name': 'key_quality', 'source_layer_kind': 'K'})
('PrDegree1:1', {'label': 'PrDegree1:1', 'layer_rank': (3, 0), 'feature_name': 'degree1', 'source_layer_kind': 'C'})
('PrDegree2:1', {'label': 'PrDegree2:1', 'layer_rank': (3, 0), 'feature_name': 'degree2', 'source_layer_kind': 'C'})
('PrChord_quality:M', {'label': 'PrChord_quality:M', 'layer_rank': (3, 0), 'feature_name': 'chord_quality', 'source_layer_kind': 'C'})
('PrDegree2:5', {'label': 'PrDegree2:5', 'layer_rank': (3, 0), 'feature_name': 'degree2', 'source_layer_kind': 'C'})
('PrChord_quality:D7', {'label': 'PrChord_quality:D7', 'layer_rank': (3, 0), 'feature_name': 'chord_quality', 'source_layer_kind': 'C'})
('PrAbs_interval:2', {'label': 'PrAbs_interval:2', 'layer_rank': (4, 0), 'feature_name': 'abs_interval', 'source_layer_kind': 'M'})
('PrInterval_sign:-', {'label': 'PrInterval_sign:-', 'layer_rank': (4, 0), 'feature_name': 'interval_sign', 'source_layer_kind': 'M'})
('PrAbs_interval:7', {'label': 'PrAbs_interval:7', 'layer_rank': (4, 0), 'feature_name': 'abs_interval', 'source_layer_kind': 'M'})
('PrInterval_sign:+', {'label': 'PrInterval_sign:+', 'layer_rank': (4, 0), 'feature_name': 'interval_sign', 'source_layer_kind': 'M'})
('PrAbs_interval:6', {'label': 'PrAbs_interval:6', 'layer_rank': (4, 0), 'feature_name': 'abs_interval', 'source_layer_kind': 'M'})
('PrAbs_interval:1', {'label': 'PrAbs_interval:1', 'layer_rank': (4, 0), 'feature_name': 'abs_interval', 'source_layer_kind': 'M'})
'''

G1_edges = '''
('S0L1N1', 'P0O1N1')
('S0L1N1', 'P0O2N2')
('S0L1N1', 'PfillerN2.5')
('P0O1N1', 'K0QMN1')
('P0O1N1', 'P0O2N2')
('P0O2N2', 'PfillerN2.5')
('PfillerN2.5', 'K0QMN1')
('K0QMN1', 'C1,1QMN1')
('K0QMN1', 'C1,5QD7N2')
('K0QMN1', 'C1,1QMN3')
('K0QMN1', 'C1,5QD7N4')
('C1,1QMN1', 'M-2N1')
('C1,1QMN1', 'M-6N2')
('C1,1QMN1', 'C1,5QD7N2')
('C1,5QD7N2', 'M-6N2')
('C1,5QD7N2', 'M1N3')
('C1,5QD7N2', 'M1N4')
('C1,5QD7N2', 'C1,1QMN3')
('C1,1QMN3', 'C1,5QD7N4')
('C1,5QD7N4', 'M1N4')
('M1N3', 'M1N4')
('PrSection_num:0', 'S0L1N1')
('PrPattern_num:0', 'P0O1N1')
('PrPattern_num:0', 'P0O2N2')
('PrPfiller:0', 'PfillerN2.5')
('PrRelative_key_num:0', 'K0QMN1')
('PrKey_quality:M', 'K0QMN1')
('PrDegree1:1', 'C1,1QMN1')
('PrDegree1:1', 'C1,5QD7N2')
('PrDegree1:1', 'C1,1QMN3')
('PrDegree1:1', 'C1,5QD7N4')
('PrDegree2:1', 'C1,1QMN1')
('PrDegree2:1', 'C1,1QMN3')
('PrChord_quality:M', 'C1,1QMN1')
('PrChord_quality:M', 'C1,1QMN3')
('PrDegree2:5', 'C1,5QD7N2')
('PrDegree2:5', 'C1,5QD7N4')
('PrChord_quality:D7', 'C1,5QD7N2')
('PrChord_quality:D7', 'C1,5QD7N4')
('PrAbs_interval:2', 'M-2N1')
('PrInterval_sign:-', 'M-2N1')
('PrInterval_sign:-', 'M-6N2')
('PrInterval_sign:+', 'M1N3')
('PrInterval_sign:+', 'M1N4')
('PrAbs_interval:6', 'M-6N2')
('PrAbs_interval:1', 'M1N3')
('PrAbs_interval:1', 'M1N4')
'''

G1_test = nx.DiGraph()
for line in G1_nodes.strip().split('\n'):
	if not line.strip().startswith('#'): # skip "commented out" lines
		node_id, metadata = ast.literal_eval(line)
		G1_test.add_node(node_id, **metadata)
for line in G1_edges.strip().split('\n'):
	if not line.strip().startswith('#'):
		edge = ast.literal_eval(line)
		G1_test.add_edge(*edge)

G2_test = G1_test.copy()
# new_nodes = '''
#   ('NewNode1', {'some_attribute': 'value1'})
#   ('NewNode2', {'some_attribute': 'value2'})
#   ('NewNode3', {'some_attribute': 'value3'})
# '''

# for line in new_nodes.strip().split('\n'):
# 	if not line.strip().startswith('#'):  # skip "commented out" lines
# 		new_node_id, new_node_metadata = ast.literal_eval(line)
# 		G2_test.add_node(new_node_id, **new_node_metadata)

G1_edges_to_remove = [('C1,5QD7N2', 'M1N4')]
for edge in G1_edges_to_remove:
	if G2_test.has_edge(*edge):
		G2_test.remove_edge(*edge)

G1_edges_to_add = [('C1,1QMN3', 'M1N4')]
for edge in G1_edges_to_add:
	if not G2_test.has_edge(*edge):
		G2_test.add_edge(*edge)

if __name__ == "__main__":
	smallest_files = find_n_smallest_pickles(n=15)
	for file in smallest_files:
		print(file)

	smallest_files_dict = find_n_pickles_within_size_by_composer(min_n=7, max_n=14, max_file_size=50000)
	for composer, files in smallest_files_dict.items():
		print(f"Composer: {composer.upper()}. Corpus size: {len(files)}")
		for file in files:
				print(f" File: {file[0]}, Size: {file[1]} bytes, Duration: {file[2]} sec")
	sys.exit(0)

	layers_G1_test = build_graph.get_unsorted_layers_from_graph_by_index(G1_test)
	layers_G2_test = build_graph.get_unsorted_layers_from_graph_by_index(G2_test)
	build_graph.visualize_p([G1_test, G2_test], [layers_G1_test, layers_G2_test])