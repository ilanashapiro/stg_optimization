import pickle
import os, sys
import networkx as nx
import ast

def find_two_smallest_pickles(directory='/home/ilshapiro/project/datasets'):
  smallest_file = None
  smallest_size = float('inf')
  second_smallest_file = None
  second_smallest_size = float('inf')
  
  for root, _, files in os.walk(directory):
    for file in files:
      if file.endswith('_augmented_graph_flat.pickle'):
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path)
        
        if file_size < smallest_size:
          second_smallest_size = smallest_size
          second_smallest_file = smallest_file
          
          smallest_size = file_size
          smallest_file = file_path
        elif file_size < second_smallest_size:
          second_smallest_size = file_size
          second_smallest_file = file_path
                    
  return (smallest_file, smallest_size), (second_smallest_file, second_smallest_size)

fp1 = '/home/ilshapiro/project/datasets/beethoven/kunstderfuge/biamonti_461_(c)orlandi/biamonti_461_(c)orlandi_augmented_graph_flat.pickle'
fp2 = '/home/ilshapiro/project/datasets/beethoven/kunstderfuge/biamonti_811_(c)orlandi/biamonti_811_(c)orlandi_augmented_graph_flat.pickle'

if __name__ == "__main__":
  smallest, second_smallest = find_two_smallest_pickles()
  print(smallest, second_smallest)

  with open(fp1, 'rb') as f:
    G1 = pickle.load(f)
  with open(fp2, 'rb') as f:
    G2 = pickle.load(f)

  # for node in G1.nodes(data=True):
  #   print(node)
  # print()
  # for edge in G1.edges():
  #   print(edge)

  G1_nodes = '''
    ('S0L1N1', {'start': 0.0, 'end': 4.8, 'label': 'S0L1', 'index': 1, 'features_dict': {'section_num': 0}, 'layer_rank': (0, 1)})
    ('P0O1N1', {'start': 0.0, 'end': 1.05, 'label': 'P0O1', 'index': 1, 'features_dict': {'pattern_num': 0}, 'layer_rank': (1, 0)})
    ('P0O2N2', {'start': 1.2, 'end': 2.25, 'label': 'P0O2', 'index': 2, 'features_dict': {'pattern_num': 0}, 'layer_rank': (1, 0)})
    ('PfillerN2.5', {'start': 2.25, 'end': 4.8, 'label': 'Pfiller', 'index': 2.5, 'features_dict': {}})
    ('K0QMN1', {'start': 0.0, 'end': 4.8, 'label': 'K0QM', 'index': 1, 'features_dict': {'relative_key_num': 0, 'key_quality': 'M'}, 'layer_rank': (2, 0)})
    ('C1,1QMN1', {'start': 0.0, 'end': 2.5500000000000003, 'label': 'I', 'index': 1, 'features_dict': {'degree1': '1', 'degree2': '1', 'chord_quality': 'M'}, 'layer_rank': (3, 0)})
    ('C1,5QD7N2', {'start': 2.5500000000000003, 'end': 3.15, 'label': 'V7', 'index': 2, 'features_dict': {'degree1': '1', 'degree2': '5', 'chord_quality': 'D7'}, 'layer_rank': (3, 0)})
    ('C1,1QMN3', {'start': 3.15, 'end': 3.3000000000000003, 'label': 'I', 'index': 3, 'features_dict': {'degree1': '1', 'degree2': '1', 'chord_quality': 'M'}, 'layer_rank': (3, 0)})
    ('C1,5QD7N4', {'start': 3.3000000000000003, 'end': 4.2, 'label': 'V7', 'index': 4, 'features_dict': {'degree1': '1', 'degree2': '5', 'chord_quality': 'D7'}, 'layer_rank': (3, 0)})
    ('C1,1QMN5', {'start': 4.2, 'end': 4.5, 'label': 'I', 'index': 5, 'features_dict': {'degree1': '1', 'degree2': '1', 'chord_quality': 'M'}, 'layer_rank': (3, 0)})
    ('C1,5QD7N6', {'start': 4.5, 'end': 4.8, 'label': 'V7', 'index': 6, 'features_dict': {'degree1': '1', 'degree2': '5', 'chord_quality': 'D7'}, 'layer_rank': (3, 0)})
    ('M-2N1', {'start': 0.0, 'end': 0.470204082, 'label': 'M-2', 'index': 1, 'features_dict': {'abs_interval': 2, 'interval_sign': '-'}, 'layer_rank': (4, 0)})
    ('M-7N2', {'start': 0.470204082, 'end': 0.635646259, 'label': 'M-7', 'index': 2, 'features_dict': {'abs_interval': 7, 'interval_sign': '-'}, 'layer_rank': (4, 0)})
    ('M4N3', {'start': 0.635646259, 'end': 0.777868481, 'label': 'M4', 'index': 3, 'features_dict': {'abs_interval': 4, 'interval_sign': '+'}, 'layer_rank': (4, 0)})
    ('M-16N4', {'start': 0.777868481, 'end': 1.224852608, 'label': 'M-16', 'index': 4, 'features_dict': {'abs_interval': 16, 'interval_sign': '-'}, 'layer_rank': (4, 0)})
    ('M4N5', {'start': 1.224852608, 'end': 1.515102041, 'label': 'M4', 'index': 5, 'features_dict': {'abs_interval': 4, 'interval_sign': '+'}, 'layer_rank': (4, 0)})
    ('M5N6', {'start': 1.515102041, 'end': 1.520907029, 'label': 'M5', 'index': 6, 'features_dict': {'abs_interval': 5, 'interval_sign': '+'}, 'layer_rank': (4, 0)})
    ('M-6N7', {'start': 1.520907029, 'end': 2.7138322, 'label': 'M-6', 'index': 7, 'features_dict': {'abs_interval': 6, 'interval_sign': '-'}, 'layer_rank': (4, 0)})
    ('M-1N8', {'start': 2.7138322, 'end': 2.731247166, 'label': 'M-1', 'index': 8, 'features_dict': {'abs_interval': 1, 'interval_sign': '-'}, 'layer_rank': (4, 0)})
    ('M1N9', {'start': 2.731247166, 'end': 3.027301587, 'label': 'M1', 'index': 9, 'features_dict': {'abs_interval': 1, 'interval_sign': '+'}, 'layer_rank': (4, 0)})
    ('M1N10', {'start': 3.027301587, 'end': 3.03600907, 'label': 'M1', 'index': 10, 'features_dict': {'abs_interval': 1, 'interval_sign': '+'}, 'layer_rank': (4, 0)})
    ('M1N11', {'start': 3.03600907, 'end': 3.334965986, 'label': 'M1', 'index': 11, 'features_dict': {'abs_interval': 1, 'interval_sign': '+'}, 'layer_rank': (4, 0)})
    ('M-1N12', {'start': 3.334965986, 'end': 4.220226757, 'label': 'M-1', 'index': 12, 'features_dict': {'abs_interval': 1, 'interval_sign': '-'}, 'layer_rank': (4, 0)})
    ('M-2N13', {'start': 4.220226757, 'end': 4.536598639, 'label': 'M-2', 'index': 13, 'features_dict': {'abs_interval': 2, 'interval_sign': '-'}, 'layer_rank': (4, 0)})
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
    ('PrAbs_interval:4', {'label': 'PrAbs_interval:4', 'layer_rank': (4, 0), 'feature_name': 'abs_interval', 'source_layer_kind': 'M'})
    ('PrInterval_sign:+', {'label': 'PrInterval_sign:+', 'layer_rank': (4, 0), 'feature_name': 'interval_sign', 'source_layer_kind': 'M'})
    ('PrAbs_interval:16', {'label': 'PrAbs_interval:16', 'layer_rank': (4, 0), 'feature_name': 'abs_interval', 'source_layer_kind': 'M'})
    ('PrAbs_interval:5', {'label': 'PrAbs_interval:5', 'layer_rank': (4, 0), 'feature_name': 'abs_interval', 'source_layer_kind': 'M'})
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
  ('K0QMN1', 'C1,1QMN5')
  ('K0QMN1', 'C1,5QD7N6')
  ('C1,1QMN1', 'M-2N1')
  ('C1,1QMN1', 'M-7N2')
  ('C1,1QMN1', 'M4N3')
  ('C1,1QMN1', 'M-16N4')
  ('C1,1QMN1', 'M4N5')
  ('C1,1QMN1', 'M5N6')
  ('C1,1QMN1', 'M-6N7')
  ('C1,1QMN1', 'C1,5QD7N2')
  ('C1,5QD7N2', 'M-6N7')
  ('C1,5QD7N2', 'M-1N8')
  ('C1,5QD7N2', 'M1N9')
  ('C1,5QD7N2', 'M1N10')
  ('C1,5QD7N2', 'M1N11')
  ('C1,5QD7N2', 'C1,1QMN3')
  ('C1,1QMN3', 'C1,5QD7N4')
  ('C1,5QD7N4', 'M1N11')
  ('C1,5QD7N4', 'M-1N12')
  ('C1,5QD7N4', 'C1,1QMN5')
  ('C1,1QMN5', 'M-1N12')
  ('C1,1QMN5', 'M-2N13')
  ('C1,1QMN5', 'C1,5QD7N6')
  ('C1,5QD7N6', 'M-2N13')
  ('M-2N1', 'M-7N2')
  ('M-7N2', 'M4N3')
  ('M4N3', 'M-16N4')
  ('M-16N4', 'M4N5')
  ('M4N5', 'M5N6')
  ('M5N6', 'M-6N7')
  ('M-6N7', 'M-1N8')
  ('M-1N8', 'M1N9')
  ('M1N9', 'M1N10')
  ('M1N10', 'M1N11')
  ('M1N11', 'M-1N12')
  ('M-1N12', 'M-2N13')
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
  ('PrDegree1:1', 'C1,1QMN5')
  ('PrDegree1:1', 'C1,5QD7N6')
  ('PrDegree2:1', 'C1,1QMN1')
  ('PrDegree2:1', 'C1,1QMN3')
  ('PrDegree2:1', 'C1,1QMN5')
  ('PrChord_quality:M', 'C1,1QMN1')
  ('PrChord_quality:M', 'C1,1QMN3')
  ('PrChord_quality:M', 'C1,1QMN5')
  ('PrDegree2:5', 'C1,5QD7N2')
  ('PrDegree2:5', 'C1,5QD7N4')
  ('PrDegree2:5', 'C1,5QD7N6')
  ('PrChord_quality:D7', 'C1,5QD7N2')
  ('PrChord_quality:D7', 'C1,5QD7N4')
  ('PrChord_quality:D7', 'C1,5QD7N6')
  ('PrAbs_interval:2', 'M-2N1')
  ('PrAbs_interval:2', 'M-2N13')
  ('PrInterval_sign:-', 'M-2N1')
  ('PrInterval_sign:-', 'M-7N2')
  ('PrInterval_sign:-', 'M-16N4')
  ('PrInterval_sign:-', 'M-6N7')
  ('PrInterval_sign:-', 'M-1N8')
  ('PrInterval_sign:-', 'M-1N12')
  ('PrInterval_sign:-', 'M-2N13')
  ('PrAbs_interval:7', 'M-7N2')
  ('PrAbs_interval:4', 'M4N3')
  ('PrAbs_interval:4', 'M4N5')
  ('PrInterval_sign:+', 'M4N3')
  ('PrInterval_sign:+', 'M4N5')
  ('PrInterval_sign:+', 'M5N6')
  ('PrInterval_sign:+', 'M1N9')
  ('PrInterval_sign:+', 'M1N10')
  ('PrInterval_sign:+', 'M1N11')
  ('PrAbs_interval:16', 'M-16N4')
  ('PrAbs_interval:5', 'M5N6')
  ('PrAbs_interval:6', 'M-6N7')
  ('PrAbs_interval:1', 'M-1N8')
  ('PrAbs_interval:1', 'M1N9')
  ('PrAbs_interval:1', 'M1N10')
  ('PrAbs_interval:1', 'M1N11')
  ('PrAbs_interval:1', 'M-1N12')
'''

G1_test = nx.DiGraph()
for line in G1_nodes.strip().split('\n'):
  node_id, metadata = ast.literal_eval(line)
  G1_test.add_node(node_id, **metadata)
for line in G1_edges.strip().split('\n'):
  edge = ast.literal_eval(line)
  G1_test.add_edge(*edge)