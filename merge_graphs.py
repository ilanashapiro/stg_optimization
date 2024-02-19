import build_graph
import networkx as nx

def get_node_labels(G):
	return nx.get_node_attributes(G, 'label').values()

def get_edge_labels(G):
	return nx.get_edge_attributes(G, 'label').values()

def common_subgraph(G1, G2):
	common_node_labels = set(get_node_labels(G1)) & set(get_node_labels(G2))
	common_edge_labels = set(get_edge_labels(G1)) & set(get_edge_labels(G2))

	def get_label_xs_dict(G, x_labels_dict):
		label_xs_dict = {}

		for key, value in x_labels_dict.items():
			if value not in label_xs_dict:
				label_xs_dict[value] = []
			label_xs_dict[value].append(key)
		
		return label_xs_dict
	
	common_node_ids = []
	for label in common_node_labels:
		node_labels_dict = nx.get_node_attributes(G1, 'label')
		common_node_ids.extend(get_label_xs_dict(G1, node_labels_dict).get(label, []))
	common_nodes = [id_data_tuple for id_data_tuple in G1.nodes(data=True) if id_data_tuple[0] in common_node_ids]

	common_edge_ids = []
	for label in common_edge_labels:
		edge_labels_dict = nx.get_edge_attributes(G1, 'label')
		common_edge_ids.extend(get_label_xs_dict(G1, edge_labels_dict).get(label, []))
	common_edges = [id_data_tuple for id_data_tuple in G1.edges(data=True) if id_data_tuple[:2] in common_edge_ids]

	G = nx.DiGraph()
	G.add_nodes_from(common_nodes)
	G.add_edges_from(common_edges)

	return G

(G0, layers0, label_dict0) = build_graph.generate_graph('LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short_segments.txt', 'LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short_motives.txt')
(G1, layers1, label_dict1) = build_graph.generate_graph('LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short_segments.txt', 'LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short_motives.txt')

G = common_subgraph(G0, G1)
common_layers = [[node_data for node_data in layer if node_data['label'] in get_node_labels(G)] for layer in layers0]
common_layers = [layer for layer in common_layers if layer] # Remove empty sublists from the result

build_graph.visualize([G0, G1, G], [layers0, layers1, common_layers], [label_dict0, label_dict1, nx.get_node_attributes(G, 'label')])