import networkx as nx
import numpy as np
import build_graph

def add_labels(G):
  # Label nodes
  node_labels = {node: node for node in G.nodes()}  # Create a dict where each node maps to its own ID
  nx.set_node_attributes(G, node_labels, 'label')  # Set the 'label' attribute of each node

  # Label edges
  edge_labels = {(u, v): f"({u}, {v})" for u, v in G.edges()}  # Create a dict of edge labels based on node IDs
  nx.set_edge_attributes(G, edge_labels, 'label')  # Set the 'label' attribute of each edge

G1 = nx.DiGraph()
G1.add_node('PrS0')
G1.add_node('PrS1')
G1.add_node('PrS2')
G1.add_node('PrP0')
G1.add_node('PrP1')
G1.add_node('S0L1N1')
G1.add_node('S1L1N2')
G1.add_node('S0L1N3')
G1.add_node('S0L2N1')
G1.add_node('S1L2N2')
G1.add_node('S2L2N3')
G1.add_node('S1L2N4')
G1.add_node('P1O1N1')
G1.add_node('P0O1N2')
G1.add_node('P1O2N3')

G1.add_edge('PrS0', 'S0L1N1')
G1.add_edge('PrS0', 'S0L1N3')
G1.add_edge('PrS0', 'S0L2N1')
G1.add_edge('PrS1', 'S1L1N2')
G1.add_edge('PrS1', 'S1L2N2')
G1.add_edge('PrS1', 'S1L2N4')
G1.add_edge('PrS2', 'S2L2N3')
G1.add_edge('PrP0', 'P0O1N2')
G1.add_edge('PrP1', 'P1O1N1')
G1.add_edge('PrP1', 'P1O2N3')

G1.add_edge('S0L1N1', 'S1L1N2')
G1.add_edge('S0L1N1', 'S0L2N1')
G1.add_edge('S0L1N1', 'S1L2N2')
G1.add_edge('S1L1N2', 'S0L1N3')
G1.add_edge('S1L1N2', 'S1L2N2')
G1.add_edge('S1L1N2', 'S2L2N3')
G1.add_edge('S0L1N3', 'S2L2N3')
G1.add_edge('S0L1N3', 'S1L2N4')
G1.add_edge('S0L2N1', 'S1L2N2')
G1.add_edge('S0L2N1', 'P1O1N1')
G1.add_edge('S1L2N2', 'S2L2N3')
G1.add_edge('S1L2N2', 'P0O1N2')
G1.add_edge('S2L2N3', 'S1L2N4')
G1.add_edge('S2L2N3', 'P1O2N3')
G1.add_edge('S1L2N4', 'P1O2N3')
G1.add_edge('P1O1N1', 'P0O1N2')
G1.add_edge('P0O1N2', 'P1O2N3')

G2 = nx.DiGraph()
G2.add_node('PrS0')
G2.add_node('PrS1')
G2.add_node('PrP0')
G2.add_node('PrP1')
G2.add_node('S0L1N1')
G2.add_node('S1L1N2')
G2.add_node('S0L2N1')
G2.add_node('S1L2N2')
G2.add_node('S0L2N3')
G2.add_node('P0O1N1')
G2.add_node('P1O1N2')
G2.add_node('P0O2N3')

G2.add_edge('PrS0', 'S0L1N1')
G2.add_edge('PrS0', 'S0L2N1')
G2.add_edge('PrS0', 'S0L2N3')
G2.add_edge('PrS1', 'S1L1N2')
G2.add_edge('PrS1', 'S1L2N2')
G2.add_edge('PrP0', 'P0O1N1')
G2.add_edge('PrP0', 'P0O2N3')
G2.add_edge('PrP1', 'P1O1N2')

G2.add_edge('S0L1N1', 'S1L1N2')
G2.add_edge('S0L1N1', 'S0L2N1')
G2.add_edge('S1L1N2', 'S1L2N2')
G2.add_edge('S1L1N2', 'S0L2N3')
G2.add_edge('S0L2N1', 'S1L2N2')
G2.add_edge('S0L2N1', 'P0O1N1')
G2.add_edge('S1L2N2', 'S0L2N3')
G2.add_edge('S1L2N2', 'P0O1N1')
G2.add_edge('S1L2N2', 'P1O1N2')
G2.add_edge('S0L2N3', 'P0O2N3')
G2.add_edge('P0O1N1', 'P1O1N2')
G2.add_edge('P1O1N2', 'P0O2N3')

if __name__ == "main":
  add_labels(G1)
  add_labels(G2)
  layers1 = build_graph.get_unsorted_layers_from_graph_by_index(G1) 
  layers2 = build_graph.get_unsorted_layers_from_graph_by_index(G2) 
  build_graph.visualize([G1, G2], [layers1, layers2])