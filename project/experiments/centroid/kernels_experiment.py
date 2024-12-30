import os, sys, pickle, json, re
import numpy as np
import spectral_experiment

from grakel.kernels import WeisfeilerLehman, NeighborhoodSubgraphPairwiseDistance, RandomWalk, MultiscaleLaplacian, ShortestPath, PyramidMatch, SvmTheta
from grakel import Graph, NeighborhoodHash, SubgraphMatching, VertexHistogram, WeisfeilerLehmanOptimalAssignment, GraphKernel

from sklearn.metrics.pairwise import pairwise_kernels
from scipy.stats import wilcoxon

# DIRECTORY = "/home/ubuntu/project"
# DIRECTORY = "/Users/ilanashapiro/Documents/constraints_project/project"
DIRECTORY = "/home/ilshapiro/project"
TIME_PARAM = '50s'
ABLATION_LEVEL = None # set to None if we don't want ablation

sys.path.append(f"{DIRECTORY}/centroid")
import simanneal_centroid_helpers

if __name__ == "__main__":
	if ABLATION_LEVEL:
		centroid_path = f"{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}_ablation{ABLATION_LEVEL}"
	else:
		centroid_path = f"{DIRECTORY}/experiments/centroid/final_centroids/final_centroid_{TIME_PARAM}"
	
	training_pieces_path = f"{DIRECTORY}/experiments/centroid/clusters/composer_centroid_input_graphs_{TIME_PARAM}.txt"
	
	composer_centroids_dict = spectral_experiment.load_centroids() 
	composer_centroids_dict = {k: composer_centroids_dict[k] for k in sorted(composer_centroids_dict)} # to ensure deterministic order

	with open(training_pieces_path, 'r') as file:
		composer_training_pieces_paths = json.load(file)
		def load_graph(file_path):
			with open(file_path, 'rb') as f:
				graph = pickle.load(f)
			return graph
		composer_training_pieces_dict = {}
		for composer, filepaths in composer_training_pieces_paths.items():
			if ABLATION_LEVEL:
				new_suffix = f"_ablation_{ABLATION_LEVEL}level_flat.pickle"
				composer_training_pieces_dict[composer] = [load_graph(re.sub(r'^.*?/project', DIRECTORY, file_path)[:-len("_flat.pickle")] + new_suffix) for file_path in filepaths]
			else:
				composer_training_pieces_dict[composer] = [load_graph(re.sub(r'^.*?/project', DIRECTORY, file_path)) for file_path in filepaths]

	for composer, centroid in composer_centroids_dict.items():
		training_pieces = composer_training_pieces_dict[composer]
		listA_G, idx_node_mapping, nodes_features_dict = simanneal_centroid_helpers.pad_adj_matrices([centroid] + training_pieces)
		
		grakel_graphs = []
		for adjacency_matrix in listA_G:
			labels = {i: f'node{i}' for i in range(adjacency_matrix.shape[0])}  # Dummy labels
			edge_labels = {(i, j): 'edge{i},{j}' for i in range(adjacency_matrix.shape[0]) for j in range(adjacency_matrix.shape[1]) if adjacency_matrix[i, j] > 0}  # Dummy edge labels
			graph = Graph(adjacency_matrix, node_labels=labels, edge_labels=edge_labels)
			grakel_graphs.append(graph)

		wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=NeighborhoodSubgraphPairwiseDistance)
		random_walk_kernel = RandomWalk(normalize=True) # Intractable for my graphs
		shortest_path_kernel = ShortestPath(normalize=True) # Intractable for my graphs
		nspd_kernel = NeighborhoodSubgraphPairwiseDistance(normalize=True)
		neighborhood_hash_kernel = NeighborhoodHash(normalize=True) # Not accurate enough
		pm_kernel = PyramidMatch(normalize=True)
		svm_kernel = SvmTheta(normalize=True)

		scores = []

		# Iterate over each graph in the corpus, treating each as the candidate centroid
		for i in [len(grakel_graphs)-1]:
			# Create a new list where each candidate graph becomes the "test centroid" (first is original centroid, then duplicating the test graphs to replace original centroid)
			test_grakel_graphs = [grakel_graphs[i]] + grakel_graphs[1:]
			
			# Compute the kernel matrix
			kernel_matrix = wl_kernel.fit_transform(test_grakel_graphs)
			dist_matrix = 1 - kernel_matrix

			# dist_matrix = np.array([1 - pairwise_kernels([listA_G[i].flatten()], [A_G.flatten() for A_G in listA_G[1:]], metric='sigmoid')])

			# Calculate the score, excluding the first graph (the "centroid" itself)
			mean_dist = np.mean(dist_matrix)
			std_dist  = np.std(dist_matrix)
			score = mean_dist * std_dist 
			scores.append(score)

			print(f"Score when graph {i} is treated as the centroid: {score}")

		best_score = min(scores)
		best_graph_index = scores.index(best_score)

		print(f"The most representative graph in the corpus is graph {best_graph_index} with a score of {best_score}, and ablation {ABLATION_LEVEL}.")

#--------------------------------------------------------------------------------------------------------------------------------------------
# RESULT FOR WL KERNEL, NSPD BASE KERNEL, NO ABLATION/FULL STG
# BACH
# Score when graph 0 is treated as the centroid: 0.22802259602060843
# Score when graph 1 is treated as the centroid: 0.24160071482745488
# Score when graph 2 is treated as the centroid: 0.24183587552258085
# Score when graph 3 is treated as the centroid: 0.24328109599549771
# Score when graph 4 is treated as the centroid: 0.24218880937233003
# Score when graph 5 is treated as the centroid: 0.246957896075718
# Score when graph 6 is treated as the centroid: 0.24297885674826092
# The most representative graph in the corpus is graph 0 with a score of 0.22802259602060843 and ablation None.

# BEETHOVEN
# Score when graph 0 is treated as the centroid: 0.2005583406491981
# Score when graph 1 is treated as the centroid: 0.2137014380623202
# Score when graph 2 is treated as the centroid: 0.20963024029010083
# Score when graph 3 is treated as the centroid: 0.209725006591298
# Score when graph 4 is treated as the centroid: 0.21039404281795526
# Score when graph 5 is treated as the centroid: 0.20957912088393196
# Score when graph 6 is treated as the centroid: 0.20920196914619632
# Score when graph 7 is treated as the centroid: 0.210070738351469
# Score when graph 8 is treated as the centroid: 0.21120417826175442
# Score when graph 9 is treated as the centroid: 0.21214919552963876
# The most representative graph in the corpus is graph 0 with a score of 0.2005583406491981 and ablation None.

# HAYDN
# Score when graph 0 is treated as the centroid: 0.25947540266761876
# Score when graph 1 is treated as the centroid: 0.26390112381707
# Score when graph 2 is treated as the centroid: 0.2661881493909181
# Score when graph 3 is treated as the centroid: 0.26756240021479816
# Score when graph 4 is treated as the centroid: 0.2661894043561427
# The most representative graph in the corpus is graph 0 with a score of 0.25947540266761876 and ablation None.

# MOZART
# Score when graph 0 is treated as the centroid: 0.24459634867984967
# Score when graph 1 is treated as the centroid: 0.2580619729972281
# Score when graph 2 is treated as the centroid: 0.2568975067463745
# Score when graph 3 is treated as the centroid: 0.2549755847604174
# Score when graph 4 is treated as the centroid: 0.2563505287774512
# Score when graph 5 is treated as the centroid: 0.2573692476400684
# The most representative graph in the corpus is graph 0 with a score of 0.24459634867984967 and ablation None.

#--------------------------------------------------------------------------------------------------------------------------------------------
# RESULT FOR WL KERNEL, NSPD BASE KERNEL, ABLATION 4 LEVEL
# BACH
# Score when graph 0 is treated as the centroid: 0.22715760726017842
# Score when graph 1 is treated as the centroid: 0.2365351773717528
# Score when graph 2 is treated as the centroid: 0.23625622074277594
# Score when graph 3 is treated as the centroid: 0.23845456158662373
# Score when graph 4 is treated as the centroid: 0.23736722954963174
# Score when graph 5 is treated as the centroid: 0.24273667774109667
# The most representative graph in the corpus is graph 0 with a score of 0.22715760726017842 and ablation 4.

# BEETHOVEN
# Score when graph 0 is treated as the centroid: 0.19933224165685007
# Score when graph 1 is treated as the centroid: 0.20908762447318113
# Score when graph 2 is treated as the centroid: 0.20455428200837275
# Score when graph 3 is treated as the centroid: 0.2046610425099915
# Score when graph 4 is treated as the centroid: 0.20540468524607008
# Score when graph 5 is treated as the centroid: 0.20396287936431662
# Score when graph 6 is treated as the centroid: 0.20373949853617562
# Score when graph 7 is treated as the centroid: 0.2049522350221997
# Score when graph 8 is treated as the centroid: 0.20589123762167652
# The most representative graph in the corpus is graph 0 with a score of 0.19933224165685007 and ablation 4.

# HAYDN
# Score when graph 0 is treated as the centroid: 0.25606898475650774
# Score when graph 1 is treated as the centroid: 0.25126947868960536
# Score when graph 2 is treated as the centroid: 0.2543817130666127
# Score when graph 3 is treated as the centroid: 0.2559068591357299
# The most representative graph in the corpus is graph 1 with a score of 0.25126947868960536 and ablation 4.

# MOZART
# Score when graph 0 is treated as the centroid: 0.24449901441450367
# Score when graph 1 is treated as the centroid: 0.2499085969899553
# Score when graph 2 is treated as the centroid: 0.24958528294009197
# Score when graph 3 is treated as the centroid: 0.2465589770852779
# Score when graph 4 is treated as the centroid: 0.24978227125531804
# The most representative graph in the corpus is graph 0 with a score of 0.24449901441450367 and ablation 4.

#--------------------------------------------------------------------------------------------------------------------------------------------
# RESULT FOR WL KERNEL, NSPD BASE KERNEL, ABLATION 3 LEVEL
# BACH
# Score when graph 0 is treated as the centroid: 0.2235027348057916
# Score when graph 1 is treated as the centroid: 0.22584092989207213
# Score when graph 2 is treated as the centroid: 0.22572351347450156
# Score when graph 3 is treated as the centroid: 0.22699395196586328
# Score when graph 4 is treated as the centroid: 0.22514788528459417
# Score when graph 5 is treated as the centroid: 0.23336995661613597
# The most representative graph in the corpus is graph 0 with a score of 0.2235027348057916, and ablation 3.

# BEETHOVEN
# Score when graph 0 is treated as the centroid: 0.19734524716800553
# Score when graph 1 is treated as the centroid: 0.20038537075077864
# Score when graph 2 is treated as the centroid: 0.19079865552220326
# Score when graph 3 is treated as the centroid: 0.19713557317377628
# Score when graph 4 is treated as the centroid: 0.19746719497316922
# Score when graph 5 is treated as the centroid: 0.19007644317435868
# Score when graph 6 is treated as the centroid: 0.19011135637947882
# Score when graph 7 is treated as the centroid: 0.19648888954151722
# Score when graph 8 is treated as the centroid: 0.19743309599824443
# The most representative graph in the corpus is graph 5 with a score of 0.19007644317435868, and ablation 3.

# HAYDN
# Score when graph 0 is treated as the centroid: 0.25074962701564085
# Score when graph 1 is treated as the centroid: 0.23505099489190584
# Score when graph 2 is treated as the centroid: 0.23851702612399828
# Score when graph 3 is treated as the centroid: 0.2412951720029888
# The most representative graph in the corpus is graph 1 with a score of 0.23505099489190584, and ablation 3.

# MOZART
# Score when graph 0 is treated as the centroid: 0.24047101154933923
# Score when graph 1 is treated as the centroid: 0.22836056004776387
# Score when graph 2 is treated as the centroid: 0.23140667272625104
# Score when graph 3 is treated as the centroid: 0.2275694525490538
# Score when graph 4 is treated as the centroid: 0.2290165534340957
# The most representative graph in the corpus is graph 3 with a score of 0.2275694525490538, and ablation 3.

#--------------------------------------------------------------------------------------------------------------------------------------------
# RESULT FOR WL KERNEL, NSPD BASE KERNEL, ABLATION 2 LEVEL
# BACH
# Score when graph 0 is treated as the centroid: 0.22281902460594707
# Score when graph 1 is treated as the centroid: 0.22400465149007445
# Score when graph 2 is treated as the centroid: 0.22404773509508885
# Score when graph 3 is treated as the centroid: 0.22465293925116878
# Score when graph 4 is treated as the centroid: 0.22333533530564814
# Score when graph 5 is treated as the centroid: 0.2303029617397209
# The most representative graph in the corpus is graph 0 with a score of 0.22281902460594707, and ablation 2.

# BEETHOVEN
# Score when graph 0 is treated as the centroid: 0.20220098489584623
# Score when graph 1 is treated as the centroid: 0.20132929118769446
# Score when graph 2 is treated as the centroid: 0.19114064820241689
# Score when graph 3 is treated as the centroid: 0.1911985087851552
# Score when graph 4 is treated as the centroid: 0.19132790489708698
# Score when graph 5 is treated as the centroid: 0.19069859304840106
# Score when graph 6 is treated as the centroid: 0.19421171365982492
# Score when graph 7 is treated as the centroid: 0.19664899663778868
# Score when graph 8 is treated as the centroid: 0.19421171365982492
# The most representative graph in the corpus is graph 5 with a score of 0.19069859304840106, and ablation 2.

# HAYDN
# Score when graph 0 is treated as the centroid: 0.24958212614505584
# Score when graph 1 is treated as the centroid: 0.23192528714701272
# Score when graph 2 is treated as the centroid: 0.23545199758724067
# Score when graph 3 is treated as the centroid: 0.23827854278748636
# The most representative graph in the corpus is graph 1 with a score of 0.23192528714701272, and ablation 2.

# MOZART
# Score when graph 0 is treated as the centroid: 0.2387045783353854
# Score when graph 1 is treated as the centroid: 0.22214900110216412
# Score when graph 2 is treated as the centroid: 0.22472389977667853
# Score when graph 3 is treated as the centroid: 0.22227682338294863
# Score when graph 4 is treated as the centroid: 0.22368330905440337
# The most representative graph in the corpus is graph 1 with a score of 0.22214900110216412, and ablation 2.

#--------------------------------------------------------------------------------------------------------------------------------------------
# RESULT FOR WL KERNEL, NSPD BASE KERNEL, ABLATION 1 LEVEL
# BACH
# Score when graph 0 is treated as the centroid: 0.22639778696014212
# Score when graph 1 is treated as the centroid: 0.17457490025664996
# Score when graph 2 is treated as the centroid: 0.18425732586760116
# Score when graph 3 is treated as the centroid: 0.17457490025664996
# Score when graph 4 is treated as the centroid: 0.1832150208504833
# Score when graph 5 is treated as the centroid: 0.18811757288605285
# The most representative graph in the corpus is graph 1 with a score of 0.17457490025664996 and ablation 1.

# BEETHOVEN
# Score when graph 0 is treated as the centroid: 0.20970084562589822
# Score when graph 1 is treated as the centroid: 0.16494920257860898
# Score when graph 2 is treated as the centroid: 0.16628350875368728
# Score when graph 3 is treated as the centroid: 0.16628350875368728
# Score when graph 4 is treated as the centroid: 0.16161568551413666
# Score when graph 5 is treated as the centroid: 0.16161568551413666
# Score when graph 6 is treated as the centroid: 0.16161568551413666
# Score when graph 7 is treated as the centroid: 0.16494920257860898
# Score when graph 8 is treated as the centroid: 0.16161568551413666
# The most representative graph in the corpus is graph 4 with a score of 0.16161568551413666 and ablation 1.

# HAYDN
# Score when graph 0 is treated as the centroid: 0.22234614740581826
# Score when graph 1 is treated as the centroid: 0.07637838624359936
# Score when graph 2 is treated as the centroid: 0.07637838624359936
# Score when graph 3 is treated as the centroid: 0.07637838624359936
# The most representative graph in the corpus is graph 1 with a score of 0.07637838624359936 and ablation 1.

# MOZART
# Score when graph 0 is treated as the centroid: 0.2291895321113504
# Score when graph 1 is treated as the centroid: 0.1316846011439699
# Score when graph 2 is treated as the centroid: 0.15914018194453888
# Score when graph 3 is treated as the centroid: 0.1316846011439699
# Score when graph 4 is treated as the centroid: 0.1316846011439699
# The most representative graph in the corpus is graph 1 with a score of 0.1316846011439699 and ablation 1.