import build_graph
import networkx as nx
import numpy as np
from collections import Counter
import MCMC_helpers

# Stoke uses beta=1 so I'm doing that for now: https://github.com/StanfordPL/stoke/blob/98d8a0f028f2daf2052bfe607dbc32ec8d55ba9e/tools/args/search.inc#L29
beta = 1 # temperature/annealing param 

# R: rewrite (i.e. the canonical graph proposal)
# T: target (i.e. corpus of graphs)
def cost(R, T, need_new_proposal_dist, proposal_dist):
	# at every step where we do the cost function, we want to update the transorm stats for every transform possible
	# (not just transform types). specifically, like insert node X. from this, these form some kind of distribution,
	# update based on the counts of each transform / total counts
	# this depends on how labels are consistent across the different graphs
	# sample from this dist of transforms given these probabilities I have constructed
	# current rewrite R. corpus has 2 other graphs. we compute the edit paths from R to G1 and G2.
	# path P1 has transforms A,B,C. path P2 has transforms A,D,E. so prob(A) = 2/6, but 1/6 for the rest
	# all the operations that don't appear in the paths, start with prob zero and then apply smoothing
	# optimize_graph_edit_distance is about 7.5 seconds per graph based on experiment
	total_approx_edit_dist = 0
	transform_counts = Counter()
	
	for g in T:
		# edge substitution/relabeling is redudant once the nodes are relabeled so we keep at cost 0
		node_edit_path, edge_edit_path, approx_edit_dist = next(nx.optimize_edit_paths(R, g, node_subst_cost=MCMC_helpers.node_subst_cost, edge_match=MCMC_helpers.edge_match)) 
		if need_new_proposal_dist: # i.e. we want a new distribution because we just accepted/added a new transform in the prev step, and thus R has changed
			transform_counts += MCMC_helpers.build_transform_counts(node_edit_path + edge_edit_path)
		total_approx_edit_dist += approx_edit_dist
	
	proposal_dist = MCMC_helpers.additive_smooth(transform_counts) if need_new_proposal_dist else proposal_dist
	return (total_approx_edit_dist, proposal_dist)

# T: target (i.e. corpus of graphs)
# R: rewrite (i.e. the canonical graph proposal)
# https://theory.stanford.edu/~aiken/publications/papers/cacm16.pdf eqn 3 page 116
# Convert discrete cost function to probability distribution
def p(R, T, prev_t_was_accepted, proposal_dist):
	"""
	technically, should be 1/Z * exp(-beta*cost(R, T))
	Although computing Z is generally intractable, the
	Metropolis–Hastings algorithm is designed to explore
	density functions such as p(·) without having to compute
	Z directly (i.e. it gets divided by itself to 1). 
	Thus, we omit it.
	"""
	t_cost, new_proposal_dist = cost(R, T, prev_t_was_accepted, proposal_dist)
	return (np.exp(-beta * t_cost), new_proposal_dist) 

def q(proposal_dist, t):
	print(proposal_dist)
	return proposal_dist[t]

# R_curr: current rewrite (i.e. R)
# R_new: proposed rewrite (i.e. R*)
# T: target (i.e. corpus of graphs)
# t: the proposal transform that creates R_new from R_curr
# https://theory.stanford.edu/~aiken/publications/papers/cacm16.pdf eqn 4 page 116
# Metropolis Hastings local acceptance probability 
def local_accept_prob(R_curr, T, t, need_new_proposal_dist, proposal_dist):
	t_inv = MCMC_helpers.get_transform_inverse(t)
	p_new, new_proposal_dist = p(R_curr, T, need_new_proposal_dist, proposal_dist)
	p_curr, _ = p(R_curr, T, False, proposal_dist)
	accept_prob = min(1, (p_new * q(proposal_dist, t_inv))/(p_curr * q(proposal_dist, t)))
	return (accept_prob, new_proposal_dist)

	# assuming q i.e. proposal distribution is symmetric, we can do this instead (simplified from original equation, see stanford paper)
	# curr_cost, _ = cost(R_curr, T, need_new_proposal_dist, proposal_dist)
	# new_cost, new_proposal_dist = cost(R_new, T, need_new_proposal_dist, proposal_dist)
	# return (np.exp(-beta*(new_cost-curr_cost)), new_proposal_dist)

def metropolis_hastings_step(R_curr, T, need_new_proposal_dist, proposal_dist):
	transform_is_ok = False
	while not transform_is_ok:
		t = MCMC_helpers.generate_proposal(proposal_dist)
		transform_is_ok = not MCMC_helpers.is_invalid_proposal_application(R_curr, t)

	R_new = MCMC_helpers.apply_transform(R_curr, t)
	(accept_prob, new_proposal_dist) = local_accept_prob(R_curr, T, t, need_new_proposal_dist, proposal_dist)
	u = np.random.uniform()
	if u <= accept_prob:
		R = R_new
		accepted = True
		proposal_dist = new_proposal_dist
	else:
		R = R_curr
		accepted = False
	return {'rewrite': R, 'accepted': accepted, 'proposal_dist': proposal_dist}

def run_metropolis_hastings(initial_graph, initial_proposal_dist, target_corpus, n=10, burnin=0, lag=1):
	centroid_graph = initial_graph
	results = [initial_graph]
	need_new_proposal_dist = False # Since we already have an initial proposal dist
	proposal_dist = initial_proposal_dist

	# Burn-in period
	for _ in range(burnin):
		step_result = metropolis_hastings_step(centroid_graph, target_corpus, need_new_proposal_dist, proposal_dist)
		centroid_graph = step_result['rewrite']
		need_new_proposal_dist = step_result['accepted']
		proposal_dist = step_result['proposal_dist']
	# Sampling period
	for _ in range(n):
		for i in range(lag):
			print(i)
			step_result = metropolis_hastings_step(centroid_graph, target_corpus, need_new_proposal_dist, proposal_dist)
			centroid_graph = step_result['rewrite']
			results.append(centroid_graph)
			need_new_proposal_dist = step_result['accepted']
			proposal_dist = step_result['proposal_dist']
	
	return (centroid_graph, results)


(G0, layers0, label_dict0) = build_graph.generate_graph('LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short_segments.txt', 'LOP_database_06_09_17/liszt_classical_archives/0_short_test/bl11_solo_short_motives.txt')
(G1, layers1, label_dict1) = build_graph.generate_graph('LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short_segments.txt', 'LOP_database_06_09_17/liszt_classical_archives/1_short_test/beet_3_2_solo_short_motives.txt')
(_, initial_proposal_dist) = cost(G0, [G0, G1], True, {})
G_centroid, results = run_metropolis_hastings(G0, initial_proposal_dist, [G0, G1])
layers_centroid = build_graph.get_layers_from_graph(G_centroid)
build_graph.visualize_with_index([G0], [layers_centroid])