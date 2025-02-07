import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from scipy.spatial.distance import cdist
from scipy import optimize
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_graph(edge_list, weighted=False, directed=False):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
      # Construct the directed graph from the edge list, using weights
      G = nx.DiGraph()
      G.add_weighted_edges_from(edge_list)  # Adds edges with weights directly
    else:
      # Construct the directed graph without weights, assuming default weight of 1
      G = nx.DiGraph()
      for edge in edge_list:
          G.add_edge(edge[0], edge[1], weight=1)  # Default weight of 1 for unweighted edges
    #G = nx.read_edgelist(file_name, nodetype=int)
    if not directed:
        G = G.to_undirected()
    return G
    
def fitting_func(dims,s,a,L):  
  return s/np.power(dims,a) + L
  
def fitting_func_2var(dims, s, a):
    return s / np.power(dims, a)

def identify_optimal_dim(embedding_dims, loss):
    '''
    Identify the optimal dimension range and compute the curve fitting parameter for graph.
    '''  
    if len(embedding_dims) == 2:
      (s,a),cov = optimize.curve_fit(fitting_func_2var, embedding_dims,loss)
      fit_values = (fitting_func_2var(np.array(embedding_dims),s,a))
    else:
      (s,a,l),cov = optimize.curve_fit(fitting_func, embedding_dims,loss)
      fit_values = (fitting_func(np.array(embedding_dims),s,a,l))
    MSE = ((np.array(loss)-np.array(fit_values))**2).mean()
    opt = np.power((s/0.05),1/a)
    print('the optimal dimension at 0.05 accuracy level is {}'.format(int(math.ceil(opt))))
    print('the MSE of curve fitting is {}'.format(MSE))
    return int(math.ceil(opt)), MSE

def cal_cosine_matrices(G, walks, start_dim=2, end_dim=100, step=4, window_size=5, workers=8, iter=10):
    '''
    Compute the cosine distance between every node pair over different embedding dimensions.
    '''
    norm_loss = []
    walks = [list(map(str, walk)) for walk in walks]
    node_num = len(G.nodes())
    if node_num < end_dim:
      end_dim = node_num 
    embedding_dims = list(range(start_dim, end_dim, step))
    if node_num < 500:
      embedding_dims.insert(0,node_num)
      print('graph size smaller than the default end dimension, thus has been automatically set to {}'.format(node_num))
    else:
      embedding_dims.insert(0,500)  
    #cosine_matrices = np.zeros((len(embedding_dims),node_num,node_num)) 
    for _index, dim in enumerate(embedding_dims):
      #print((dim))
      model = Word2Vec(walks, vector_size=dim,window=window_size, min_count=0, sg=1, workers=workers, epochs=iter)    
      emb_matrix = np.zeros((node_num,dim))      
      for _cnt,node in enumerate(G.nodes()):
        emb_matrix[_cnt,:] = model.wv[str(node)] 
      emb_matrix = emb_matrix - np.mean(emb_matrix,axis=0) 
      cosine_matrix = 1 - cdist(emb_matrix,emb_matrix,'cosine')
      if _index == 0:
        benchmark_array = np.array(upper_tri_masking(cosine_matrix))
        #np.savez_compressed('./pic/conect_data/npz/{}'.format(str.split(args.input,'/')[6]),benchmark_array)      
      else:
        dim_array = np.array(upper_tri_masking(cosine_matrix)) 
        loss = np.linalg.norm((dim_array-benchmark_array),ord=1)
        norm_loss.append(loss/len(dim_array))
    return embedding_dims[1:],norm_loss
    
def upper_tri_masking(A):
    '''
    Masking the upper triangular matrix. 
    '''
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]
  
def cal_embedding_distance(edge_list, weighted=False, directed=False, p=1, q=1, num_walks=20, length=10, start_dim=2, end_dim=100, step=4, window_size=5, workers=8, iter=10):
    '''
    The overall random walk, graph embedding and cosine distance calculation process.
    '''
    nx_G = read_graph(edge_list, weighted, directed)
    G = node2vec.Graph(nx_G, directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, length)
    dims, loss = cal_cosine_matrices(nx_G,walks,start_dim,end_dim,step,window_size,workers,iter)
    plt.plot(dims,loss)
    plt.savefig('./a.png')
    opt_dim, mse_loss = identify_optimal_dim(dims, loss)
    return opt_dim, mse_loss


