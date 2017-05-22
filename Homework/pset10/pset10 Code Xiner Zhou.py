# -*- coding: utf-8 -*-
"""
Pset 10
Xiner Zhou
4/15/2017
"""
# set working dir
import os
os.getcwd()
os.chdir('E:/Course Work at Harvard/Networks/Problem sets/pset10/pset10')   # Change current working directory  
 
import numpy as np
"""
5 a
"""
#load the undirected graph 
def myfunc(filename, direct):
    """The function takes a txt filename and an indicator for whether the network is directed or not,
    and returns a graph."""
    ### read in txt datasets
    edges=[]
    nodes=[]
    with open(filename+'.txt', 'r') as f:
        for i,entry in enumerate(f):
            FromNodeId, ToNodeId=entry.rstrip().split('\t')
            edges.append((int(FromNodeId), int(ToNodeId)))
            nodes.append(int(FromNodeId))
            nodes.append(int(ToNodeId))

    ###  make graph
    import networkx as nx
    if direct==True:
        G=nx.DiGraph()
    else:
        G=nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
 
    del nodes
    del edges
    
    ### plot the network
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,20))
    nx.draw(G,node_size=3,alpha=0.2,edge_color='r' )
    plt.savefig(filename+".png")
    
    return(G)

 

# load network1.txt and make into a directed graph
G = myfunc('data',direct=False)
#how many nodes ? 2992
len(G.nodes())
#how may edges ? 4046
len(G.edges())

"""
5 b
"""
from collections import deque

def BFS(node, graph):
    """
    The function takes a graph and a node in that graph as inputs, and returns a dict of distance between that node and any other nodes in the graph. 
    """
    queue=deque()
    queue.append(node)
    visited={k:False for k in graph.nodes()}# if visited yet
    visited[node]=True
           
    dist={}
    dist[node]=0
        
    while len(queue) !=0: # while we still have things to visit
        current = queue.popleft()  # the current node we want to search for its neighbors
        for i in graph.neighbors(current):
            #if unvisited before, then the distance of this node i =dist(current)+1
            #if visited, do thing
            if visited[i]==False:
                dist[i]=dist[current]+1
                queue.append(i)
                visited[i]=True
    
    return(dist)

def min_pairwise(graph):
    """
    The function takes a graph, calls the function defined outside, and returns a matrix distance of size (n,n) where n is the number of nodes in the graph,
    the matrix is ordred by the order of graph nodes,
    if no path between two nodes, then the distance is set to infinity   
    """
    dist_list_list=[] #initialize holder for list of list of distance
    ordered_nodes=graph.nodes()
    ordered_nodes.sort()
 
    for node in ordered_nodes:
        dist_list=[] #temprary holder for this node
 
        BFS_result=BFS(node, graph)
        for k in ordered_nodes:
            #if there is at least a path, we have the shortest path in BFS_result
            #otherwise, there is no path between node and k, set to infinity
            if k in BFS_result.keys():
                dist_list.append(BFS_result[k])
            else:
                dist_list.append(np.inf)
        dist_list_list.append(dist_list)
    
    dist_matrix=np.matrix(dist_list_list)
    
    # check if dist_matrix is symmetric, if not, report error
    if (dist_matrix==dist_matrix.transpose()).all():
        print("The distance matrix is symmetric!")
    else:
        print("The distance matrix is NOT symmetric, Error!")
        
    # check if there is inf anywhere in the dist_matrix, that is, if the graph is connected
    if np.sum(np.isinf(dist_matrix))==0:
        print("There is a path between any pairs of nodes, the graph is connected!")
    else:
        print("There is no path between at least 1 pair of nodes, the graph is NOT connected!")
      
    return(dist_matrix)
    
# The distance matrix is symmetric!
# There is a path between any pairs of nodes, the graph is connected!
 
# What is the average shortest pairs distance among all pairs?
m=min_pairwise(G)
dist=[]
for i in range(0,len(G.nodes()),1):
    for j in range(i+1,len(G.nodes()),1): # no self distance
        dist.append(m[i,j])
np.mean(dist)    
    
"""
5 c
"""
def dist(p,m,c,norm):
    """
    The function takes a node p, a matrix of shortest distance m, a cluster c, and a string paramter norm with possible values "min", "max", or "mean",
    and returns the approproate distance measure as specified in norm between the point p and the cluster c.
    """
    if norm=="min":
        dist=np.min([m[p,i] for i in c])
    elif norm=="max":
        dist=np.max([m[p,i] for i in c])
    elif norm=="mean":
        dist=np.mean([m[p,i] for i in c])
    else:
        print("Error: Please specify norm with possible values 'min','max',or 'mean'.")
    return(dist)

# What is the distance bw node 5 and the cluster {2,8,20} under each of the three metrics?
# min=2 max=4 mean=3.0
dist(5,min_pairwise(G),[2,8,20],"min")
dist(5,min_pairwise(G),[2,8,20],"max")
dist(5,min_pairwise(G),[2,8,20],"mean")

"""
5 d
"""
def assign(p,m,c_list,norm):
    """
    The function takes takes a node p, a matrix of shortest distances m, a list of clusters c_list, and a string parameter norm ("min","max","mean"),
    it calls the function dist defined before to find the closest cluster to p in terms of the metric supplied in norm, 
    and returns the index of that cluster in the list of cluster c_list.
    """
    pToc=[]
    for c in c_list:
        pToc.append(dist(p,m,c,norm))
    
    return(pToc.index(min(pToc)))
    
#Give node 5 and clusters [[2,8,20], [3,4,8,26]], what does assign return for each of the three metrics?
# min=1 max=0 mean=1
assign(5,min_pairwise(G),[[2,8,20], [3,4,8,26]],"min")
assign(5,min_pairwise(G),[[2,8,20], [3,4,8,26]],"max")
assign(5,min_pairwise(G),[[2,8,20], [3,4,8,26]],"mean")       

    
"""
5 e
"""
def center(m,c):
    """
    The function takes a matrix of shortest distances m and a cluster c, 
    and returns the node that mininize the k-means objective funciton within the cluster c.
    """
    # for this case, the node coincides with its matrix index
    sum_dist={} # node: sum of distance if the node is chosen as centroid
    for node in c:
        sum_dist[node]=np.sum([m[node,k]*m[node,k] for k in c])
 
    return(min(sum_dist, key=sum_dist.get))
    
#Give cluster [2,3,4,8,20,26], what node is the center of the cluster? 3
center(min_pairwise(G),[2,3,4,8,20,26])
    
"""
5 f
"""
def cluster(m,k,norm,i):
    """
    K-means Algorithm:
        Step 1: Randomly select k nodes to initialize the cluster
        
        Step 2: In random order, Assign every other node in the graph to one the clusters, based on entire clusters instead of precalculated single node centers
        Step 3: Re-initialize new clusters at the "center" nodes of each clusters
        
        Ste 4:  Repeat step 2 and 3 as many times as required/desired
        
    The function returns: a list of clusters, a list of centers, objective function after i iteraction, and the size of each cluster.
    """
    #step 1
    center_list=[]
    for j in range(0,k,1):
        center_list.append(int(np.random.choice(G.nodes(),size=1)))
    # assign centers themselfves to clusters
    c_list=[[item] for item in center_list]
    print("k=",k)
    print("norm=",norm)
    print("Iteration=",i)
    # iteract i times
    for iter in range(0,i,1):   
        print(iter)
        # step 2: Assign clusters
        other_nodes=np.random.permutation([item for item in G.nodes() if item not in center_list])
        for node in other_nodes:
            c_list[assign(node,m,c_list,norm)].append(node)
        # step 3: Update centers
        for j in range(0,k,1):
            center_list[j]=center(m,c_list[j])   
    
    # calculate the objective function on the updated center_list after i iteration
    objective_list=[]
    for j in range(0,k,1):
        objective_list.append(np.sum([m[node,center_list[j]]*m[node,center_list[j]] for node in c_list[j]]))
    objective=np.sum(objective_list) 
    
    # size of each cluster
    c_size=[len(c) for c in c_list]
    
    return((c_list,center_list,objective,c_size))

m=min_pairwise(G)
for k in [3,5,10,20]:
    for norm in ["min","max","mean"]:
        for run in range(0,3,1):
            c_list, center_list, objective, c_size=cluster(m,k,norm,20)
            print("For k=",k,' norm=',norm,' run=',run,'\n')
            print("center list=", center_list,'\n')
            print("objective=", objective,'\n')
            print("size of each cluster=", c_size,'\n')
         
    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                
    
  