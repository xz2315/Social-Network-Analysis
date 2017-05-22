 -*- coding: utf-8 -*-
"""
Pset 8
Xiner Zhou
4/1/2017
"""
# set working dir
import os
os.getcwd()
os.chdir('E:\\Course Work at Harvard\\Networks\\Problem sets\\pset8\\pset8')   # Change current working directory  
 
"""
2 b c
"""
import numpy as np
from numpy import linalg as LA
M=np.array([[1/5,1/5,1/5,1/5,1/5],
            [1/3,1/3,1/3,0,0],
            [1/4,1/4,1/4,1/4,0],
            [1/4,0,1/4,1/4,1/4],
            [1/3,0,0,1/3,1/3]])
a=np.array([1,0,0,0,0])
e=np.array([0,0,0,0,1])
np.dot(np.dot(a.transpose(), LA.matrix_power(M,2)),e) 
"""
3 c
"""
M=np.matrix([[0.4, 0.6, 0],
             [0,   0,   1],
             [0.3, 0,   0.7]], dtype=float)

p0=np.matrix([[1],
             [0.7],
             [0]], dtype=float)

 
p1=np.dot(M, p0)
p2=np.dot(M, p1)

"""
3 d
"""
#limiting opinion
np.dot(LA.matrix_power(M, 1000), p0)

"""
3 f
"""
# ring if n=4
M=np.array([[1/2,1/2,0,0],
            [0,1/2,1/2,0],
            [0,0,1/2,1/2],
            [1/2,0,0,1/2]])

LA.matrix_power(M, 1000)




"""
5. Learning Influence Locally
"""

### a. load the file pset8_opinions.pk
import pickle 
pkl_file = open('pset8_opinions.pkl', 'rb')
opinion = pickle.load(pkl_file)
type(opinion)
opinion[0]
# opinion is a list, with each entry represents a cascade process stored as a dict, 
# and within the cascade, the key is node, values are the node's opinion 0/1 which are timestamped
 
# What is the length of cascade ?
tao=list()
for cas in opinion:
    for node in cas:
        tao.append(len(cas[node]))
 
# tau=35 

# how many different diffusion processes are there?   
len(opinion)
 
### b. load the file pset8_network1.txt
#### Begin the function
def myfunc(filename, direct):
    ### read in txt datasets
    edges=[]
    nodes=[]
    with open(filename+'.txt', 'r') as f:
        for i,entry in enumerate(f):
            FromNodeId, ToNodeId=entry.rstrip().split(' ')
            edges.append((int(FromNodeId), int(ToNodeId)))
            edges.append((int(FromNodeId), int(FromNodeId))) # add self loop
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
    nx.draw(G,node_size=3,alpha=0.2,edge_color='r' )
    plt.savefig(filename+".png")
    
    return(G)

#### End of the function

# load network1.txt and make a graph
network_G = myfunc('pset8_network1',direct=True)
#how many nodes are there?
len(network_G.nodes())
#What is the average out-degree?
np.mean(list(network_G.out_degree().values()))


### d.
def EdgeWt(graph, opinion, node):
    """This function takes three inputs: a graph representing a network, a list opinion representing tau diffusions/cascades,
    and a node which we want to estimate edge weights (out edges); and returns the all weightes associated with that node."""
    from scipy.optimize import linprog
    # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    
    # find out node's neighbors
    nb=graph.neighbors(node)
    """
    Minimize: c^T * x

    Subject to: A_ub * x <= b_ub
                A_eq * x == b_eq
    """
    # Add self-loop, the first element in the following array is for self-loop
    c=[-1]*len(nb) # Minimize the negative of f = Maximiza f
    A_ub=[[1]*len(nb)] # upper-bound inequality constraints
    b_ub=[1] # upper-bound inequality constraints
    #A_eq=[[1]*len(nb)] # equality constraints
    #b_eq=[1] # equality constraints
    bounds=[(0,1) for i in range(0,len(nb))]
    # loop through diffusions
    for d in opinion:
        # loop through time steps, except the initial step
        for t in range(1,len(d[node]),1):
            temp=[] # holder for neighbors' opinion at prior step
            if d[node][t]==0:
                b_ub.append(0.5) 
                for i in nb:
                    temp.append(d[i][t-1])

            elif d[node][t]==1:
                b_ub.append(-0.5) 
                for i in nb:
                    temp.append(-d[i][t-1])
            A_ub.append(temp) 
 
    # outside the loop, we have A, b, c, now optimize 
    fit=linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, options={"disp": True, "maxiter":100000})
    return(nb, fit)
     
 
# What’s the estimated weight of edge (2; 16)? 

neighbors,weights=EdgeWt(network_G, opinion, 2) 
# weights for neighbors
print(neighbors)
print(weights['x'])
# check if weights sum up to 1
print(-weights['fun'])

# Of (29; 22)? 
neighbors,weights=EdgeWt(network_G, opinion, 29)
# weights for neighbors
print(neighbors)
print(weights['x'])
# check if weights sum up to 1
print(-weights['fun'])

# save estimates as a separete file formatted as a .csv

# initialize empty list for [FromNode, ToNode, weight]
value_triple=[]
for node in network_G.nodes():
    neighbors,weights=EdgeWt(network_G, opinion, node) 
    for i in range(0,len(neighbors),1):
        value_triple.append([node, neighbors[i], weights['x'][i]])
 
import pandas as pd
df = pd.DataFrame(value_triple)
df = df.sort([0,1])
df.columns=["from node", "to node", "weight"]
df.to_csv('weight.csv', index=False, header=True)         
        
 
### e.
"""
Assuming popularity is the average of influence, what five students (nodes) should
Raynor first invite to the app and what’s their average influence? You should use edge weights
as measures of influence, but consider the Voter model and be careful about the direction of your
edges!
"""
# average weight of in-degree as a measure of influence
influence={}
for node in network_G.nodes():
    influence[node]=np.mean([item[2]  for item in value_triple if item[1]==node])

for i in sorted(influence, key=influence.get, reverse=True):
    print((i,influence[i]))
# 15,12,31,11,29 

# sum weight of in-degree as influecne
influence={}
for node in network_G.nodes():
    influence[node]=np.sum([item[2]  for item in value_triple if item[1]==node])

for i in sorted(influence, key=influence.get, reverse=True):
    print((i,influence[i]))
























