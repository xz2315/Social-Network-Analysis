# -*- coding: utf-8 -*-
"""
Pset 6
Xiner Zhou
3/14/2017
"""
# set working dir
import os
os.getcwd()
os.chdir('E:/Course Work at Harvard/Networks/Problem sets/pset6/pset6')   # Change current working directory  
 
"""
3 d.
"""
import numpy as np
from numpy import linalg as LA

P1=np.matrix([[0.1, 0.4, 0,   0.3, 0.2],
             [0,   0,   0.5, 0.5, 0],
             [0.6, 0,   0.2, 0.1, 0.1],
             [0.2, 0.2, 0.2, 0.2, 0.2],
             [0.3, 0.3, 0.1, 0.1, 0.2]], dtype=float)

P2=np.matrix([[0.2, 0.2, 0.2, 0.2, 0.2],
             [0.1, 0.3, 0.4, 0,   0.2],
             [0.3, 0.1, 0.1, 0.5, 0],
             [0,   0,   0.5, 0.3, 0.2],
             [0.2, 0.1, 0.3, 0.2, 0.2]], dtype=float)

P3=np.matrix([[0.1, 0.3, 0.4, 0.2, 0],
             [0,   0,   0.5, 0.4, 0.1],
             [0.2, 0.2, 0.3, 0.2, 0.1],
             [0,   0.3, 0.5, 0.1, 0.1],
             [0.6, 0.2, 0.1, 0,   0.1]], dtype=float)

# Raise a square matrix to the (integer) power `n`.
print("For P1, L= \n")
np.round(LA.matrix_power(P1, 100),3)
print("LP= \n")
np.round(np.dot(LA.matrix_power(P1, 100),P1),3)

print("For P2, L= \n")
np.round(LA.matrix_power(P2, 100),3)
print("LP= \n")
np.round(np.dot(LA.matrix_power(P2, 100),P2),3)

print("For P3, L= \n")
np.round(LA.matrix_power(P3, 100),3)
print("LP= \n")
np.round(np.dot(LA.matrix_power(P3, 100),P3),3)


"""
5. Learning Influence Locally
"""

### a. Load opinions.pk
import pickle 
pkl_file = open('opinions.pk', 'rb')
opinion = pickle.load(pkl_file)
# opinion is a list, each entry represents a cascade and each entry is stored as a dictionary
# within each entry, the keys are node n, values are the node's opinion 0/1 which are timestamped

# opinion_i[n] has length tao? All of them have length 31
tao=list()
for c in opinion:
    for n in c:
        tao.append(len(c[n]))
 
# how many cascades are there?
len(opinion)

# At what timestep (using 0-indexing) does node 13 first have its opinion activated in cascade 6?
def ActiveTime(c, n):
    """The function returns the timestep(using 0-indexing) when node n first have its opinion activated in cascade c. """
    # time0 is a placeholder for the initial timestep we will return
    # default value set to the length of timesteps, which is tao, if the node never activated
    time0=len(opinion[c][n])
    # once activated, store the index (0-indexing) to time0, and break the loop
    for index, val in enumerate(opinion[c][n]):
        if val==1:
            time0=index
            break
    return(time0)
 
ActiveTime(6, 13)
 
### b. load network1.txt


#### Begin the function
def myfunc(filename, direct):
    ### read in txt datasets
    edges=[]
    nodes=[]
    with open(filename+'.txt', 'r') as f:
        for i,entry in enumerate(f):
            FromNodeId, ToNodeId=entry.rstrip().split(' ')
            edges.append((FromNodeId,ToNodeId))
            nodes.append(FromNodeId)
            nodes.append(ToNodeId)

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
network1_G = myfunc('network1',direct=True)
#how many nodes are there?
len(network1_G.nodes())
#What is the average out-degree?
np.mean(list(network1_G.out_degree().values()))

### c. 
def doesInfluence(n, m, t, o):
    """The function takes nodes n and m, time t, and cascade dictionary o as inputs; 
    return 1 if n is activated at time t and m is activated at time t+1 and not activated at time t, 
    and return 0 otherwise (or t+1>=\tao) """
    
    # tao is the time length of node n
    tao=len(o[n])
    
    def ActiveTime2(node):
        """The function returns the timestep(using 0-indexing) when node first have its opinion activated in cascade. """
        # time0 is a placeholder for the initial timestep we will return
        # default value set to the length of timesteps, which is tao, if the node never activated
        time0=len(o[node])
        # once activated, store the index (0-indexing) to time0, and break the loop
        for index, val in enumerate(o[node]):
            if val==1:
                time0=index
                break
        return(time0)
 
    # This is the main body of the function doesInfluence
    if ActiveTime2(n)==t and ActiveTime2(m)==t+1 and t+1<tao:
        influence=1
    else:
        influence=0
        
    return(influence)
        
doesInfluence(10, 4, 1, opinion[0])
 

### d. MLE estimates of influence
def influenceMLE(G, cascades):
    """Then function takes a graph G and a list of cascades, and returns a numpy 2-D array with outgoing node, incoming node,
    and the MLE estimated weight."""
    
    # initialize empty list for [n, m, MLE]
    value_triple=[]
    def ActiveTime3(cascade, n):
        """The function returns the timestep(using 0-indexing) when node n first have its opinion activated in cascade c. """
        # time0 is a placeholder for the initial timestep we will return
        # default value set to the length of timesteps, which is tao, if the node never activated
        time0=len(cascade[n])
        # once activated, store the index (0-indexing) to time0, and break the loop
        for index, val in enumerate(cascade[n]):
            if val==1:
                time0=index
                break
        return(time0)
    
    for n in G.nodes():
        for m in G.neighbors(n):
            potential=list() # denominator
            success=list() # numerator  
            for o in cascades:          
                tao=len(o[int(n)])
                # if n is activated before the last timestep 
                if ActiveTime3(o, int(n))<tao:
                    # denominator add 1
                    potential.append(1)
                    # wether n influenced m at t+1
                    t=ActiveTime3(o, int(n))
                    success.append(doesInfluence(int(n), int(m), t, o))
                    # debug check
                    #print("n activated at time "+str(ActiveTime3(o, int(n))))
                    #print("M activated at time "+str(ActiveTime3(o, int(m))))
                    #print("n influences m? "+str(doesInfluence(int(n), int(m), t, o)))
                else:
                    potential.append(0)
                    success.append(0)

            value_triple.append([int(n), int(m), np.sum(success)/np.sum(potential)])
    
    #print(value_triple)
    return(value_triple)
                 
MLE=influenceMLE(network1_G, opinion)

# What's the estimated weight of edge (1,2)?
[i for i in MLE if i[0]==1 and i[1]==2]
# What's the estimated weight of edge (26,21)?
[i for i in MLE if i[0]==26 and i[1]==21]

# save estimates as a separete file formatted as a .csv
# where entries in the first column is the first node in the edge, second column is the second node in the edge
# and third column is the MLE estimate of edge weight defined as "influence"
import pandas as pd
MLE_df = pd.DataFrame(MLE)
MLE_df = MLE_df.sort([0,1])
MLE_df.columns=["first node", "second node", "weight"]
MLE_df.to_csv('MLE.csv', index=False, header=True)

### e. 

# what node has the highest average edge weight (for outgoing edges)?
# what node has the lowest average edge weight for outgoing edges?
ave_weight=[]
node=[]
for n in network1_G.nodes():
    node.append(int(n))
    ave_weight.append(np.mean([i[2] for i in MLE if i[0]==int(n)]))
 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.plot(node, ave_weight, 'ro', markersize=15)
plt.xticks(range(0,30,1))
plt.ylabel("Average MLE estimates of weight or influence")
plt.xlabel("node")
plt.title("Average Influence")
plt.savefig("Influence.png")

 
# what node, on average, is activated first (in cascades where it activates)? 
# What node, on average, is activated last?
# only nodes not activated initially!!!
ave_ActivateTime=[]
node=[]
for n in network1_G.nodes():
    node.append(int(n))
    temp=[] # temporary holder for fist activ time 
    for c in range(0,len(opinion),1):
        temp.append(ActiveTime(c, int(n)))
        
    ave_ActivateTime.append(np.mean([i for i in temp if i<31 and i>0]))

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.plot(node, ave_ActivateTime, 'ro', markersize=15)
plt.xticks(range(0,30,1))
plt.ylabel("Average Activated Time")
plt.xlabel("node")
plt.title("Average Activated Time")
plt.savefig("ActivatedTime.png")

 
print(np.min(ave_ActivateTime))









    