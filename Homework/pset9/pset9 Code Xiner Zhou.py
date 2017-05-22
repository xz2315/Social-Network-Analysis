 -*- coding: utf-8 -*-
"""
Pset 9
Xiner Zhou
4/10/2017
"""
# set working dir
import os
os.getcwd()
os.chdir('/Volumes/NO NAME/Course Work at Harvard/Networks/Problem sets/pset9')   # Change current working directory  

import numpy as np 
from numpy import linalg as LA
import matplotlib.pyplot as plt

"""
3.a
"""
M=np.array([[0,1/2,1/2,0,0,0,0,0],
            [0,0,0,1/2,1/2,0,0,0],
            [0,0,0,0,0,1/2,1/2,0],
            [1/2,0,0,0,0,0,0,1/2],
            [1/2,0,0,0,0,0,0,1/2],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,1,0,0],
            [1,0,0,0,0,0,0,0]])
w,v = LA.eig(M.transpose())
w
# eigenvector cooresponding to eigenvalue 1
v[:,3]/np.sum(v[:,3])
# t=1000
np.dot(LA.matrix_power(M.transpose(), 1000), np.array([1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8]))



"""
5. Implementing PageRank
"""

### b. 

# load google.txt
 
def myfunc(filename, direct):
    """The function takes a txt filename and an indicator for whether the network is directed or not,
    and returns a graph."""
    ### read in txt datasets
    edges=[]
    nodes=[]
    with open(filename+'.txt', 'r') as f:
        for i,entry in enumerate(f):
            FromNodeId, ToNodeId=entry.rstrip().split(' ')
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
    #import matplotlib.pyplot as plt
    #nx.draw(G,node_size=3,alpha=0.2,edge_color='r' )
    #plt.savefig(filename+".png")
    
    return(G)

 

# load network1.txt and make into a directed graph
google_G = myfunc('google',direct=True)
#how many nodes are there?387597
len(google_G.nodes())
#What is the average out-degree? 1.3700131837965721
np.mean(list(google_G.out_degree().values()))



### c. 

def pageRankIter(g,d):
    """The function takes a graph g and dictionary d specifying the current PR score of all the nodes in g; 
    and returns a new dictionary d_new giving the PR score of all nodes after applying one round of the basic PR updates. 
    Additionally, the function also supply a historgrams of the PR scores both before and after one iteration of pageRankIter."""
    # sort nodes and make a list
    nodes=list(g.nodes())
    nodes.sort()
    
    # transition matrix in the order of the sorted list
    M=[]
    for i in nodes:
        row_i=[] # row[i] of the transition matrix, which represents connectivity of node i
        if len(g.neighbors(i))>0:
            print("Has out edges!")
            for j in nodes:
                if j in g.neighbors(i):
                    row_i.append(1/len(g.neighbors(i)))
                else:
                    row_i.append(0)
            
        elif len(g.neighbors(i))==0:
            print("No out edges!")
            for j in nodes:
                if i==j:
                    row_i.append(1)
                else:
                    row_i.append(0)
            
        M.append(row_i)
        
        
    # make a list r_pre for d, i.e., current PR
    r_pre=[]
    for node in nodes:
        r_pre.append(d[node])
    
    # one iteration of basic PR update r_post=M^T r_pre
    M=np.array(M)
    r_pre=np.array(r_pre)
    r_post=np.dot(M.transpose(), r_pre)

    # make a dictionary d_new, i.,e., PR after one iteration
    d_new={}
    for index, node in enumerate(nodes):
        d_new[node]=r_post[index]

    # histogram of r_pre and r_post
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,20))
    plt.hist(r_pre, bins=300, color="red")
    plt.title("PR scores Pre")
    plt.xlabel("PR scores")
    plt.ylabel("Frequency")
    plt.savefig("PR score pre.png")
    
    plt.figure(figsize=(20,20))
    plt.hist(r_post, bins=300, color="red")
    plt.title("PR scores Post")
    plt.xlabel("PR scores")
    plt.ylabel("Frequency")
    plt.savefig("PR score post.png")
    
    # return d_new
    return(d_new)

# randomly sample 10,000 nodes
from numpy import random
sample_nodes=np.random.choice(google_G.nodes(), size=10000, replace=False)
remove_nodes=[n for n in google_G.nodes() if n not in sample_nodes]
google_G.remove_nodes_from(remove_nodes)

# Assign all nodes initally the same PageRank value, 1/n
d_0={}
for node in google_G.nodes():
    d_0[node]=1/len(google_G.nodes())
# One iteration
d_1=pageRankIter(google_G, d_0)   

 
 

#### d. Basic PageRank Algorithm
def basicPR(g,d,k):
    """The function takes a graph g, an initialy PR values d, and number of iterations k, call previously 
    defined funciton pageRankIter, and returns the PR values after k iteractions, in a dictionary."""
    for i in range(0,k,1):
        if i==0:
            d_post=pageRankIter(g, d)
        else:
            d_pre=d_post
            d_post=pageRankIter(g, d_pre)
            
    return(d_post)

    
d_10=basicPR(google_G,d_0,10)    
d_50=basicPR(google_G,d_10,40)   
d_200=basicPR(google_G,d_50,150)      
    
#### e. Scaled PageRank Algorithm
def ScaledpageRankIter(g,d,s):
    """The function takes a graph g and dictionary d specifying the current PR score of all the nodes in g, and a scaling factor s; 
    and returns a new dictionary d_new giving the PR score of all nodes after applying one round of the basic PR updates. 
    Additionally, the function also supply a historgrams of the PR scores both before and after one iteration of pageRankIter."""
    # sort nodes and make a list
    nodes=list(g.nodes())
    nodes.sort()
    
    # transition matrix in the order of the sorted list
    M=[]
    for i in nodes:
        row_i=[] # row[i] of the transition matrix, which represents connectivity of node i
        if len(g.neighbors(i))>0:
            print("Has out edges!")
            for j in nodes:
                if j in g.neighbors(i):
                    row_i.append(1/len(g.neighbors(i)))
                else:
                    row_i.append(0)
            
        elif len(g.neighbors(i))==0:
            print("No out edges!")
            for j in nodes:
                if i==j:
                    row_i.append(1)
                else:
                    row_i.append(0)
            
        M.append(row_i)
    M=np.multiply(s,M)+(1-s)/len(nodes)  
        
    # make a list r_pre for d, i.e., current PR
    r_pre=[]
    for node in nodes:
        r_pre.append(d[node])
    
    # one iteration of basic PR update r_post=M^T r_pre
    M=np.array(M)
    r_pre=np.array(r_pre)
    r_post=np.dot(M.transpose(), r_pre)

    # make a dictionary d_new, i.,e., PR after one iteration
    d_new={}
    for index, node in enumerate(nodes):
        d_new[node]=r_post[index]

    # histogram of r_pre and r_post
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,20))
    plt.hist(r_pre, bins=300, color="red")
    plt.title("PR scores Pre")
    plt.xlabel("PR scores")
    plt.ylabel("Frequency")
    plt.savefig("PR score pre scaled.png")
    
    plt.figure(figsize=(20,20))
    plt.hist(r_post, bins=300, color="red")
    plt.title("PR scores Post")
    plt.xlabel("PR scores")
    plt.ylabel("Frequency")
    plt.savefig("PR score post scaled.png")
    
    # return d_new
    return(d_new)

ScaledpageRankIter(google_G, d_0, 0.8)   

    
def scaledPR(g, d, k, s):
    """The function takes a graph g, an initialy PR values d, number of iterations k, and a scaling factor s,
    call previously defined funciton ScaledpageRankIter, and returns the PR values after k iteractions, in a dictionary."""
    for i in range(0,k,1):
        if i==0:
            d_post=ScaledpageRankIter(g, d, s)
        else:
            d_pre=d_post
            d_post=ScaledpageRankIter(g, d_pre, s)
            
    return(d_post)

    
d_10=scaledPR(google_G,d_0,10, 0.8)    
d_50=scaledPR(google_G,d_10,40, 0.8)   
d_200=scaledPR(google_G,d_50,150, 0.8)   


#### f.
links=[] # placeholder for the links.txt
with open('links.txt', 'r') as f:
    for i,entry in enumerate(f):
            node, link=entry.rstrip().split(' ')
            links.append([int(node), link])  

node_for_34=[item[0] for item in links if item[1].find('34')>=0] 
not_node_for_34=[item for item in google_G.nodes() if item not in node_for_34] 
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
