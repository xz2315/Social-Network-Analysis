# -*- coding: utf-8 -*-
"""
Pset 7
Xiner Zhou
3/28/2017
"""
# set working dir
import os
os.getcwd()
os.chdir('E:/Course Work at Harvard/Networks/Problem sets/pset7/pset7')   # Change current working directory  
 
### a.
# read in network.txt, make a directed weighted graph
import networkx as nx
G=nx.DiGraph()
with open('network.txt', 'r') as f:
    for i, entry in enumerate(f):
        FromNodeId, ToNodeId, wt=entry.rstrip().split('\t')
        G.add_edge(int(FromNodeId), int(ToNodeId), weight=float(wt))
        G.add_node(int(FromNodeId))
        G.add_node(int(ToNodeId))

# What is the probability of node 42 influencing node 75?
print("What is the probability of node 42 influencing node 75? \n",[e for e in G.edges(data=True) if e[0]==42 and e[1]==75])    


### b. Write a funciton genRandGraph(weights) that takes a graph weights with weighted, directed edges, and returns a randomly realized graph
import numpy as np
 
def genRandGraph(graph):
    """ The newG random graph only has realized edges and nodes that have at least 1 edges remained in the sampled graph."""
    newG=nx.DiGraph()
    for e in graph.edges(data=True):
        if np.random.binomial(n=1,p=e[2]['weight'], size=1)==1:
            newG.add_edge(e[0],e[1])
            newG.add_node(e[0])
            newG.add_node(e[1])
    return(newG)   
     
 
# Run genRandGraph 100 times. What is the average number of nodes in the randomly realized graph?
np.random.seed(1989)
num_nodes=[]
for i in range(0,100,1):
    num_nodes.append(len(genRandGraph(G).nodes()))
print("Run genRandGraph 100 times. What is the average number of nodes in the randomly realized graph? \n",np.mean(num_nodes))

 
### c. Write a function sampleInfluence(G,S,m) takes a weighted graph G, a subset of nodes S, a sample limit m, 
###    and returns the influence f(S) of S according to the SampleInfluence algorithm.

"""
BFS:
a graph traversal is a route of nodes to follow or traverse, where we can only move
between two nodes if there is an edge between them. Breadth first search specifies a process for
doing so. In the simplest implementation, BFS visits a given node, adds its neighbors to the queue,
then visits the element returned by the queue. To begin the BFS, we start at a given node, then visit
all of that node’s nieghbors, then visit all of one neighbor’s neighbors, then another neighbor’s
neighbors, etc. Whenever we visit a node for the first time, we mark it as visited; otherwise we
do nothing when re-visiting a node. The algorithm terminates once we have visited everything
reachable from the given node, and we have a list of nodes reachable from that node.
"""
from collections import deque

# define BFS that takes a graph and starting node v, and returns a dict indicating reachable or not for all nodes
def BFS(graph, v): 
    """ takes a graph and starting node v, and returns a dict indicating reachable or not for all nodes"""
    nodes=graph.nodes() # list of nodes
    #make a dict, (node: visited?)
    reachable ={}
    for i in nodes:
        reachable[i] = -1 # -1 indicates unvisited
    reachable[v] = 1 # start from v node
    queue = deque()
    queue.append(v)
    while len(queue) != 0: # while we still have things to visit
        current = queue.popleft()  
        # look for which nodes are reachable from current node
        for node in graph.neighbors(current):
                #if un-visited before, then add the node to the queue, otherwise do nothing
                if reachable[node] == -1:
                    reachable[node] = 1
                    queue.append(node)
    return reachable
 
 
# Main fucntion
"""
for i = 1 to m do
    Realize every edge in edges with probability p and set new.edges to be the set of realized edges.
    Set ri to be the number of nodes in V reachable from any s 2 S (via, say, a BFS search).
end for
return f(S)
"""
def sampleInfluence(G,S,m):
    # initialize a placeholder for the number of nodes influenced by S in each sample graph
    influence=[]
    for i in range(0,m,1):
        # generate random graph with a pre-defined funciton 
        sampleG=genRandGraph(G)
        # for each node in S, use BFS to find out which nodes are reachable
        # make a new dict indicating reacheability by all nodes in S
        # then, count how many reachable nodes as our ith estimate for f(S)
        reachable={}

        for i in G.nodes():
            reachable[i] = -1 # -1 indicates unvisited
            
        for node in S: 
            if node in sampleG.nodes():
                indInfluence = BFS(sampleG, node) 
            elif node not in sampleG.nodes(): 
                indInfluence ={node:1} # node can only influecne itself
                 
            #Now, we know which set of nodes an individual node in S can influence
            #Modify the overall reachable dict
            for j in indInfluence.keys():
                    if indInfluence[j]==1:
                        reachable[j] = 1 # node j can be influeced by at least a node in S
   
        influence.append(len([e for e in reachable.values() if e==1]))
     
    # empirical estiamte of the influence f(S)   
    return(np.mean(influence))
    
# Run sampleInfluence(G; S;m) 
f=sampleInfluence(G,[17,23,42,2017],500)  
print("f(S)=", f)   
 
 

### d.
"""
Using the Greedy Algorithm described in problem 3, and SampleInfluence to approximate
influence, what 5 students (nodes) should Raynor pick to give free L’Espalier to? For this
subpart, please use a value of m >= 10. (No, you cannot just give it to yourself.) In addition,
please report the value of f_S(a) at every step as you build your subset S, as described in the
Greedy Algorithm.  
"""
def Greedy(G, n, m):
    """The Greedy algorithm takes a graph G, number of initial adopters n, and sample size m; 
    and returns the set of initial adopters S that maximize influence. """
    
    for i in range(0,n,1):
        
        if i==0:
            marginal_f={} # placeholder for marginal influence of node a to S
            for a in G.nodes():
                marginal_f[a]= sampleInfluence(G,[a],m) 
                print(marginal_f[a])

            # get the element with highest marginal f
            a_i=max(marginal_f, key=marginal_f.get)
            S=[a_i]
            # keep the highest marginal f
            f=[marginal_f[a_i]]
            print("S=",S)
            print("Marginal contribution:",f)
        else:
            marginal_f={} # placeholder for marginal influence of node a to S
            for a in G.nodes():
                if a not in S:
                    Sa=S+[a]
                    f_S=sampleInfluence(G,S,m)   
                    f_Sa=sampleInfluence(G,Sa,m)
                    marginal_f[a]= f_Sa-f_S
                    print(marginal_f[a])
            
            # get the element with highest marginal f
            a_i=max(marginal_f, key=marginal_f.get)
            S=S+[a_i]
            # keep the highest marginal f
            f.append(marginal_f[a_i])
            print("S=",S)
            print("Marginal contribution:",f)
         
    return((S,f))
        
            
result=Greedy(G, 5, 20)     
S, f= result   
      