"""
Provided code for Application portion of Module 1

Imports physics citation graph 
"""

# general imports
import urllib
import matplotlib.pyplot as plt
import random
import math
from load_graph_data import load_graph, CITATION_URL
from DPA import DPATrial

citation_graph = load_graph(CITATION_URL)

# Set timeout for CodeSkulptor if necessary
#import codeskulptor
#codeskulptor.set_timeout(20)

"""
This code creates dictionaries corresponding to some simple examples of graphs. 
It also implements two short functions that compute information about the distribution of the in-degrees for nodes in these graphs. 
These functions are used in the Application component of Module 1, where we analyze the degree distribution of a citation graph 
for a collection of physics papers.
"""

# We define a few graphs
EX_GRAPH0={0:set([1,2]),1:set([]),2:set([])}
EX_GRAPH1={0:set([1,4,5]),1:set([2,6]),2:set([3]),3:set([0]),4:set([1]),5:set([2]),6:set([])}
EX_GRAPH2={0:set([1,4,5]),1:set([2,6]),2:set([3,7]),3:set([7]),4:set([1]),5:set([2]),6:set([]),7:set([3]),8:set([1,2]),9:set([0,3,4,5,6,7])}

def make_complete_graph(num_nodes):
    """
    Takes the number of nodes num_nodes and returns a dictionary corresponding to a complete directed graph with the specified number of nodes. 
    A complete graph contains all possible edges subject to the restriction that self-loops are not allowed. The nodes of the graph 
    should be numbered 0 to num_nodes - 1 when num_nodes is positive. Otherwise, the function returns a dictionary corresponding to the empty graph.
    """
    dict_graph={}
    
    for key in range(num_nodes):
        dict_graph[key]=set.difference(set([i for i in range(num_nodes)]),set([key]))
    
    return dict_graph 
      
def compute_in_degrees(digraph):
    """
    Takes a directed graph digraph (represented as a dictionary) and computes the in-degrees for the nodes in the graph. 
    The function should return a dictionary with the same set of keys (nodes) as digraph whose corresponding values are the number of edges 
    whose head matches a particular node.
    """
    in_degree_dict=dict.fromkeys(digraph,0)
    
    for set_element in digraph.values():
        for single_element in set_element:
            in_degree_dict[single_element] += 1
        
    return(in_degree_dict)

def in_degree_distribution(digraph):
    """
    Takes a directed graph digraph (represented as a dictionary) and computes the unnormalized distribution of the in-degrees of the graph. 
    The function should return a dictionary whose keys correspond to in-degrees of nodes in the graph. The value associated with each particular 
    in-degree is the number of nodes with that in-degree. In-degrees with no corresponding nodes in the graph are not included in the dictionary.
    """
    in_degree_dict=compute_in_degrees(digraph)
    total_list=[x for x in in_degree_dict.values()]
    digraph_length=len(digraph)
    in_degrees_dist_dict=dict({(x,total_list.count(x)) for x in range(digraph_length) if (total_list.count(x)>0)})
    return in_degrees_dist_dict

def normalizeGraph(digraph):
    digraph_c=digraph.copy()
    total_degree=sum([item for x in digraph_c.values() for item in x])
    for key,value in digraph_c.items():
        digraph_c[key]=set([float(item)/total_degree for item in value])
    return digraph_c

def normalizeSetList(digraph):
    digraph_c=digraph.copy()
    total_degree=sum(digraph.values())
    for key,value in digraph_c.items():
        digraph_c[key]=digraph_c[key]/total_degree
    return digraph_c

#%%
# Make a plot of distribution of a graph
def normPlot(digraph):
    cit_dist_graph=in_degree_distribution(digraph)
    nor_cit_dist_graph=normalizeSetList(cit_dist_graph)
    return nor_cit_dist_graph

def printGraph(digraph,def_title='Log/Log plot of in-degree-distribution of citations in energy physics theory papers'):
    plt.xlabel('Log degrees')
    plt.ylabel('Log distribution')
    plt.title(def_title)
    
    nor_cit_dist_graph=normPlot(digraph)
    plt.loglog(list(nor_cit_dist_graph.keys()),list(nor_cit_dist_graph.values()),'o')

def printMultGraph(digraph1,digraph2,digraph3,leg1,leg2,leg3):
    plt.xlabel('Log degrees')
    plt.ylabel('Log distribution')
    plt.title('Log/Log plot of in-degree-distribution of citations in energy physics theory papers')
    
    nor_cit_dist_graph1=normPlot(digraph1)
    nor_cit_dist_graph2=normPlot(digraph2)
    nor_cit_dist_graph3=normPlot(digraph3)
    
    plt.subplot(111)
    plt.loglog(list(nor_cit_dist_graph1.keys()),list(nor_cit_dist_graph1.values()),'o',
               list(nor_cit_dist_graph2.keys()),list(nor_cit_dist_graph2.values()),'ro',
               list(nor_cit_dist_graph3.keys()),list(nor_cit_dist_graph3.values()),'go')
    plt.legend((leg1,leg2,leg3),loc='upper right')

    plt.show()

#%%
#Q1
printGraph(citation_graph)

#%%
#Q2
def randConnect(node_length,rand_prob):
    rand_graph={}
    for node_i in range(node_length):
        rand_graph[node_i]=set([x for x in range(node_length) if x!=node_i if random.random()<rand_prob])
    return(rand_graph)

rand_graph1=randConnect(5000,0.20)
rand_graph2=randConnect(5000,0.40)
rand_graph3=randConnect(5000,0.60)
printMultGraph(rand_graph1,rand_graph2,rand_graph3,'p=0.2,n=5000','p=0.4,n=5000','p=0.6,n=5000')

#%%
#Q3
n=len(citation_graph)
total_in_degree=len([item for x in citation_graph.values() for item in x])
m=total_in_degree/n
# So choose n=27770 (total number of nodes) and m=12 (rounded average number of edges)

#%%
#Q4

from DPA import DPATrial 

def DPA_fun(number_nodes,start_m):
    E=make_complete_graph(start_m)
    dpa=DPATrial(start_m)
    for node_i in range(start_m,number_nodes):
        E[node_i]=dpa.run_trial(start_m)
    return E

#Construct the graph
dpa_graph=DPA_fun(27770,12)
printGraph(dpa_graph,'Log/Log plot of in-degree-distribution of DPA graph (n=27770,m=12)')