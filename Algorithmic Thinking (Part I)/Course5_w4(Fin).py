"""
Algoritmic Thinking - Project 2 

Functions to carry out Breadth first search (bfs) and determine connectedness and resilience
on graphs 
"""

#%%
#Load data

"""
Provided code for Application portion of Module 2
"""

# general imports
import random
import time
import math
import matplotlib.pyplot as plt
from UPA import UPATrial
from load_network_graph import copy_graph, delete_node, targeted_order, NETWORK_URL, load_graph 
#%%

from collections import deque

def bfs_visited(ugraph, start_node):
    """Takes the undirected graph ugraph and the node start_node and returns the set 
    consisting of all nodes that are visited by a breadth-first search that starts 
    at start_node."""
    search_queue=deque()
    visited_set=set([start_node])
    search_queue.append(start_node)
    while(len(search_queue)>0):
        node=search_queue.pop()
        for neighbor in ugraph[node]:
            if(neighbor not in visited_set):
                visited_set.update({neighbor})
                search_queue.append(neighbor)
    
    return visited_set

def cc_visited(ugraph):
    """ Takes the undirected graph ugraph and returns a list of sets, 
    where each set consists of all the nodes (and nothing else) in a connected component, 
    and there is exactly one set in the list for each connected component in ugraph and 
    nothing else."""
    remaining_nodes=set(ugraph.keys())
    connected_list=[]
    while(len(remaining_nodes)>0):
        random_node=random.choice(list(remaining_nodes))
        connected_subset=bfs_visited(ugraph,random_node)
        connected_list.append(connected_subset)
        remaining_nodes = remaining_nodes-connected_subset
    return connected_list
        
def largest_cc_size(ugraph):
    """Takes the undirected graph ugraph and returns the size (an integer) of the 
    largest connected component in ugraph."""
    #max_size=0
    try:
        max_size=max([len(element) for element in cc_visited(ugraph)])
    except ValueError:
        max_size=0
    return max_size

def compute_resilience(ugraph,attack_order):
    """ Takes the undirected graph ugraph, a list of nodes attack_order and iterates through 
    the nodes in attack_order. For each node in the list, the function removes the given node 
    and its edges from the graph and then computes the size of the largest connected component 
    for the resulting graph. The function should return a list whose k+1th entry is the size 
    of the largest connected component in the graph after the removal of the first k nodes 
    in attack_order. The first entry (indexed by zero) is the size of the largest connected 
    component in the original graph."""
    ugraph_copy=copy_graph(ugraph)
    longest_dist_list=[largest_cc_size(ugraph_copy)]
    
    for key_element in attack_order:
        ugraph_copy.pop(key_element)
        for value_element in ugraph_copy.values():
            value_element.discard(key_element)
        longest_dist_list.append(largest_cc_size(ugraph_copy))
    return longest_dist_list
                 
#%%
# Functions that are needed

def check_undirected_graph(ugraph):
    """
    Function checks whether graph is undirected. Returns True if this is the case
    and the problematic node if this is not the case
    """
    nodes_list=list(ugraph.keys())
    check=True
    for node_i in nodes_list:
        for node_j in ugraph[node_i]:
            if(node_i not in ugraph[node_j]):
                check=False
                return node_i,node_j
    return check

def find_edges(ugraph,search_node):
    """
    Finds all nodes that contain node
    """
    nodes_set=set()
    for node in ugraph.keys():
        if(search_node in ugraph[node]):
            nodes_set.add(search_node)
            
    return nodes_set

def total_number_edges(ugraph):
    """
    Returns total number of edges in a undirected graph
    """
    
    return len([x for set_element in ugraph.values() for x in set_element])/2

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

def random_order(ugraph):
    
    all_nodes=list(ugraph.keys())
    random.shuffle(all_nodes)
    return all_nodes

#ER graph
def undirected_randConnect(node_length,rand_prob):
    #Initialize an empty dictionary
    start_list=[(i,set()) for i in range(node_length)]
    rand_graph=dict(start_list)

    for node_i in range(node_length):
        for node_j in [x for x in range(node_length) if x != node_i]:
            if random.random()<rand_prob:
                rand_graph[node_i].add(node_j)
                rand_graph[node_j].add(node_i)
    return(rand_graph)

#UPA graph
def UPA_fun(number_nodes,start_m):
    E=make_complete_graph(start_m)
    upa=UPATrial(start_m)
    for node_i in range(start_m,number_nodes):
        trial_nodes=upa.run_trial(start_m)
        E[node_i]=trial_nodes
        for node_j in trial_nodes:
            E[node_j].add(node_i)
    return E
    
#%%
#Q1

#Network graph
network_graph=load_graph(NETWORK_URL)
total_nodes=len(network_graph)
total_edges=total_number_edges(network_graph)

network_graph_resilience_random=compute_resilience(network_graph,random_order(network_graph))

#ER graph

er_graph=undirected_randConnect(total_nodes,0.0016)
#Check number of nodes
total_number_edges(er_graph)
#Examine resilience
er_graph_resilience_random=compute_resilience(er_graph,random_order(er_graph))

#UPA
m=2
upa_graph=UPA_fun(total_nodes,m)
upa_graph_resilience_random=compute_resilience(upa_graph,random_order(upa_graph))

#Make Plot
plt.xlabel('Number of removed nodes')
plt.ylabel('Largest connected path')
plt.title('Graph resilience random target order')
    
plt.subplot(111)
plt.plot(network_graph_resilience_random,'b',
           er_graph_resilience_random,'r',
           upa_graph_resilience_random,'g')
plt.legend(('Network Graph','ER Graph','UPA Graph'),loc='upper right')
plt.show()

#%%
#Q2
def random_order_bounded(ugraph,fraction):
    """"
    Randomly remove 20% of the nodes in the graph
    """
    all_nodes=list(ugraph.keys())
    random.shuffle(all_nodes)
    selection_bound=math.ceil(len(all_nodes)*fraction)
    return all_nodes[:selection_bound]

def remove_node_fraction(ugraph,fraction):
    """
    Removes specified fraction (fraction) of the ugraph
    """

    ugraph_copy=copy_graph(ugraph)
    attack_order_nodes=random_order_bounded(ugraph_copy,fraction)

    for node in attack_order_nodes:
        delete_node(ugraph_copy,node)

    return ugraph_copy

#Remove 25% of the nodes for network graph
ng_copy_attacked=remove_node_fraction(network_graph,0.25)
largest_ng_cc_size=largest_cc_size(ng_copy_attacked)
remaining_ng_nodes=len(ng_copy_attacked.keys())
abs((remaining_ng_nodes-largest_ng_cc_size)/remaining_ng_nodes)<0.20

#Remove 25% of the nodes for er graph
er_copy_attacked=remove_node_fraction(er_graph,0.25)
largest_er_cc_size=largest_cc_size(er_copy_attacked)
remaining_er_nodes=len(er_copy_attacked.keys())
abs((remaining_er_nodes-largest_er_cc_size)/remaining_er_nodes)<0.20

#Remove 25% of the nodes for network graph
upa_copy_attacked=remove_node_fraction(upa_graph,0.25)
largest_upa_cc_size=largest_cc_size(upa_copy_attacked)
remaining_upa_nodes=len(upa_copy_attacked.keys())
abs((remaining_upa_nodes-largest_upa_cc_size)/remaining_upa_nodes)<0.20

#%%
#Q3

import time

def running_times(ref_fun,value):

    start_time = time.clock()
    ref_fun(value)
    return(time.clock() - start_time)

def fast_targeted_order(ugraph):
    """
    """
    #number_nodes=len(ugraph)
    ugraph_copy=copy_graph(ugraph)
    
    degree_sets={index_i:set() for index_i in ugraph.keys()}

    for index_i in ugraph_copy.keys():
        degree=len(ugraph_copy[index_i])
        degree_sets[degree].add(index_i)
    
    nodes_ordered_list=[]
    
    for index_k in sorted(ugraph_copy.keys(),reverse=True):
        
        while(len(degree_sets[index_k])>0):
            element_u=random.sample(degree_sets[index_k],1)[0]
            degree_sets[index_k].discard(element_u)
            
            for neighbor in ugraph_copy[element_u]:
                degree=len(ugraph_copy[neighbor])
                degree_sets[degree].discard(degree) 
                degree_sets[degree-1].discard(degree)
            
            #print(len(nodes_ordered_list),index_j)
            nodes_ordered_list.append(element_u)
            #index_j += 1
            delete_node(ugraph_copy,element_u)
    
    return nodes_ordered_list

#Comparison run times targeted_order vs fast_targeted_order
x_values=list(range(10,1000,10))
measure_values=[UPA_fun(x,5) for x in x_values]  
targeted_runtimes=[running_times(targeted_order,ugraph) for ugraph in measure_values]              
fast_targeted_runtimes=[running_times(fast_targeted_order,ugraph) for ugraph in measure_values] 

# Q3: Make plot
plt.xlabel('Number of nodes')
plt.ylabel('Run time (seconds)')
plt.title('Runtimes targeted_order and fast_targeted_order for UPA graphs (m=5)')
 
   
plt.subplot(111)
plt.plot(x_values,targeted_runtimes,'b',x_values,fast_targeted_runtimes,'r')
plt.legend(('targeted order','fast targeted order'),loc='upper left')
plt.show()

#%%
#Q4

network_graph_resilience_fast=compute_resilience(network_graph,fast_targeted_order(network_graph))
upa_graph_resilience_fast=compute_resilience(upa_graph,fast_targeted_order(upa_graph))
er_graph_resilience_fast=compute_resilience(er_graph,fast_targeted_order(er_graph))

plt.xlabel('Number of removed nodes')
plt.ylabel('Largest connected path')
plt.title('Graph resilience fast target order')
    
plt.subplot(111)
plt.plot(network_graph_resilience_fast,'b',
           er_graph_resilience_fast,'r',
           upa_graph_resilience_fast,'g')
plt.legend(('Network Graph','ER Graph','UPA Graph'),loc='upper right')
plt.show()