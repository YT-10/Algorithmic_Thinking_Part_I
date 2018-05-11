"""
Algorithmic Thinking - Module 1
09-07-2014
Graph Basics and Random Digraphs
Degree Distributions for graphs
Project File
"""

#Example graphs to be created
EX_GRAPH0 = {0: set([1, 2]),
          1: set([]),
          2: set([])}

EX_GRAPH1 = {0: set([1, 4, 5]),
          1: set([2, 6]),
          2: set([3]),
          3: set([0]),
          4: set([1]),
          5: set([2]),
          6: set([])}

EX_GRAPH2 = {0: set([1, 4, 5]), 
          1: set([2, 6]),
          2: set([3, 7]),
          3: set([7]),
          4: set([1]),
          5: set([2]),
          6: set([]),
          7: set([3]),
          8: set([1, 2]),
          9: set([0, 3, 4, 5, 6, 7])}

def make_complete_graph(num_nodes):
    """
    Takes number of nodes and returns dictionary with complete directed graph.
    
    num_nodes (int) - number of nodes to build into graph
    return: graph (dict)
    """
    #initialize graph variable
    graph = {}
    
    #loop for creating complete graph
    for node in range(num_nodes):
        nodes = set([])
        for to_node in range(num_nodes):
            if node != to_node:
                nodes.add(to_node)
        graph[node] = nodes
    
    #return complete graph 
    return graph
    
def compute_in_degrees(digraph):
    """
    Takes in a digraph and computes the in-degrees for each node.
    
    digraph (dict) - directional graph
    return: dictionary with in-degrees for each node
    """
    #initialize variables
    degrees = dict.fromkeys(digraph, 0)
    
    #calculate in-degrees for each node in directional graph
    for node in digraph:
        for edge in digraph[node]:
            degrees[edge] += 1

    #return # of total in degrees
    return degrees
    
def in_degree_distribution(digraph):
    """
    Takes in a digraph and computes the unnormalized distribution for the in-degrees
    
    digraph (dict) - directional graph
    return: 
    """
    #initialize variables
    computed = compute_in_degrees(digraph)
    distrib = {}
    
    #computes distribution for the in-degrees
    for node in computed:
        degree = computed[node]
        if degree not in distrib:
            distrib[degree] = 1
        else:
            distrib[degree] += 1
    
    #return dictionary of distribution
    return distrib