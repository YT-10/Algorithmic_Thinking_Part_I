# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:06:10 2017

@author: TatlierY
"""

import urllib

CITATION_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_phys-cite.txt"

def load_graph(graph_url):
    """
    Function that loads a graph given the URL
    for a text representation of the graph
    
    Returns a dictionary that models a graph
    """
    graph_file = urllib.request.urlopen(CITATION_URL)
    graph_text = str(graph_file.read())
    graph_text = graph_text #+"'"
    graph_lines = graph_text.split(' \\r\\n')
    del graph_lines[-1]
    graph_lines[0]=graph_lines[0][2:]
    test_list=[]

    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        test_list.append(node)
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))
    
    return answer_graph

#Load data
#citation_graph = load_graph(CITATION_URL)