import networkx as nx
from pyvis.network import Network
import random

def create_networkx_graph(num_nodes=25, num_edges=40):
    graph = nx.Graph()
    # Add nodes
    graph.add_nodes_from(range(num_nodes)) # Create nodes with ID's from 0 to num_nodes -1

    # randomly edges between nodes
    while graph.number_of_edges() < num_edges:
        node_a = random.randint(0, num_nodes - 1)
        node_b = random.randint(0, num_nodes - 1)

        # no self-loops or duplicates
        if node_a != node_b and not graph.has_edge(node_a, node_b):
            graph.add_edge(node_a, node_b)

    return graph

def visualize(graph):
    net = Network(notebook=True, directed=False)

    # Add nodes and edges from the graph
    for node in graph.nodes:
        net.add_node(node, label=str(node))
    for edge in graph.edges:
        net.add_edge(*edge)
    return net


# Create a graph and visualize it with PyVis
num_nodes = 25
num_edges = 40
graph = create_networkx_graph(num_nodes=num_nodes, num_edges=num_edges)
network_visualization = visualize(graph)
network_visualization.show("network.html")
