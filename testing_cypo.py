import dash
from dash import dcc, html
import dash_cytoscape as cyto
import networkx as nx

def nx_to_cytoscape(graph, colors):
    """
    Converts a NetworkX graph into Cytoscape elements with conditional node coloring.
    :param graph: NetworkX graph.
    :param condition: A function that takes a node and agents dict as input and returns a color string.
    :param agents: A dictionary of agents corresponding to the nodes.
    """
    elements = []

    for node in graph.nodes():
        # Default color
        color = 'orange' if len(colors)<=node else colors[node]
        #print(colors)

        # Determine grid position for 7x7 layout
        row = node // 7
        col = node % 7
        position = {"x": col * 100, "y": row * 100}  # Scale the positions

        elements.append({
            'data': {'id': str(node), 'label': f'Node {node}', 'color': color},
            'position': position,
            'classes': color
        })

    for edge in graph.edges():
        elements.append({
            'data': {'source': str(edge[0]), 'target': str(edge[1])}
        })

    return elements


def grid_layout_positions(graph, grid_size=7):
    """Generate fixed positions for a 7x7 grid."""
    positions = {}
    for node in graph.nodes:
        row = node // grid_size
        col = node % grid_size
        positions[node] = {"x": col * 100, "y": row * 100}
    return positions


def cytoscape_with_layout(graph, colors):
    """Create a Dash Cytoscape component with a 7x7 grid layout."""
    elements = nx_to_cytoscape(graph, colors)
    positions = grid_layout_positions(graph)

    layout = [
        {"data": {"id": str(node), "label": str(node)},
         "position": positions[node]} for node in graph.nodes
    ]
    for edge in graph.edges:
        layout.append({
            "data": {"source": str(edge[0]), "target": str(edge[1])}
        })

    return cyto.Cytoscape(
        id='cytoscape-graph',
        elements=layout,
        layout={"name": "preset"},
        style={"width": "100%", "height": "500px"},
        stylesheet=[
            {
                "selector": "node",
                "style": {"label": "data(label)"}
            },
            {
                "selector": "edge",
                "style": {"line-color": "#ccc"}
            },
            {
                "selector": ".red",
                "style": {
                    'background-color': 'red',
                    'line-color': 'red'
                }
            },
            {
                "selector": ".blue",
                "style": {
                    'background-color': 'blue',
                    'line-color': 'blue'
                }
            },
            {
                "selector": ".green",
                "style": {
                    'background-color': 'green',
                    'line-color': 'green'
                }
            },
            {
                "selector": ".orange",
                "style": {
                    'background-color': 'orange',
                    'line-color': 'orange'
                }
            },
            {
                "selector": ".pink",
                "style": {
                    'background-color': 'pink',
                    'line-color': 'pink'
                }
            },
            {
                "selector": ".yellow",
                "style": {
                    'background-color': 'yellow',
                    'line-color': 'yellow'
                }
            }

        ]
    )


def create_dash_app(graph, colors):
    """Create and return a Dash app for the given NetworkX graph."""
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Network Visualization"),
        html.Div(id="update-div"),
        dcc.Interval(
            id="update-interval",
            interval=500,  # 2 seconds
            n_intervals=0
        ),
        cytoscape_with_layout(graph, colors)
    ])

    @app.callback(
        dash.Output('cytoscape-graph', 'elements'),
        [dash.Input('update-interval', 'n_intervals')]
    )
    def update_graph(_):
        """Update the graph whenever the callback is triggered."""
        return nx_to_cytoscape(graph, colors)

    return app


# Example usage
if __name__ == "__main__1":
    # Create a sample graph
    G = nx.Graph()
    G.add_nodes_from(range(49))  # 49 nodes, 0 to 48
    G.add_edges_from([(0, 1), (1, 2), (3, 4)])  # Add some edges

    # Run the Dash app
    app = create_dash_app(G)
    app.run_server(debug=True)
