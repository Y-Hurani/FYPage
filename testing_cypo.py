import dash
from dash import dcc, html
import dash_cytoscape as cyto
import networkx as nx

def nx_to_cytoscape(graph, colors, moods = None):
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
        row = node // 10
        col = node % 10
        position = {"x": col * 125, "y": row * 125}  # Scale the positions

        elements.append({
            'data': {
                'id': str(node),
                'label': f'{node} | Mood: {moods[node]}' if moods else f'Node {node}',
                'color': color,
                'mood': moods[node] if moods else 'Bug'
            },
            'position': position
        })

    for edge in graph.edges():
        elements.append({
            'data': {'source': str(edge[0]), 'target': str(edge[1])}
        })

    return elements


def grid_layout_positions(graph, grid_size=10):
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
                "style": 
                {
                "background-color": "data(color)",
                "label": "data(label)",
                "border-width": 2,
                "border-color": "black",
                "border-style": "solid"
                }
            },
            {
                "selector": "edge",
                "style": {"line-color": "#ccc"}
            }
        ]
    )


def create_dash_app(graph, colors):
    """Create and return a Dash app for the given NetworkX graph."""
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Network Visualization"),
        html.Div(id="update-div"),
        dcc.Store(id='mood-store', data={}),
        dcc.Store(id='elements', data={}),
        dcc.Store(id='color-store', data={}),
        dcc.Interval(
            id="update-interval",
            interval=250,  # 2 seconds
            n_intervals=0
        ),
        cytoscape_with_layout(graph, colors)
    ])

    @app.callback(
        dash.Output('cytoscape-graph', 'elements'),
        dash.Output('color-store', 'data'),
        [dash.Input('update-interval', 'n_intervals')],  # Triggers update
        [dash.State('color-store', 'data')]  # Reads stored colors
    )
    def update_graph(_, stored_colors):
        # print("Callback triggered")  # Debugging
        stored_colors = app.layout.children[-2].data
        stored_moods = app.layout.children[-3].data
        if stored_colors:
            #print("Stored colors in callback:", stored_colors)  # Debugging
            return nx_to_cytoscape(graph, stored_colors, stored_moods), stored_colors
        return nx_to_cytoscape(graph, colors), colors  # Fallback
    
    def update_graph_colors(new_colors):
        global current_colors
        current_colors = new_colors.copy() if new_colors else []
        # The callback will handle updating the graph on next interval

    # Attach the method to the app
    app.update_graph_colors = update_graph_colors

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
