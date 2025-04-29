import dash
from dash import dcc, html
import dash_cytoscape as cyto
import networkx as nx
import os, sys

# Global state variables
CURRENT_COLORS = []
CURRENT_MOODS = {}

def nx_to_cytoscape(graph, colors, dimensions, moods = None, positions = None):
    """
    Converts a NetworkX graph into Cytoscape elements with conditional node coloring.
    :param graph: NetworkX graph.
    :param condition: A function that takes a node and agents dict as input and returns a color string.
    :param agents: A dictionary of agents corresponding to the nodes.
    """
    elements = []
    offset = 12.5
    for node in graph.nodes():
        # Default color
        color = 'orange' if len(colors) <= node else colors[node]
        
        # Get mood if available
        mood = None if not moods else moods[node]
        
        # Determine grid position
        row = node // dimensions
        col = node % dimensions
        position = positions[node] if positions else {"x": col * 125 + (row % 2) * offset, "y": row * 125 + (col % 2) * offset}  # Scale the positions
        #offset -= (offset * 2) - 25
        elements.append({
            'data': {
                'id': str(node),
                'label': f'{node} | Mood: {mood}' if moods else f'Node {node}',
                'color': color,
                'mood': mood
            },
            'position': position,
            'classes': color  # Add this for CSS styling if needed
        })

    for edge in graph.edges():
        elements.append({
            'data': {'source': str(edge[0]), 'target': str(edge[1])}
        })

    return elements


def grid_layout_positions(graph, dimensions):
    """Generate fixed positions for a grid."""
    positions = {}
    offset = 50
    for node in graph.nodes:
        row = node // dimensions
        col = node % dimensions
        positions[node] = {"x": (col * 100 + offset), "y": (row * 100 + offset)}
        offset *= -1
    return positions


def cytoscape_with_layout(graph, colors, dimensions, positions):
    """Create a Dash Cytoscape component with a grid layout."""
    elements = nx_to_cytoscape(graph, colors, dimensions, positions)
    if not positions:
        positions = grid_layout_positions(graph, dimensions)

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
        zoom=3,
        pan={'x': 0, 'y': 0},
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


def create_dash_app(graph, colors, dimensions):
    app = dash.Dash(__name__)

    # Initialize data cache for external updates
    app.latest_data = {
        'colors': colors.copy() if colors else [],
        'moods': {}
    }
    
    # Method for game loop to update data
    def update_data(new_colors, new_moods):
        app.latest_data['colors'] = new_colors
        app.latest_data['moods'] = new_moods
    
    # Attach the method to the app
    app.update_data = update_data

    app.layout = html.Div([
        html.H1("Network Visualization"),
        html.Div(id="update-div"),
        dcc.Store(id='mood-store', data={}),
        dcc.Store(id='color-store', data={}),
        dcc.Interval(
            id="update-interval",
            interval=500,  # Changed to 500ms for better performance
            n_intervals=0
        ),
        cytoscape_with_layout(graph, colors, dimensions, None)
    ])

    # CALLBACK 1: Sync data from app.latest_data to the stores
    @app.callback(
        [dash.Output('color-store', 'data'),
         dash.Output('mood-store', 'data')],
        [dash.Input('update-interval', 'n_intervals')]
    )
    def sync_data_to_stores(n_intervals):
        return app.latest_data['colors'], app.latest_data['moods']

    # CALLBACK 2: Update graph when store data changes
    @app.callback(
        [dash.Output('cytoscape-graph', 'elements'),
        dash.Output('cytoscape-graph', 'zoom'),
        dash.Output('cytoscape-graph', 'pan')
        ],
        [dash.Input('color-store', 'data'),
        dash.Input('mood-store', 'data')],
        [dash.State('cytoscape-graph', 'zoom'),
        dash.State('cytoscape-graph', 'pan')],
        allow_duplicate=True
    )
    def update_graph(colors, moods, zoom, pan):
        return nx_to_cytoscape(graph, colors, dimensions, moods = moods), zoom, pan

    return app

def attach_simulation_mode(app, graph, colors, dimensions, positions):
    # Initialize data cache for external updates
    app.latest_data = {
        'colors': colors.copy() if colors else [],
        'moods': {}
    }
    def update_data(new_colors, new_moods):
        app.latest_data['colors'] = new_colors
        app.latest_data['moods'] = new_moods

    app.update_data = update_data
    print('about to pass')
    app.layout = html.Div([
        html.H1("Network Visualization"),
        html.Div([
            html.Button("Terminate Simulation", id="restart-button", 
                       style={"backgroundColor": "#4CAF50", "color": "white", 
                              "padding": "10px 15px", "border": "none", 
                              "borderRadius": "4px", "cursor": "pointer"}),
            html.Div(id="restart-status")
        ]),
        html.Div(id="update-div"),
        dcc.Store(id='mood-store', data={}),
        dcc.Store(id='color-store', data={}),
        dcc.Interval(id="update-interval", interval=500, n_intervals=0),
        cytoscape_with_layout(graph, colors, dimensions, positions)
    ])
    print('passed')
    # Rebind Callbacks
    @app.callback(
        [dash.Output('color-store', 'data'),
         dash.Output('mood-store', 'data')],
        [dash.Input('update-interval', 'n_intervals')]
    )
    def sync_data_to_stores(n_intervals):
        return app.latest_data['colors'], app.latest_data['moods']
    @app.callback(
        [dash.Output('cytoscape-graph', 'elements'),
        dash.Output('cytoscape-graph', 'zoom'),
        dash.Output('cytoscape-graph', 'pan')],
        [dash.Input('color-store', 'data'),
        dash.Input('mood-store', 'data')],
        [dash.State('cytoscape-graph', 'zoom'),
        dash.State('cytoscape-graph', 'pan')]
    )

    def update_graph(colors, moods, zoom, pan):
        return nx_to_cytoscape(graph, colors, dimensions, moods = moods, positions = positions), zoom, pan
    
    @app.callback(
        dash.Output('restart-status', 'children'),
        dash.Input('restart-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def restart_server(n_clicks):
        if n_clicks:
            # This will restart the current Python process
            print("Terminating server...")
            os.execl(sys.executable, sys.executable, *sys.argv)
            # Note: The code after this line won't execute due to the restart
        return ""
# Example usage
if __name__ == "__main__1":
    # Create a sample graph
    G = nx.Graph()
    G.add_nodes_from(range(49))  # 49 nodes, 0 to 48
    G.add_edges_from([(0, 1), (1, 2), (3, 4)])  # Add some edges

    # Run the Dash app
    app = create_dash_app(G)
    app.run_server(debug=True)
