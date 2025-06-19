import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import networkx as nx
import dash_cytoscape as cyto
import math
import json
#from testing_cypo import create_dash_app
from test3 import main, create_networkx_graph, add_edges_to_graph
from testing_cypo import attach_simulation_mode
import sys, os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


# import the shared server
from flask_server import get_server



# flask app using the shared server
server = get_server()


node_colors = {}
color_cycle = ['red', 'green', 'orange', 'pink', 'white', 'yellow']
node_positions = {}  # global dictionary to store node positions
agent_color_map = {
            'SARSAAgent': 'yellow',
            'MoodySARSAAgent': 'white',
            'CooperativeAgent': 'green',
            'DefectingAgent': 'red',
            'TFTAgent': 'pink',
            'WSLSAgent': 'orange'
}

def create_gui_app(server):
    app = dash.Dash(__name__, 
                    server=server,
                    url_base_pathname='/',
                    external_stylesheets=[dbc.themes.BOOTSTRAP],
                    suppress_callback_exceptions=True)

    agent_weights = [
        "SARSAAgent", "MoodySARSAAgent", "CooperativeAgent",
        "DefectingAgent", "TFTAgent", "WSLSAgent"
    ]

    def nx_to_cytoscape(graph, colors, dimensions, moods=None, use_saved_positions=False):
        elements = []
        offset = 12.5
        for node in graph.nodes():
            color = colors.get(node, 'orange')
            mood = None if not moods else moods[node]

            if use_saved_positions and node in node_positions:
                position = node_positions[node]
            else:
                row = node // dimensions
                col = node % dimensions
                position = {"x": col * 125 + (row % 2) * offset, "y": row * 125 + (col % 2) * offset}
                node_positions[node] = position  # Save position for future use

            elements.append({
                'data': {
                    'id': str(node),
                    'label': f'{node} | Mood: {mood}' if moods else f'Node {node}',
                    'color': color,
                    'mood': mood
                },
                'position': position,
                'classes': color
            })
        for edge in graph.edges():
            elements.append({
                'data': {'source': str(edge[0]), 'target': str(edge[1])}
            })
        return elements

    app.layout = dbc.Container([
        dcc.Location(id='url', refresh=True),
        html.Div(id='refresh-div', style={"display": "none"}),
        html.H2("Simulation Configuration"),
        dbc.Row([
            dbc.Label("Number of Nodes", className="mt-3", style={"margin-right": "15px"}),
            dcc.Input(id='num-nodes', type='number', value=100, min=1, step=1)
        ]),
        dbc.Row([
            dbc.Label("Initial Number of Edges", className="mt-3", style={"margin-right": "15px"}),
            dcc.Input(id='num-edges', type='number', value=50, min=0, step=1)
        ]),
        dbc.Row([
            dbc.Label("Number of Games per Simulation", className="mt-3", style={"margin-right": "15px"}),
            dcc.Input(id='num-games', type='number', value=50000, min=1, step=1)
        ]),
        dbc.Row([
            dbc.Label("Avg. Betrayal Threshold", className="mt-3", style={"margin-right": "15px"}),
            dcc.Input(id='betrayal-threshold', type='number', value=2.5, step=0.1),
        ]),
        dbc.Row([
            dbc.Label("Max Connection Distance (Higher than 100,000 is no limit)", className="mt-3", style={"margin-right": "15px"}),
            dcc.Input(id='max-dist', type='number', value=350, step=1),
        ]),
        dbc.Row([
            dbc.Label("% Reconnection (R%)", className="mt-3", style={"margin-right": "15px"}),
            dcc.Slider(id='reconnect-pct', min=0, max=1, step=0.01, value=0.2,
                    marks={0: '0%', 0.5: '50%', 1: '100%'}, tooltip={"placement": "bottom", "always_visible": True})
        ]),
        dbc.Row([
            dbc.Label("Edge Creation Method", className="mt-3", style={"margin-right": "15px"}),
            dbc.RadioItems(
                options=[
                    {"label": "WIPE (Wipe and Initialize Positions and Edges)", "value": "WIPE"},
                    {"label": "POP (Preserve Original Positions)", "value": "POP"}
                ],
                value="WIPE",  # Default value
                id="forgiveness-mode",
                inline=True,
                className="mt-2"
            )
        ]),


        html.Hr(),
        html.H4("Agent Type Weights"),
        *[
            dbc.Row([
                dbc.Label(
                    [
                        html.Div(
                            style={
                                'backgroundColor': agent_color_map[agent],
                                'border': '1px solid black',
                                'width': '10px',
                                'height': '10px',
                                'borderRadius': '50%',
                                'display': 'inline-block',
                                'marginRight': '8px'
                            }
                        ),
                        agent
                    ], 
                    className="mt-2 d-flex align-items-center"
                ),
                dcc.Slider(
                    id=f"weight-{agent}",
                    min=0,
                    max=1,
                    step=0.01,
                    value=1 if agent == "MoodySARSAAgent" else 0,
                    marks={0: '0', 0.5: '0.5', 1: '1'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]) for agent in agent_weights
        ],

        html.Div([
            html.H5("SARSA Parameters"),
            dbc.Label("Epsilon", style={"margin-right": "15px"}), 
            dcc.Input(id='sarsa-epsilon', type='number', value=0.1, step=0.01, style={"margin-right": "15px"}),
            dbc.Label("Alpha", style={"margin-right": "15px"}), 
            dcc.Input(id='sarsa-alpha', type='number', value=0.1, step=0.01, style={"margin-right": "15px"}),
            dbc.Label("Gamma", style={"margin-right": "15px"}), 
            dcc.Input(id='sarsa-gamma', type='number', value=0.95, step=0.01)
        ], id='sarsa-config', style={'display': 'none', 'marginTop': '20px'}),

        html.Div([
            html.H5("Moody SARSA Parameters"),
            dbc.Label("Epsilon", style={"margin-right": "15px"}), 
            dcc.Input(id='moody-epsilon', type='number', value=0.1, step=0.01, style={"margin-right": "15px"}),
            dbc.Label("Alpha", style={"margin-right": "15px"}), 
            dcc.Input(id='moody-alpha', type='number', value=0.1, step=0.01, style={"margin-right": "15px"}),
            dbc.Label("Gamma", style={"margin-right": "15px"}), 
            dcc.Input(id='moody-gamma', type='number', value=0.95, step=0.01)
        ], id='moody-config', style={'display': 'none', 'marginTop': '20px'}),

        html.Hr(),
        dbc.Button("Overview", id='overview-btn', color='primary', className='mt-4'),
        dbc.Button("Start Simulation", id='run-sim-btn', color='success', className='mt-4 ms-2', 
           style={"display": "none"}),  # Hidden by default
        html.Div(id='output', className='mt-3'),
        # Add a new element to display simulation status
        html.Div(id='simulation-status', className='mt-3'),
        
        html.Div([
            html.A("Go to Simulation View", 
                href="/simulation/", 
                className="btn btn-info mt-3",
                id="sim-link",
                style={"display": "none"})
        ]),
        html.Div(id='cytoscape-container', className='mt-4')
    ])

    @app.callback(
        Output('sarsa-config', 'style'),
        Output('moody-config', 'style'),
        Input('weight-SARSAAgent', 'value'),
        Input('weight-MoodySARSAAgent', 'value')
    )
    def toggle_learning_agent_settings(sarsa_weight, moody_weight):
        return ({'display': 'block'} if sarsa_weight > 0 else {'display': 'none'},
                {'display': 'block'} if moody_weight > 0 else {'display': 'none'})

    @app.callback(
        Output('cytoscape-container', 'children'),
        Input('overview-btn', 'n_clicks'),
        State('num-nodes', 'value'),
        State('weight-SARSAAgent', 'value'),
        State('weight-MoodySARSAAgent', 'value'),
        State('weight-CooperativeAgent', 'value'),
        State('weight-DefectingAgent', 'value'),
        State('weight-TFTAgent', 'value'),
        State('weight-WSLSAgent', 'value')
    )
    def generate_graph(n_clicks, num_nodes, sarsa_weight, moody_weight, coop_weight, defect_weight, tft_weight, wsls_weight):
        if not n_clicks:
            return dash.no_update
        global node_colors
        # Map agent types to colors
        agent_color_map = {
            'SARSAAgent': 'yellow',
            'MoodySARSAAgent': 'white',
            'CooperativeAgent': 'green',
            'DefectingAgent': 'red',
            'TFTAgent': 'pink',
            'WSLSAgent': 'orange'
        }
    
        # Create weights list and normalize them
        weights = [sarsa_weight, moody_weight, coop_weight, defect_weight, tft_weight, wsls_weight]
        total_weight = sum(weights)
    
        # If all weights are zero, set default to all orange
        if total_weight == 0:
            node_colors = {i: 'orange' for i in range(num_nodes)}
        else:
            # Normalize weights to sum to 1
            normalized_weights = [w/total_weight for w in weights]
            
            # Get colors
            colors = [agent_color_map['SARSAAgent'], agent_color_map['MoodySARSAAgent'], 
                    agent_color_map['CooperativeAgent'], agent_color_map['DefectingAgent'],
                    agent_color_map['TFTAgent'], agent_color_map['WSLSAgent']]
            
            # Randomly assign colors based on weights
            import random
            node_colors = {}
            for i in range(num_nodes):
                selected_color = random.choices(colors, weights=normalized_weights, k=1)[0]
                node_colors[i] = selected_color

        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        dimensions = int(math.sqrt(num_nodes))
        elements = nx_to_cytoscape(G, node_colors, dimensions)
        return cyto.Cytoscape(
            id='overview-graph',
            layout={'name': 'preset'},
            elements=elements,
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
                {"selector": "edge", "style": {"line-color": "#ccc"}}
            ],
            userZoomingEnabled=False
        )
    
    # Show the Simulation Button (Activated by Overview button)
    @app.callback(
        Output('run-sim-btn', 'style'),
        Input('cytoscape-container', 'children')
    )
    def show_sim_button(graph_content):
        # If the graph has been generated, show the start simulation button
        if graph_content:
            return {"display": "inline-block"}
        return {"display": "none"}

    @app.callback(
    Output('overview-graph', 'elements'), 
    Input('overview-graph', 'tapNodeData'),         
    State('num-nodes', 'value'),
    State('overview-graph', 'elements')             
    )
    def cycle_node_color(node_data, num_nodes, current_elements):
        if node_data is None:
            return dash.no_update

        node_id = int(node_data['id'])
        current_color = node_colors.get(node_id, 'orange')
        index = color_cycle.index(current_color) if current_color in color_cycle else -1
        new_color = color_cycle[(index + 1) % len(color_cycle)]
        node_colors[node_id] = new_color

        # Update stored positions from current elements
        for el in current_elements:
            if 'position' in el and 'id' in el['data']:
                node_positions[int(el['data']['id'])] = el['position']

        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        dimensions = int(math.sqrt(num_nodes))
        return nx_to_cytoscape(G, node_colors, dimensions, use_saved_positions=True)
    
    @app.callback(
    Output('refresh-div', 'children'),  # Use an existing hidden div as the output
    Input('overview-graph', 'mouseoverNodeData'),
    State('overview-graph', 'elements')
    )
    def update_node_positions(node_data, elements):
        if not elements:
            return dash.no_update
        # Update the global node_positions dictionary
        global node_positions
        
        for element in elements:
            if 'position' in element and 'id' in element['data']:
                try:
                    node_id = int(element['data']['id'])
                    node_positions[node_id] = element['position']
                except (ValueError, KeyError):
                    continue
        
        return f"Positions updated: {len(node_positions)}"

    @app.callback(
        Input('run-sim-btn', 'n_clicks'),
        State('overview-graph', 'elements'),
        State('num-nodes', 'value'),
        State('num-edges', 'value'),
        State('num-games', 'value'),
        State('betrayal-threshold', 'value'),
        State('reconnect-pct', 'value'),
        State('max-dist', 'value'),
        State('forgiveness-mode', 'value'),
        State('sarsa-epsilon', 'value'),
        State('sarsa-alpha', 'value'),
        State('sarsa-gamma', 'value'),
        State('moody-epsilon', 'value'),
        State('moody-alpha', 'value'),
        State('moody-gamma', 'value'),
        suppress_callback_exceptions=True
    )
    def launch_simulation(n_clicks, elements, num_nodes, num_edges, num_games, betrayal_thresh, reconnect_pct,
                        max_dist, forgiveness_mode, sarsa_eps, sarsa_alpha, sarsa_gamma,
                        moody_eps, moody_alpha, moody_gamma):
        if not n_clicks:
            return dash.no_update

        agent_types = {
            'red': 'DefectingAgent',
            'green': 'CooperativeAgent',
            'orange': 'WSLSAgent',
            'pink': 'TFTAgent',
            'white': 'MoodySARSAAgent',
            'yellow': 'SARSAAgent'
        }
        assignment = {
            str(node): agent_types.get(color, 'Unknown')
            for node, color in node_colors.items()
        }
        # Capture all current positions from elements
        #global node_positions
        for element in elements:
            if 'position' in element and 'data' in element and 'id' in element['data']:
                try:
                    node_id = int(element['data']['id'])  
                    node_positions[node_id] = element['position']
                except (ValueError, KeyError):
                    continue
        
        print(f"Captured {len(node_positions)} node positions before simulation")
        config = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "node_positions": node_positions,
            "node_colors": node_colors,
            "agent_assignment": assignment,
            "num_games_per_pair": num_games,
            "percent_reconnection": reconnect_pct,
            "max_connection_distance": max_dist,
            "forgiveness_mode": forgiveness_mode,
            "average_considered_betrayal": betrayal_thresh,
            "SARSAAgent_params": {"epsilon": sarsa_eps, "alpha": sarsa_alpha, "gamma": sarsa_gamma},
            "MoodySARSAAgent_params": {"epsilon": moody_eps, "alpha": moody_alpha, "gamma": moody_gamma}
        }
        

        with open("params.json", "w") as f:
            json.dump(config, f, indent=4)

        global app

        # Creates the graph and adds the other stuff to it
        graph = create_networkx_graph(num_nodes=num_nodes, num_edges=num_edges)
        graph = add_edges_to_graph(graph, num_edges, num_nodes, node_positions, float(max_dist))
        
        colors = []
        dimensions = int(math.sqrt(num_nodes)) # dimension is the root of all agents (root of all evil)
        attach_simulation_mode(app, graph, colors, dimensions, node_positions) # attaches simulation gui to flask server
        app.config.suppress_callback_exceptions = True
        import threading, traceback
        try:
            threading.Thread(target=main, args=(app, graph, config), daemon=True).start()
        except Exception as e:
            print("Exception in thread:")
            traceback.print_exc()  # Logs the full stack trace to the console
        return dcc.Location(href="/", id="dummy")
    

    @app.callback(
    Output('url', 'href'),
    Input('run-sim-btn', 'n_clicks'),
    prevent_initial_call=True
    )
    def refresh_after_sim(n_clicks):
        if n_clicks:
            return "/"

    return app



if __name__ == '__main__':
    app = create_gui_app(server)
    app.run(debug=True)
