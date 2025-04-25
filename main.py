# main.py
import sys
import dash


node_colors = {}
color_cycle = ['red', 'green', 'orange', 'pink', 'white', 'yellow']
node_positions = {}  # global dictionary to store node positions

if __name__ == "__main__":
    # Start the server
    from flask_server import get_server
    from gui import create_gui_app
    server = get_server()

    

    # Initial placeholder route for /simulation/
    
    
    gui_app = create_gui_app(server)
    gui_app.run(debug=True, port=8050)

    
