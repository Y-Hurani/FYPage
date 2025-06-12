from flask import Flask

# Create the Flask server to be shared
server = Flask(__name__)

def get_server():
    return server


if __name__ == "__main__":
    server.run(debug=True)