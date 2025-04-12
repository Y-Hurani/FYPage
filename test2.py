import time
from collections import deque
import numpy as np
import random
import networkx as nx
import math
from pyvis.network import Network
from Agent import Agent
from CooperativeAgent import CooperativeAgent
from TFTAgent import TFTAgent
from DefectingAgent import DefectingAgent
from SARSAAgent import SARSAAgent
#from MoodySARSAAgent import MoodySARSAAgent
from testing_cypo import create_dash_app  # Replace with your actual filename
from testing_cypo import nx_to_cytoscape
import threading

class MoodySARSAAgent(Agent):
    def __init__(self, n_states, n_actions, n_agents, id, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.id = id
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor gamma
        self.epsilon = epsilon  # Exploration
        self.betrayal_memory = deque()
        self.mood = 50 # random.uniform(1, 99)  # Mood value (1 to 100, neutral mood = 50)
        self.prev_omegas = {i: 0 for i in range(n_agents)}
        self.total_games = 0  # Number of games played so far
        self.average_payoff = 0  # Running average of payoffs

        # Q-tables for each opponent agent
        # dictionary with key = agent id and the value is the q table
        self.q_tables = {i: np.zeros((n_states, n_actions))
                         for i in range(n_agents)}

        # Memory dictionary: key = opponent_id, value = list of last 20 moves/rewards
        self.memories = {i: []
                         for i in range(n_agents)}

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def update_average_payoff(self, payoff):
        self.total_games += 1
        self.average_payoff += (payoff - self.average_payoff) / self.total_games

    def after_game_function(self, state, action, next_state, next_action, reward, opponent_reward, opponent_id):
        self.update_q_value(state, action, reward, opponent_id)
        self.update_mood(opponent_id, reward, opponent_reward)
        self.update_memory(opponent_id, reward)
        self.update_average_payoff(reward)
    
    def calculate_new_omega(self, opponent_id, reward, opponent_reward):
        avg_self_reward = self.average_reward(opponent_id)

        # Calculate average rewards including the new payoff
        self_payoffs = self.memories[opponent_id]
        opponent_payoffs = []
        for rewarded in self_payoffs:
            match rewarded:
                case 0:
                    opponent_payoffs.append(5)
                    break
                case 1:
                    opponent_payoffs.append(1)
                    break
                case 3:
                    opponent_payoffs.append(3)
                    break
                case 5:
                    opponent_payoffs.append(0)
                    break

        avg_self_reward_t = np.mean(self_payoffs[-19:] + [reward]) if self_payoffs else reward

        #opponent_payoffs = agents[opponent_id].memories[self.id]
        avg_opponent_reward_t = np.mean(opponent_payoffs[-19:] + [opponent_reward]) if opponent_payoffs else opponent_reward

        """avg_self_reward = self.average_payoff
        avg_self_reward_t = avg_self_reward + ((reward - avg_self_reward) / (self.total_games + 1))

        avg_opponent_reward_t = opponent_average"""

        # Calculate alpha and omega (Homo Egualis adjustment)
        alpha = (100 - self.mood) / 100
        beta = alpha
        omega = avg_self_reward_t - (alpha * max(avg_opponent_reward_t - avg_self_reward_t, 0)) - (beta * max(avg_self_reward_t - avg_opponent_reward_t, 0))
        return omega

    def update_mood(self, opponent_id, reward, opponent_reward):
        """
        Adjust mood based on self-performance and fairness using opponent's payoff history.
        """
        omega = self.calculate_new_omega(opponent_id, reward, opponent_reward)
        self.mood += int(reward - self.prev_omegas[opponent_id])
        self.mood = max(0, min(99, self.mood))  # Clamp mood to [1, 100]
        self.prev_omegas[opponent_id] = omega

    def compute_mood_adjusted_estimate(self, opponent_id):
        """
        Calculate Ψ, a mood-adjusted estimate for future rewards.
        """
        memory = self.memories[opponent_id]
        if not memory:
            return 0

        '''# Use a portion of memory based on mood
        mood_factor = (100 - self.mood) / 100  # Higher mood uses less memory (alpha)
        memory_slice = math.ceil(len(memory) / mood_factor)
        relevant_memory = memory[-memory_slice:] if memory_slice > 0 else memory'''

        mood_factor = (100 - self.mood) / 100  # Scales between 0 and 1
        max_depth = 20  # Set the maximum depth of memory to consider
        depth = math.ceil(mood_factor * max_depth)  # Scales depth based on mood
        relevant_memory = memory[-depth:] if depth > 0 else memory

        #print(f"Memory Slice (Depth n): {depth}, Mood: {self.mood}")
        return np.mean(relevant_memory) if relevant_memory else 0

    def choose_action(self, state, opponent_id, fixed=0):
        q_table = self.q_tables[opponent_id]
        decision_epsilon = self.epsilon
        if fixed:
            return 1

        # Exploit
        max_value = np.max(q_table[state, :])
        max_actions = [action for action, value in enumerate(q_table[state, :]) if
                       value == max_value]  # Choose action using epsilon-greedy policy for the specific opponent.
        chosen_action = random.choice(max_actions)  # Randomly choose among the actions with max Q-value

        if self.mood < 30 and chosen_action == 1:
            decision_epsilon = 0.9
        if self.mood > 70 and chosen_action == 0:
            decision_epsilon = 0.9

        if random.uniform(0, 1) < decision_epsilon:  # Random chance based on epsilon to choose randomly instead
            #chosen_action = random.choice(range(self.n_actions))  # Explore random choice
            if self.mood > 90:
                chosen_action = random.choices(population=range(self.n_actions), weights=[0.5, 0.5])[0]
            else:
                chosen_action = random.choices(population=range(self.n_actions), weights=[0.5, 0.5])[0]
            #print(chosen_action, chosen_action2)
        return chosen_action

    def update_q_value(self, state, action, reward, opponent_id):
        """
        Modify SARSA's Q-value update rule to incorporate mood-adjusted.
        """
        q_table = self.q_tables[opponent_id]
        mood_adjusted_estimate = self.compute_mood_adjusted_estimate(opponent_id)

        # Standard SARSA update with mood-adjusted Ψ
        td_target = reward + self.gamma * mood_adjusted_estimate
        td_error = td_target - q_table[state, action]
        q_table[state, action] += self.alpha * td_error

    def update_memory(self, opponent_id, reward):
        if len(self.memories[opponent_id]) < 20:
            self.memories[opponent_id].append(reward)
        else:
            self.memories[opponent_id].pop(0)
            self.memories[opponent_id].append(reward)

    def average_reward(self, opponent_id, cap=20):
        if len(self.memories[opponent_id]) == 0:
            return 0
        else:
            return np.mean(self.memories[opponent_id][:cap])

    def keep_connected_to_opponent(self, opponent_id, average_considered_betrayal, round=50):
        avg_A = self.average_reward(opponent_id)  # Avg payoff against opponent
        if avg_A < average_considered_betrayal:
            self.betrayal_memory.append(opponent_id)  # A Add B to list of betrayers
            return 0
        else:
            return 1

class PrisonersDilemmaEnvironment:
    def __init__(self, n_agents, n_states=10):
        self.n_agents = n_agents  # Total number of agents
        self.n_states = n_states  # Trust levels range from 0 to 9
        self.total = [0, 0]  # Tracks total cooperation/defection counts

        # Initialize trust levels: dictionary of dictionaries
        self.trust_levels = {agent: {other: 0 for other in range(n_agents) if other != agent}
                             for agent in range(n_agents)}

    def reset(self):
        """Reset all trust levels to neutral."""
        self.trust_levels = {agent: {other: 0 for other in range(self.n_agents) if other != agent}
                             for agent in range(self.n_agents)}
        self.total = [0, 0]

    def step(self, id_A, id_B, action_A, action_B):
        """
        Simulate a single game between two agents and update trust levels.
        """
        # Determine rewards based on actions
        if action_A == 1 and action_B == 1:  # Both Cooperate
            reward_A, reward_B = 3, 3
            #reward_A +=  self.trust_levels[id_A][id_B] * 0.1
            #reward_B += self.trust_levels[id_B][id_A] * 0.1
            self.total[1] += 2  # Cooperation counter
        elif action_A == 1 and action_B == 0:  # A Cooperates, B Defects
            reward_A, reward_B = 0, 5
            self.total[0] += 1  # Defection counter
        elif action_A == 0 and action_B == 1:  # A Defects, B Cooperates
            reward_A, reward_B = 5, 0
            self.total[0] += 1
        else:  # Both Defect
            reward_A, reward_B = 1, 1
            self.total[0] += 2

        # Update trust levels based on outcomes
        #self.update_trust(id_A, id_B, action_A, action_B)

        #new_state_a = self.trust_levels[id_A][id_B]
        #new_state_b = self.trust_levels[id_B][id_A]

        return (reward_A, reward_B)

    def update_trust(self, agent_A, agent_B, action_A, action_B):
        """
        Adjust trust levels based on actions.
        Trust increases with cooperation and decreases with defection.
        """
        if action_A == 1:  # Agent A cooperated
            self.trust_levels[agent_B][agent_A] = min(self.n_states - 1, self.trust_levels[agent_B][agent_A] + 1)
        else:  # Agent A defected
            self.trust_levels[agent_B][agent_A] = max(0, self.trust_levels[agent_B][agent_A] - 1)

        if action_B == 1:  # Agent B cooperated
            self.trust_levels[agent_A][agent_B] = min(self.n_states - 1, self.trust_levels[agent_A][agent_B] + 1)
        else:  # Agent B defected
            self.trust_levels[agent_A][agent_B] = max(0, self.trust_levels[agent_A][agent_B] - 1)

def play_game(agent_A, agent_B, env, id_A, id_B, fixed=0):
    """
    Plays one game of IPD between two agents, retrieving states from the environment.
    """


    state_A = agent_A.mood if isinstance(agent_A, MoodySARSAAgent) else 50
    state_B = agent_B.mood if isinstance(agent_B, MoodySARSAAgent) else 50

    # Agents choose actions
    action_A = agent_A.choose_action(state_B, id_B, fixed)
    action_B = agent_B.choose_action(state_A, id_A, fixed)

    # Step in the environment to get next states and rewards
    (reward_A, reward_B) = env.step(id_A, id_B, action_A, action_B)

    next_state_A = max(0, min(99, agent_A.mood + int(reward_A - agent_A.prev_omegas[agent_B.id]))) if isinstance(agent_A, MoodySARSAAgent) else 50
    next_state_B = max(0, min(99, agent_B.mood + int(reward_B - agent_B.prev_omegas[agent_A.id]))) if isinstance(agent_B, MoodySARSAAgent) else 50

    # Agents choose next actions
    next_action_A = agent_A.choose_action(next_state_A, id_B, fixed)
    next_action_B = agent_B.choose_action(next_state_B, id_A, fixed)

    # Calculate new average payoffs
    new_average_A = agent_A.average_payoff + ((reward_A - agent_A.average_payoff) / (agent_A.total_games + 1)) 
    new_average_B = agent_B.average_payoff + ((reward_B - agent_B.average_payoff) / (agent_B.total_games + 1)) 

    # Update Q-values for both agents
    agent_A.after_game_function(state_A, action_A, next_state_A, next_action_A, reward_A, reward_B, id_B)
    agent_B.after_game_function(state_B, action_B, next_state_B, next_action_B, reward_B, reward_A, id_A)


    # Update the trust states in the environment
    # env.trust_levels[id_A][id_B] = next_state_A
    # env.trust_levels[id_B][id_A] = next_state_B

node_positions = {} # For Pyvis illustration (Coordinates of each node)
def calculate_distance(node_a, node_b, positions=node_positions):
    """
    Calculate Euclidean distance between two nodes.
    :param node_a: First node ID
    :param node_b: Second node ID
    :param positions: Dictionary of node positions
    :return: Distance
    """
    x1, y1 = positions[node_a]
    x2, y2 = positions[node_b]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def create_networkx_graph(num_nodes=25, num_edges=40):
    graph = nx.Graph()
    # Add nodes
    graph.add_nodes_from(range(num_nodes))  # Create nodes with ID's from 0 to num_nodes -1
    return graph

def agent_class_condition(node, agents):
    """
    Determines the color of a node based on the agent's class.
    :param node: The node ID.
    :param agents: A dictionary of agents where keys are node IDs and values are agent instances.
    :return: A string representing the node color.
    """
    if isinstance(agents[node], MoodySARSAAgent):
        return 'green'
    elif isinstance(agents[node], CooperativeAgent):
        return 'blue'
    else:
        return 'orange'  # Default color for other classes

def add_nodes_to_graph(graph):
    # Randomly create edges between nodes
    while graph.number_of_edges() < num_edges:
        node_a = random.randint(0, num_nodes - 1)
        node_b = random.randint(0, num_nodes - 1)

        # No self-loops or duplicates
        distance = calculate_distance(node_a, node_b)
        if node_a != node_b and not graph.has_edge(node_a, node_b) and distance < max_connection_distance:
            graph.add_edge(node_a, node_b)

        return graph

def visualize(graph, grid_size=7):
    net = Network(notebook=True, directed=False, cdn_resources='remote')  # Non-directed graph

    # Define fixed positions for the nodes in a grid
    for node in graph.nodes:
        row = node // grid_size
        col = node % grid_size
        x = col * 100
        y = row * 100
        node_positions[node] = (x, y)

    for node, (x, y) in node_positions.items():
        net.add_node(node, x=x, y=y, fixed=True)

    for edge in graph.edges:
        net.add_edge(*edge)

    return net

def generate_agents(n_agents, weights, n_states, n_actions):
    """
    Generate a list of agents based on weighted probabilities.

    :param n_agents: Total number of agents to generate.
    :param weights: A dictionary containing the weights for each agent type.
                    Example: {"MoodySARSAAgent": 0.5, "CooperativeAgent": 0.2, "DefectingAgent": 0.2, "TFTAgent": 0.1}
    :param n_states: Number of states for the agents.
    :param n_actions: Number of actions for the agents.
    :return: A list of generated agents.
    """
    agent_classes = {
        "MoodySARSAAgent": MoodySARSAAgent,
        "SARSAAgent": SARSAAgent,
        "CooperativeAgent": CooperativeAgent,
        "DefectingAgent": DefectingAgent,
        "TFTAgent": TFTAgent
    }

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {key: val / total_weight for key, val in weights.items()}

    # Prepare a list of agent types and probabilities
    agent_types = list(agent_classes.keys())
    probabilities = [normalized_weights[agent_type] for agent_type in agent_types]

    # Generate agents based on the weighted random selection
    agents = []
    for agent_id in range(n_agents):
        chosen_type = random.choices(agent_types, probabilities, k=1)[0]
        agent_class = agent_classes[chosen_type]
        agents.append(agent_class(id=agent_id, n_states=n_states, n_actions=n_actions, n_agents=n_agents))

    return agents

def mood_to_color(mood):
    """
    Converts mood (1 to 100) to a grayscale hex color.
    """
    # Clamp to range
    mood = max(1, min(100, mood))
    # Map mood to grayscale intensity (0 = black, 255 = white)
    intensity = int((mood / 100) * 255)
    hex_value = f'{intensity:02x}'  # 2-digit hex
    return f'#{hex_value}{hex_value}{hex_value}'

def update_colors_moods():
    colors, moods = [], []
    for agent in agents:
        moods.append('None')
        if isinstance(agent, CooperativeAgent):
            colors.append('green')
        elif isinstance(agent, MoodySARSAAgent):
            colors.append(mood_to_color(agent.mood))
            moods[-1] = agent.mood
        elif isinstance(agent, TFTAgent):
            colors.append('pink')
        elif isinstance(agent, DefectingAgent):
            colors.append('red')
        else:
            colors.append('yellow')
    return colors, moods

def trigger_forgiveness(mode):
    if mode == 'WIPE':
        for agent in agents:
            if isinstance(agent, MoodySARSAAgent) or isinstance(agent, SARSAAgent):
                agent.betrayal_memory.clear()
    elif mode == 'POP':
        for agent in agents:
            if isinstance(agent, MoodySARSAAgent) or isinstance(agent, SARSAAgent):
                if len(agent.betrayal_memory) > 10:
                    agent.betrayal_memory.popleft()


# Create a graph and visualize it
num_nodes = 100
num_edges = 50
dimensions = int(math.sqrt(num_nodes))
max_connection_distance = 225
graph = create_networkx_graph(num_nodes=num_nodes, num_edges=num_edges)
# Initiate variables
n_agents = num_nodes
n_states = 100
n_actions = 2  # Actions: 0 = defect, 1 = coop
fixed = 0
weights = {
    "SARSAAgent": 0.2,
    "MoodySARSAAgent": 0.8,
    "CooperativeAgent": 0.00,  
    "DefectingAgent": 0.00,  
    "TFTAgent": 0.00
}

agents = generate_agents(n_agents, weights, n_states, n_actions)
colors, moods = update_colors_moods()
visualize(graph, dimensions)
add_nodes_to_graph(graph)
#agents = [MoodySARSAAgent(n_states=n_states, n_actions=n_actions, n_agents=n_agents, id=_) for _ in range(n_agents)]
#agents[24] = CooperativeAgent(id=24, n_states=n_states, n_actions=n_actions, n_agents=n_agents)
#agents[1] = TFTAgent(id=1, n_states=n_states, n_actions=n_actions, n_agents=n_agents)
#agents[5] = DefectingAgent(id=5, n_states=n_states, n_actions=n_actions, n_agents=n_agents)
env = PrisonersDilemmaEnvironment(n_agents=n_agents, n_states=n_states)

# pair matches
num_games_per_pair = 249999
removed_edges_list = []
reconstruction_interval = 10  # number of rounds before reconstruction
percent_reconnection = 0.20  # % of the population for random reconnection
average_considered_betrayal = 3


# Start the Dash app in a separate thread or process
app = create_dash_app(graph, colors, dimensions)

# Start Dash server in thread
thread = threading.Thread(target=lambda: app.run(debug=True, use_reloader=False))
thread.daemon = True
thread.start()
possible_pairs = [(a, b) for a in range(n_agents) for b in range(a + 1, n_agents)]
subset_size = int(len(possible_pairs) * percent_reconnection)

# game loop to update the Dash graph
for i in range(num_games_per_pair):
    for edge in graph.edges():
        play_game(agents[edge[0]], agents[edge[1]], env, edge[0], edge[1], fixed)
    if fixed >= 1:
        fixed = fixed - 1

    if (i + 1) % (reconstruction_interval * 10) == 1:
        print(env.total)
        env.reset()
        trigger_forgiveness('POP')

    if (i + 1) % reconstruction_interval == 0:
        print(f"Reconstruction event at iteration {i + 1}")

        evaluated_pairs = random.sample(possible_pairs, subset_size)

        for agent_a, agent_b in evaluated_pairs:
            distance = calculate_distance(agent_b, agent_a)
            if distance > max_connection_distance:
                continue
            if graph.has_edge(agent_a, agent_b):
                decision_a = agents[agent_a].keep_connected_to_opponent(agent_b, average_considered_betrayal, i)
                decision_b = agents[agent_b].keep_connected_to_opponent(agent_a, average_considered_betrayal, i)
                if decision_a == 0:# or decision_b == 0:
                    graph.remove_edge(agent_a, agent_b)
            else:
                # If either of the agents are moody and hold grudge against opponent, dont connect
                condition_A, condition_B = True, True
                if isinstance(agent_a, MoodySARSAAgent):
                    if agent_b in agents[agent_a].betrayal_memory:
                        condition_A = False

                if isinstance(agent_b, MoodySARSAAgent):
                    if agent_a in agents[agent_b].betrayal_memory:
                        condition_B = False

                if condition_A and condition_B:
                    graph.add_edge(agent_a, agent_b)

        # Trigger Dash Cytoscape to redraw the updated graph
        # This replaces `network_visualization.show("network.html")`
        colors, moods = update_colors_moods()
        elements = nx_to_cytoscape(graph, colors, dimensions, moods)
        app.layout.children[-1].elements = elements
        app.update_data(colors, moods)
        for agent in agents:
            if isinstance(agent, MoodySARSAAgent):
                agent.set_epsilon(agent.epsilon * 0.9995)
        time.sleep(0.1)