import numpy as np
import random
import networkx as nx
import math
from pyvis.network import Network
import warnings
from Agent import Agent
from CooperativeAgent import CooperativeAgent
from TFTAgent import TFTAgent
from DefectingAgent import DefectingAgent


class SARSAAgent(Agent):
    def __init__(self, n_states, n_actions, n_agents, id, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.id = id
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor gamma
        self.epsilon = epsilon  # Exploration
        self.betrayal_memory = set()  # Track agents that betrayed this agent
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

    def after_game_function(self, state, action, reward, opponent_reward, opponent_id, opponent_average):
        self.update_q_value(state, action, reward, opponent_id)
        self.update_mood(opponent_id, reward, opponent_reward)
        self.update_memory(opponent_id, reward)

    def update_mood(self, opponent_id, reward, opponent_reward):
        """
        Adjust mood based on self-performance and fairness using opponent's payoff history.
        """

        avg_self_reward = self.average_reward(opponent_id)


        # Calculate average rewards including the new payoff
        self_payoffs = self.memories[opponent_id]
        avg_self_reward_t = np.mean(self_payoffs[-19:] + [reward]) if self_payoffs else reward

        opponent_payoffs = agents[opponent_id].memories[self.id]
        avg_opponent_reward_t = np.mean(opponent_payoffs[-19:] + [opponent_reward]) if opponent_payoffs else opponent_reward

        """avg_self_reward = self.average_payoff
        avg_self_reward_t = avg_self_reward + ((reward - avg_self_reward) / (self.total_games + 1))

        avg_opponent_reward_t = opponent_average"""

        # Calculate alpha and omega (Homo Egualis adjustment)
        alpha = (100 - self.mood) / 100
        beta = alpha
        omega = avg_self_reward_t - (alpha * max(avg_opponent_reward_t - avg_self_reward_t, 0)) - (beta * max(avg_self_reward_t - avg_opponent_reward_t, 0))
        #print((reward - avg_self_reward) + self.prev_omegas[opponent_id])
        # Update mood
        #print(self.mood + (reward - avg_self_reward))
        if self.prev_omegas[opponent_id] > reward:
            self.prev_omegas[opponent_id] *= 1.1
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

        # Use a portion of memory based on mood
        mood_factor = (100 - self.mood) / 100  # Higher mood uses less memory
        memory_slice = math.ceil(len(memory) / mood_factor)
        relevant_memory = memory[-memory_slice:] if memory_slice > 0 else memory

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
            decision_epsilon += 0.8
        if self.mood > 70 and chosen_action == 0:
            decision_epsilon += 0.8

        if random.uniform(0, 1) < decision_epsilon:  # Random chance based on epsilon to choose randomly instead
            chosen_action = random.choice(range(self.n_actions))  # Explore random choice
        return chosen_action

    def update_q_value(self, state, action, reward, opponent_id):
        """
        Modify SARSA's Q-value update rule to incorporate mood-adjusted Ψ.
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
        self.update_average_payoff(reward)

    def average_reward(self, opponent_id, cap=20):
        if len(self.memories[opponent_id]) == 0:
            return 0
        else:
            return np.mean(self.memories[opponent_id][:cap])

    def keep_connected_to_opponent(self, opponent_id):
        avg_A = self.average_reward(opponent_id)  # Avg payoff against opponent
        if avg_A < average_considered_betrayal:
            self.betrayal_memory.add(opponent_id)  # A Add B to list of betrayers
            return 0
        else:
            if opponent_id == 5:
                print(self.id, avg_A)
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

    # Retrieve current states from the environment
    # state_A = env.trust_levels[id_A][id_B]
    # state_B = env.trust_levels[id_B][id_A]

    state_A = agent_A.mood
    state_B = agent_B.mood

    # Agents choose actions
    action_A = agent_A.choose_action(state_A, id_B, fixed)
    action_B = agent_B.choose_action(state_B, id_A, fixed)

    # Step in the environment to get next states and rewards
    (reward_A, reward_B) = env.step(id_A, id_B, action_A, action_B)

    # Agents choose next actions
    # next_action_A = agent_A.choose_action(next_state_A, id_B, fixed)
    # next_action_B = agent_B.choose_action(next_state_B, id_A, fixed)

    # Calculate new average payoffs
    new_average_A = agent_A.average_payoff + ((reward_A - agent_A.average_payoff)/(agent_A.total_games + 1))
    new_average_B = agent_B.average_payoff + ((reward_B - agent_B.average_payoff) / (agent_B.total_games + 1))

    # Update Q-values for both agents
    agent_A.after_game_function(state_A, action_A, reward_A, reward_B, id_B, new_average_B)
    agent_B.after_game_function(state_B, action_B, reward_B, reward_A, id_A, new_average_A)


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
        net.add_node(node, label=str(node), x=x, y=y, fixed=True)

    for edge in graph.edges:
        net.add_edge(*edge)

    # Disable physics
    net.set_options('''var options = { "physics": { "enabled": false } }''')

    return net


# Create a graph and visualize it using PyVis library
num_nodes = 49
num_edges = 50
max_connection_distance = 220
graph = create_networkx_graph(num_nodes=num_nodes, num_edges=num_edges)
visualize(graph)
add_nodes_to_graph(graph)

# Initiate variables
n_agents = num_nodes
n_states = 100
n_actions = 2  # Actions: 0 = defect, 1 = coop
fixed = 0
agents = [SARSAAgent(n_states=n_states, n_actions=n_actions, n_agents=n_agents, id=_) for _ in range(n_agents)]
#agents[24] = CooperativeAgent(id=24, n_states=n_states, n_actions=n_actions, n_agents=n_agents)
agents[1] = TFTAgent(id=1, n_states=n_states, n_actions=n_actions, n_agents=n_agents)
agents[5] = DefectingAgent(id=5, n_states=n_states, n_actions=n_actions, n_agents=n_agents)
env = PrisonersDilemmaEnvironment(n_agents=n_agents, n_states=n_states)

# pair matches
num_games_per_pair = 24999
removed_edges_list = []
reconstruction_interval = 10  # Number of rounds before reconstruction
percent_reconnection = 0.20  # 10% of the population for random reconnection
average_considered_betrayal = 2.2
network_visualization = visualize(graph)
network_visualization.show("network.html")

# Visualize with grid layout
network_visualization = visualize(graph)
network_visualization.show("network.html")

# Rest of your code remains unchanged


for i in range(num_games_per_pair):
    for edge in graph.edges():
        #print(agents[edge[1]], graph.edges(), agents)
        play_game(agents[edge[0]], agents[edge[1]], env, edge[0], edge[1], fixed)
    if fixed >= 1:
        fixed = fixed - 1
    #if fixed == 1:
    #    print(agents[0].q_tables)

    # Perform reconstruction every 'reconstruction_interval' iterations
    if (i+1) % (reconstruction_interval * 100) == 1:
        print(env.total)
        print(i)
        env.reset()
        for agent in agents:
            agent.betrayal_memory.clear()

    if (i + 1) % reconstruction_interval == 0:
        print(f"Reconstruction event at iteration {i + 1}")
        #time.sleep(0.25)

        # Sever connections based on average payoff
        for edge in list(graph.edges()):
            randomValue = random.randint(0, 100)
            if randomValue <= 75:
                continue
            agent_A, agent_B = agents[edge[0]], agents[edge[1]]
            decision_A = agent_A.keep_connected_to_opponent(edge[1])
            decision_B = agent_B.keep_connected_to_opponent(edge[0])

            # Sever the connection if either average payoff is below threshold
            if decision_A == 0 or decision_B == 0:
                graph.remove_edge(edge[0], edge[1])
                #print("Remove edge")

        # Reconnect nodes randomly to 10% of the population
        network_visualization = visualize(graph)
        network_visualization.show("network.html")
        for node in graph.nodes():
            current_neighbors = set(graph.neighbors(node))
            potential_partners = set(range(len(agents))) - current_neighbors - {node} - set(
                agents[node].betrayal_memory)

            if len(potential_partners) > 4:
                new_partners = random.sample(list(potential_partners), int(len(graph.nodes()) * percent_reconnection))
            else:
                new_partners = list(potential_partners)

            # Add new edges
            for partner in new_partners:
                #random_chance = random.random()
                distance = calculate_distance(partner, node)
                if graph.degree(node) > 10 or distance > max_connection_distance:
                    break
                if graph.degree(partner) <= 5:
                    graph.add_edge(node, partner)

network_visualization = visualize(graph)
network_visualization.show("network.html")


print(agents[0].q_tables[1], agents[0].q_tables[7])
'''for i in graph.edges():
    if i[0] == 0 or i[0] == 1:
        print(f"agent {i[1]}:", agents[i[1]].memories[i[0]], agents[i[1]].mood, agents[i[1]].q_tables[i[0]], (i[0], i[1]))
    else:
        continue
        print(f"agent {i[0]}:", agents[i[0]].memories[i[1]], agents[i[0]].mood, agents[i[0]].q_tables[i[1]], (i[0], i[1]))'''
for agent in agents:
    print(agent.mood)
for edge in graph.edges():
    print(edge, agents[edge[0]].memories[edge[1]], agents[edge[1]].memories[edge[0]])