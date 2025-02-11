import numpy as np
import random
import networkx as nx
import time
from pyvis.network import Network
from Agent import Agent
from CooperativeAgent import CooperativeAgent

class SARSAAgent(Agent):
    def __init__(self, id, n_states, n_actions, n_agents, alpha=0.1, gamma=0.95, epsilon=0.10):
        self.id = id
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor gamma
        self.epsilon = epsilon  # Exploration
        self.betrayal_memory = set()  # Track agents that betrayed this agent

        # Q-tables for each opponent agent
        # dictionary with key = agent id and the value is the q table
        self.q_tables = {i: np.zeros((n_states, n_actions))
                         for i in range(n_agents)}

        # Memory dictionary: key = opponent_id, value = list of last 20 moves/rewards
        self.memories = {i: []
                         for i in range(n_agents)}

    def choose_action(self, state, opponent_id, fixed=0):
        if not fixed:
            # Choose action using epsilon-greedy policy for the specific opponent.
            q_table = self.q_tables[opponent_id]
            if random.uniform(0, 1) < self.epsilon:
                return random.choice(range(self.n_actions))  # Explore random choice
            else:
                # Exploit
                max_value = np.max(q_table[state, :])
                max_actions = [action for action, value in enumerate(q_table[state, :]) if value == max_value]
                return random.choice(max_actions)  # Randomly choose among the actions with max Q-value
            # this is done to eliminate the bias at the beginning, where choosing the heighest action will be zero,
            # meaning that the agent would always choose the first action (defect)
            # self.epsilon = self.epsilon * 0.9995
        else:
            return 1


    def update_q_value(self, state, action, reward, next_state, next_action, opponent_id):
        """SARSA Q-value update for the state-action pair with a specific opponent."""
        q_table = self.q_tables[opponent_id]
        td_target = reward + self.gamma * q_table[next_state, next_action]
        td_error = td_target - q_table[state, action]
        q_table[state, action] += self.alpha * td_error


    def update_memory(self, opponent_id, reward):
        if len(self.memories[opponent_id]) < 20:
            self.memories[opponent_id].append(reward)
        else:
            self.memories[opponent_id].pop(0)
            self.memories[opponent_id].append(reward)

    def average_reward(self, opponent_id):
        if len(self.memories[opponent_id]) == 0:
            return 0
        else:
            return np.mean(self.memories[opponent_id])

    def keep_connected_to_opponent(self, opponent_id):
        avg_A = agent_A.average_reward(opponent_id)  # Avg payoff against opponent
        if avg_A < average_considered_betrayal:
            agent_A.betrayal_memory.add(opponent_id)  # A Add B to list of betrayers
            return 0
        else:
            return 1




class PrisonersDilemmaEnvironment:
    def __init__(self, n_agents, n_states=10):
        self.n_agents = n_agents  # Total number of agents
        self.n_states = n_states  # Trust levels range from 0 to 9
        self.total = [0, 0]  # Tracks total cooperation/defection counts

        # Initialize trust levels: dictionary of dictionaries
        self.trust_levels = {agent: {other: 5 for other in range(n_agents) if other != agent}
                             for agent in range(n_agents)}

    def reset(self):
        """Reset all trust levels to neutral."""
        self.trust_levels = {agent: {other: 5 for other in range(self.n_agents) if other != agent}
                             for agent in range(self.n_agents)}
        self.total = [0, 0]

    def step(self, id_A, id_B, action_A, action_B):
        """
        Simulate a single game between two agents and update trust levels.
        """
        # Determine rewards based on actions
        if action_A == 1 and action_B == 1:  # Both Cooperate
            reward_A, reward_B = 3, 3
            reward_A +=  self.trust_levels[id_A][id_B] * 0.1
            reward_B += self.trust_levels[id_B][id_A] * 0.1
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
        self.update_trust(id_A, id_B, action_A, action_B)

        new_state_a = self.trust_levels[id_A][id_B]
        new_state_b = self.trust_levels[id_B][id_A]

        return (new_state_a, new_state_b), (reward_A, reward_B)

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
    state_A = env.trust_levels[id_A][id_B]
    state_B = env.trust_levels[id_B][id_A]

    # Agents choose actions
    action_A = agent_A.choose_action(state_A, id_B, fixed)
    action_B = agent_B.choose_action(state_B, id_A, fixed)

    # Step in the environment to get next states and rewards
    (next_state_A, next_state_B), (reward_A, reward_B) = env.step(id_A, id_B, action_A, action_B)

    # Agents choose next actions
    next_action_A = agent_A.choose_action(next_state_A, id_B, fixed)
    next_action_B = agent_B.choose_action(next_state_B, id_A, fixed)

    # Update Q-values for both agents
    agent_A.update_q_value(state_A, action_A, reward_A, next_state_A, next_action_A, id_B)
    agent_B.update_q_value(state_B, action_B, reward_B, next_state_B, next_action_B, id_A)

    # Update memories
    agent_A.update_memory(id_B, reward_A)
    agent_B.update_memory(id_A, reward_B)

    # Update the trust states in the environment
    env.trust_levels[id_A][id_B] = next_state_A
    env.trust_levels[id_B][id_A] = next_state_B



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
    net = Network(notebook=True, directed=False) # non-directed as both agents play it together

    # Add nodes and edges from the graph
    for node in graph.nodes:
        net.add_node(node, label=str(node))
    for edge in graph.edges:
        net.add_edge(*edge)
    return net


# Create a graph and visualize it using PyVis library
num_nodes = 40
num_edges = 40
graph = create_networkx_graph(num_nodes=num_nodes, num_edges=num_edges)
network_visualization = visualize(graph)
network_visualization.show("network.html")

# Initiate variables
n_agents = num_nodes
n_states = 10
n_actions = 2  # Actions: 0 = defect, 1 = coop
fixed = 5
agents = [SARSAAgent(id=_, n_states=n_states, n_actions=n_actions, n_agents=n_agents) for _ in range(n_agents)]
agents[2] = CooperativeAgent(id=2, n_states=n_states, n_actions=n_actions, n_agents=n_agents)
env = PrisonersDilemmaEnvironment(n_agents=n_agents, n_states=n_states)

# pair matches
num_games_per_pair = 25000
removed_edges_list = []
reconstruction_interval = 20  # Number of game iterations before reconstruction
percent_reconnection = 0.10  # 10% of the population for random reconnection
average_considered_betrayal = 2.5


for i in range(num_games_per_pair):
    for edge in graph.edges():
        play_game(agents[edge[0]], agents[edge[1]], env, edge[0], edge[1], fixed)
    if fixed >= 1:
        fixed = fixed - 1

    # Perform reconstruction every 'reconstruction_interval' iterations
    if (i+1) % (reconstruction_interval * 50) == 1:
        print(env.total)
        env.reset()
        for agent in agents:
            agent.betrayal_memory.clear()

    if (i + 1) % reconstruction_interval == 0:
        print(f"Reconstruction event at iteration {i + 1}")
        #time.sleep(0.25)

        # Sever connections based on average payoff
        for edge in list(graph.edges()):
            agent_A, agent_B = agents[edge[0]], agents[edge[1]]
            decision_A = agent_A.keep_connected_to_opponent(edge[1])
            decision_B = agent_B.keep_connected_to_opponent(edge[0])

            # Sever the connection if either average payoff is below threshold
            if not decision_A or not decision_B:
                graph.remove_edge(edge[0], edge[1])

        # Reconnect nodes randomly to 10% of the population
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
                graph.add_edge(node, partner)

        network_visualization = visualize(graph)
        network_visualization.show("network.html")



for i in graph.edges():
    if i[0] == 0:
        print(f"Q-table for agent 0 against agent {i[1]}:")
        print(agents[0].q_tables[i[1]])
    elif i[1] == 0:
        print(f"Q-table for agent 0 against agent {i[0]}:")
        print(agents[0].q_tables[i[0]])
print(fixed)