import numpy as np
import random

class SARSAAgent:
    def __init__(self, n_states, n_actions, n_agents, alpha=0.1, gamma=0.95, epsilon=0.5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor gamma
        self.epsilon = epsilon  # Exploration

        # Q-tables for each opponent agent
        # dictionary with key = agent id and the value is the q table
        self.q_tables = {i: np.zeros((n_states, n_actions))
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
        else:
            return 1

    def update_q_value(self, state, action, reward, next_state, next_action, opponent_id):
        """SARSA Q-value update for the state-action pair with a specific opponent."""
        q_table = self.q_tables[opponent_id]
        td_target = reward + self.gamma * q_table[next_state, next_action]
        td_error = td_target - q_table[state, action]
        q_table[state, action] += self.alpha * td_error


class PrisonersDilemmaEnvironment:
    def __init__(self, n_states=10):
        self.n_states = n_states  # Discrete states to track trust levels
        self.total = [0, 0]

    def reset(self):
        # Initialize agents' states (could represent trust levels or simply reset states)
        self.state_A = random.randint(0, self.n_states - 1)
        self.state_B = random.randint(0, self.n_states - 1)
        return self.state_A, self.state_B

    def step(self, action_A, action_B):
        """
        rewards of prisoner's dilemma:
        - Both Cooperate= 3 each
        - One Cooperates One Defects= 5 for one, none for the other
        - Both Defect=1 each
        """
        if action_A == 1 and action_B == 1:  # Both Cooperate
            reward_A, reward_B = 3, 3
            self.total[1] += 2
        elif action_A == 1 and action_B == 0:  # A Cooperates, B Defects
            reward_A, reward_B = 0, 5
            self.total[0] += 1
            self.total[1] += 1
        elif action_A == 0 and action_B == 1:  # A Defects, B Cooperates
            reward_A, reward_B = 5, 0
            self.total[0] += 1
            self.total[1] += 1
        else:  # Both Defect
            reward_A, reward_B = 1, 1
            self.total[0] += 2

        # Update states (trust levels, defect is -1 and cooperate is +1)
        self.state_A = max(min(self.state_A + (1 if reward_A > 1 else -1), self.n_states - 1), 0)
        self.state_B = max(min(self.state_B + (1 if reward_B > 1 else -1), self.n_states - 1), 0)

        return (self.state_A, self.state_B), (reward_A, reward_B)


def play_game(agent_A, agent_B, env, num_games, id_A, id_B, fixed=0):
    """Play a number of games between two agents."""
    for game in range(num_games):
        # Reset the environment and agents' states
        state_A, state_B = env.reset()
        action_A = agent_A.choose_action(state_A, id_B, fixed)
        action_B = agent_B.choose_action(state_B, id_A, fixed)


        # Agents take actions and get the outcome
        (next_state_A, next_state_B), (reward_A, reward_B) = env.step(action_A, action_B)
        # Choose next actions based on new states
        next_action_A = agent_A.choose_action(next_state_A, id_B, fixed)
        next_action_B = agent_B.choose_action(next_state_B, id_A, fixed)

        # Update Q-values for both agents
        agent_A.update_q_value(state_A, action_A, reward_A, next_state_A, next_action_A, id_B)
        agent_B.update_q_value(state_B, action_B, reward_B, next_state_B, next_action_B, id_A)

        # Move to the next state and action
        state_A, action_A = next_state_A, next_action_A
        state_B, action_B = next_state_B, next_action_B

    print(f"Finished {num_games} games between Agent {id_A} and Agent {id_B}.")


# Example usage:
n_agents = 5
n_states = 10
n_actions = 2  # Actions: 0 = defect, 1 = cooperate
agents = [SARSAAgent(n_states=n_states, n_actions=n_actions, n_agents=n_agents) for _ in range(n_agents)]
env = PrisonersDilemmaEnvironment(n_states=n_states)

# Pairwise matches
num_games_per_pair = 10000
for i in range(len(agents)):
    for j in range(i + 1, len(agents)):
        play_game(agents[i], agents[j], env, num_games_per_pair, i, j)
