from Agent import Agent
class DefectingAgent(Agent):
    def __init__(self, id, n_states, n_actions, n_agents):
        super().__init__(id, n_states, n_actions, n_agents)

    def update_memory(self, opponent_id, reward):
        if len(self.memories[opponent_id]) < 20:
            self.memories[opponent_id].append(reward)
        else:
            self.memories[opponent_id].pop(0)
            self.memories[opponent_id].append(reward)

    def choose_action(self, state, opponent_id, whatever, **kwargs):
        """
        Always choose action 1 (cooperate) regardless of state or opponent.
        """
        return 0

    def after_game_function(self, state, action, reward, opponent_reward, opponent_id, opponent_average, **kwargs):
        self.update_memory(opponent_id, reward)

    def reset(self):
        """
        Reset memories if needed between episodes or games.
        """
        self.memories = {opponent_id: [] for opponent_id in range(self.n_agents)}

    def keep_connected_to_opponent(self, opponent_id, average_considered_betrayal, round=50):
        avg_A = self.average_reward(opponent_id)  # Avg payoff against opponent
        if avg_A < average_considered_betrayal:

            self.betrayal_memory.add(opponent_id)  # A Add B to list of betrayers
            return 0
        else:
            if opponent_id == 5:
                print(self.id, avg_A)
            return 1