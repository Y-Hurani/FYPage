import random


class Agent:
    def __init__(self, id, set_coop= None, set_trust= 1):
        self.id = id
        self.cooperation_probability = set_coop if set_coop is not None else random.uniform(0.2, 1.0)
        self.trust_score = 1.0  # Initial trust score
        self.wealth = 0
        #self.betrayal = random.uniform(0.01, 1)
        self.set_trust = set_trust if set_trust != 1 else round(random.uniform(0.5, 1.5), 1)
        self.tolerance = random.randint(4, 7)
        self.trust_tolerance= round(self.tolerance * 0.1, 2)
        self.wants_to_play = True

    def play_as_A(self, amount):
        """Agent A sends an amount. Returns the tripled amount."""
        return amount * 3

    def play_as_B(self, amount_received):
        """Agent B decides to cooperate or betray based on a probability check."""
        # Calcwulate cooperation threshold from cooperation probability and trust score
        cooperation_threshold = self.cooperation_probability * self.trust_score
        decision = random.random()  # Random value between 0 and 1

        if decision < cooperation_threshold:
            # Cooperative action
            return amount_received * random.uniform(0.3, 0.5)
        else:
            # Greedy action
            return amount_received * random.uniform(0.1, 0.2)

    def interact(self, other_agent, amount_sent):
        """Conduct a full interaction where self is Agent A and other_agent is Agent B."""
        print(
            f"Agent {self.id} (Player A) with cooperation_probability {self.cooperation_probability:.2f} and trust {self.trust_score:.2f} sends ${amount_sent}")
        amount_tripled = self.play_as_A(amount_sent)

        amount_returned = other_agent.play_as_B(amount_tripled)
        print(
            f"Agent {other_agent.id} (Player B) with cooperation_probability {other_agent.cooperation_probability:.2f} and trust {other_agent.trust_score:.2f} returns ${amount_returned:.2f}")

        # Update wealth for both agents
        #self.wealth -= amount_sent
        other_agent.wealth += amount_tripled
        other_agent.wealth -= amount_returned
        self.wealth += amount_returned
        print(other_agent.wealth, self.wealth)

        # Adjust trust based on the return amount and strategy
        if amount_returned >= amount_sent:  # If B returned at least the original amount sent
            #self.trust_score += round(0.01 * amount_returned * self.set_trust, 2)
            self.trust_score = round((self.trust_score + (amount_returned * self.set_trust / 10)) / 2, 2)
        else:
            self.trust_score = round((self.trust_score + (amount_returned / 10)) / 2, 2)
            if (self.trust_score < self.trust_tolerance):
                print(f"Agent {self.id}: man screw you")
                self.wants_to_play = False

agent_a = Agent(id=1)
agent_b = Agent(id=2)

# Simulate a few rounds
amount_to_send = 10
rounds = 10
i = 0
while i < rounds and agent_a.wants_to_play and agent_b.wants_to_play:
    agent_a.interact(agent_b, amount_to_send)
    agent_b.interact(agent_a, amount_to_send)
    i += 1

print(f"Agent {agent_a.id} wealth: ${agent_a.wealth:.2f}, trust_score: {agent_a.trust_score}, trusting: {agent_a.set_trust}")
print(f"Agent {agent_b.id} wealth: ${agent_b.wealth:.2f}, trust_score: {agent_b.trust_score}, trusting: {agent_b.set_trust}")
