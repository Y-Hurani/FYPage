import numpy as np
import pandas as pd
import os
from collections import defaultdict
from MoodySARSAAgent import MoodySARSAAgent

class AgentTracker:
    def __init__(self, agents, layer_size, tracking_frequency, clear_existing=True):
        """
        Initialize the tracker for monitoring agent metrics by layer.
        
        Args:
            agents (list): List of agent objects
            layer_size (int): Number of agents in each layer
            tracking_frequency (int): How often to record data (in rounds)
        """
        self.agents = agents
        self.layer_size = layer_size
        self.tracking_frequency = tracking_frequency
        self.num_layers = len(agents) // layer_size
        self.current_round = 0
        
        # Track metrics in a nested dictionary: {round: {layer: {metric: value}}}
        self.tracking_data = defaultdict(lambda: defaultdict(dict))
        
        # File path for CSV output
        self.csv_path = "agent_metrics_by_layer.csv"
        
        # Check if the CSV exists already
        if clear_existing and os.path.exists(self.csv_path):
            os.remove(self.csv_path)
            self.csv_exists = False
        else:
            self.csv_exists = os.path.exists(self.csv_path)
    
    def track_metrics(self):
        """Record the average mood and score for each layer of agents."""
        self.current_round += self.tracking_frequency
        
        round_data = {}
        
        # Calculate metrics for each layer
        for layer_idx in range(self.num_layers):
            start_idx = layer_idx * self.layer_size # Starting index 10, 20, etc
            end_idx = start_idx + self.layer_size 
            layer_agents = self.agents[start_idx:end_idx]
            
            # Calculate average mood and score for this layer
            # Calculate average mood for MoodySARSAAgents only
            moody_agents = [agent for agent in layer_agents if isinstance(agent, MoodySARSAAgent)]
            if moody_agents:
                avg_mood = round(np.mean([agent.mood for agent in moody_agents]), 3)
            else:
                avg_mood = 0  # Default if no MoodySARSAAgents in this layer
            
            avg_score = round(np.mean([agent.average_payoff for agent in layer_agents]), 3)
            
            # Store the data
            self.tracking_data[self.current_round][layer_idx] = {
                'avg_mood': avg_mood,
                'avg_score': avg_score
            }
            
            # Add to round data for immediate CSV writing
            round_data[f'layer_{layer_idx}_mood'] = avg_mood
            round_data[f'layer_{layer_idx}_score'] = avg_score
        
        # Add round number to the data
        round_data['round'] = self.current_round
        
        # Write the current round data to CSV
        self._write_to_csv(round_data)
    
    def _write_to_csv(self, round_data):
        """Write the current round's data to the CSV file."""
        df = pd.DataFrame([round_data])
        
        # If the file doesn't exist, create it with headers
        if not self.csv_exists:
            df.to_csv(self.csv_path, index=False)
            self.csv_exists = True
        else:
            # Append without writing headers
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
    
    def get_summary(self):
        """Return a summary of the tracked data."""
        return dict(self.tracking_data)