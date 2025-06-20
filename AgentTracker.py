import numpy as np
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
from MoodySARSAAgent import MoodySARSAAgent
from SARSAAgent import SARSAAgent
from TFTAgent import TFTAgent
from WSLSAgent import WSLSAgent

class AgentTracker:
    def __init__(self, agents, layer_size, tracking_frequency, max_degrees, clear_existing=True):
        """
        Initialize the tracker for monitoring agent metrics.
        """
        self.agents = agents
        self.layer_size = layer_size
        self.max_degrees = max_degrees
        self.tracking_frequency = tracking_frequency
        self.num_layers = len(agents) // layer_size
        self.current_round = 0
        
        # track metrics in a nested dictionary:}
        self.tracking_data = defaultdict(lambda: defaultdict(dict))
        
        # path for CSV output
        now = datetime.now()
        timestamp = now.strftime("%d-%m-%Y %H-%M-%S")

        # Filename with timestamp
        self.csv_path_mood = f"stats/results_mood_{timestamp}.csv"
        self.csv_path_score = f"stats/results_score_{timestamp}.csv"
        if not os.path.exists('stats'):
            os.makedirs('stats')
          
        # if the CSV exists already, remove it
        if clear_existing and (os.path.exists(self.csv_path_score) or os.path.exists(self.csv_path_mood)):
            os.remove(self.csv_path_mood) if os.path.exists(self.csv_path_mood) else None
            os.remove(self.csv_path_score) if os.path.exists(self.csv_path_score) else None
            self.csv_exists = False
        else:
            self.csv_exists = os.path.exists(self.csv_path_score)

    def _calculate_normalized_degree(self, agent_id, current_degrees):
        """
        Calculate the normalized degree centrality for an agent.
        
        Args:
            agent_id (int): The ID of the agent
            current_degrees (dict): Dictionary mapping nodes to their current degrees
            
        Returns:
            float: Normalized degree (degree / max_possible_degree)
        """
        if self.max_degrees is None or current_degrees is None:
            return 0.0  # Return 0 if degrees not provided
        
        current_degree = current_degrees.get(agent_id, 0)
        max_degree = self.max_degrees.get(agent_id, 1)  # default to 1 to avoid division by zero
        
        # normalize the degree (0 to 1 scale)
        normalized_degree = current_degree / max_degree if max_degree > 0 else 0
        
        return round(normalized_degree, 3)

    def track_layer_metrics(self, current_degrees):
        """Record the average mood and score for each layer of agents.
           Used early on for specific experiments.
        """
        self.current_round += self.tracking_frequency
        
        round_data = ({}, {}, {})
        
        # Calculate metrics for each layer
        for layer_idx in range(self.num_layers):
            start_idx = layer_idx * self.layer_size # Starting index 10, 20, etc
            end_idx = start_idx + self.layer_size 
            layer_agents = self.agents[start_idx:end_idx]
            
            # Calculate average mood for moody agents only in each layer
            moody_agents = [agent for agent in layer_agents if isinstance(agent, MoodySARSAAgent)]
            if moody_agents:
                avg_mood = round(np.mean([agent.mood for agent in moody_agents]), 3)
            else:
                avg_mood = 0  # Default if no moody in this layer
            
            avg_score = round(np.mean([agent.average_payoff for agent in layer_agents]), 3)
            avg_connectivity = round(np.mean([self._calculate_normalized_degree(agent.id, current_degrees) 
                                         for agent in layer_agents]), 3)
            
            self.tracking_data[self.current_round][layer_idx] = {
                'avg_mood': avg_mood,
                'avg_score': avg_score,
                'avg_connectivity': avg_connectivity
            }
            
            # Write the data on CSV file
            round_data[0][f'layer_{layer_idx}_mood'] = avg_mood
            round_data[1][f'layer_{layer_idx}_score'] = avg_score
            round_data[2][f'layer_{layer_idx}_connectivity'] = avg_connectivity
        
        # Add round number to the data
        round_data[0]['round'] = self.current_round
        round_data[1]['round'] = self.current_round
        round_data[2]['round'] = self.current_round
        
        # Write the current round data to CSV
        self._write_to_csv(round_data, mode='layer')
    
    def track_types_metrics(self, current_degrees):
        """
        Record the average mood and score for moody agents vs non-moody agents,
        regardless of which layer they are in. Supports all four implemented agents (Moody, SARSA, TFT, WSLS)
        """
        moody_agents = [agent for agent in self.agents if isinstance(agent, MoodySARSAAgent)]
        sarsa_agents = [agent for agent in self.agents if isinstance(agent, SARSAAgent)]
        tft_agents =   [agent for agent in self.agents if isinstance(agent, TFTAgent)]
        wsls_agents =  [agent for agent in self.agents if isinstance(agent, WSLSAgent)]
    
        # Calculate metrics for moody agents
        if moody_agents:
            avg_mood_moody = round(np.mean([agent.mood for agent in moody_agents]), 3)
            avg_score_moody = round(np.mean([agent.average_payoff for agent in moody_agents]), 3)
            avg_connectivity_moody = round(np.mean([self._calculate_normalized_degree(agent.id, current_degrees) 
                                              for agent in moody_agents]), 3)
            avg_cooperation_moody = round(np.mean([agent.total_cooperation/agent.total_games for agent in moody_agents]), 3)
        else:
            avg_mood_moody = 0
            avg_score_moody = 0
            avg_connectivity_moody = 0
            avg_cooperation_moody = 0
        
    
        # Calculate metrics for sarsa agents
        if sarsa_agents:
            avg_score_sarsa = round(np.mean([agent.average_payoff for agent in sarsa_agents]), 3)
            avg_connectivity_sarsa = round(np.mean([self._calculate_normalized_degree(agent.id, current_degrees) 
                                                  for agent in sarsa_agents]), 3)
            avg_cooperation_sarsa = round(np.mean([agent.total_cooperation/agent.total_games for agent in sarsa_agents]), 3)
        else:
            avg_score_sarsa = 0
            avg_connectivity_sarsa = 0
            avg_cooperation_sarsa = 0
    
        # Calculate metrics for TFT agents
        if tft_agents:
            avg_score_tft = round(np.mean([agent.average_payoff for agent in tft_agents]), 3)
            avg_connectivity_tft = round(np.mean([self._calculate_normalized_degree(agent.id, current_degrees) 
                                                for agent in tft_agents]), 3)
            avg_cooperation_tft = round(np.mean([agent.total_cooperation/agent.total_games for agent in tft_agents]), 3)
        else:
            avg_score_tft = 0
            avg_connectivity_tft = 0
            avg_cooperation_tft = 0

        # Calculate metrics for WSLS agents
        if wsls_agents:
            avg_score_wsls = round(np.mean([agent.average_payoff for agent in wsls_agents]), 3)
            avg_connectivity_wsls = round(np.mean([self._calculate_normalized_degree(agent.id, current_degrees) 
                                                for agent in wsls_agents]), 3)
            avg_cooperation_wsls = round(np.mean([agent.total_cooperation/agent.total_games for agent in wsls_agents]), 3)
        else:
            avg_score_wsls = 0
            avg_connectivity_wsls = 0
            avg_cooperation_wsls = 0

        # Store the data in the tracking dictionary
        self.tracking_data[self.current_round]['agent_types'] = {
            'moody_avg_mood': avg_mood_moody,
            'moody_avg_score': avg_score_moody,
            'moody_avg_connectivity': avg_connectivity_moody,
            'moody_avg_cooperation': avg_cooperation_moody,
            'sarsa_avg_score': avg_score_sarsa,
            'sarsa_avg_connectivity': avg_connectivity_sarsa,
            'sarsa_avg_cooperation': avg_cooperation_sarsa,
            'tft_avg_score': avg_score_tft,
            'tft_avg_connectivity': avg_connectivity_tft,
            'tft_avg_cooperation': avg_cooperation_tft,
            'wsls_avg_score': avg_score_wsls,
            'wsls_avg_connectivity': avg_connectivity_wsls,
            'wsls_avg_cooperation': avg_cooperation_wsls
        }

        # Create a dictionary for CSV output
        agent_type_data = {
            'round': self.current_round,
            'moody_avg_mood': avg_mood_moody,
            'moody_avg_score': avg_score_moody,
            'moody_avg_connectivity': avg_connectivity_moody,
            'moody_avg_cooperation': avg_cooperation_moody,
            'sarsa_avg_score': avg_score_sarsa,
            'sarsa_avg_connectivity': avg_connectivity_sarsa,
            'sarsa_avg_cooperation': avg_cooperation_sarsa,
            'tft_avg_score': avg_score_tft,
            'tft_avg_connectivity': avg_connectivity_tft,
            'tft_avg_cooperation': avg_cooperation_tft,
            'wsls_avg_score': avg_score_wsls,
            'wsls_avg_connectivity': avg_connectivity_wsls,
            'wsls_avg_cooperation': avg_cooperation_wsls
        }

        self._write_to_csv(agent_type_data, mode='types')  
        self.current_round += self.tracking_frequency

    def _write_to_csv(self, data, mode='layer'):
        """
        Write tracking data to CSV files based on the specified mode.
        """
        if mode == 'layer':
            # Handle layer based metrics
            df_mood = pd.DataFrame([data[0]])
            df_score = pd.DataFrame([data[1]])
        
            # If file don't exist create it with headers
            if not self.csv_exists:
                df_mood.to_csv(self.csv_path_mood, index=False)
                df_score.to_csv(self.csv_path_score, index=False)
                self.csv_exists = True
            else:
                # append without writing headers
                df_mood.to_csv(self.csv_path_mood, mode='a', header=False, index=False)
                df_score.to_csv(self.csv_path_score, mode='a', header=False, index=False)
    
        elif mode == 'types':
            # handle type metrics
            csv_path = self.csv_path_score
            df = pd.DataFrame([data])
        
            if not os.path.exists(csv_path):
                df.to_csv(csv_path, index=False)
            else:
                df.to_csv(csv_path, mode='a', header=False, index=False)
    
    def get_summary(self):
        """spits out all the tracked data. not sure why, but here u go
           I most likely implemented this to eventually make dynamic charts but i can't remember"""
        return dict(self.tracking_data)
    
