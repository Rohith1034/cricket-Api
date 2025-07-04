import torch
import numpy as np
import random
from flask import Flask, request, jsonify
import json
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

# Preprocessing function
def preprocess_player(player):
    # Extract wickets from Best_Bowling_Match
    match = re.match(r'^(\d+)-', str(player.get('Best_Bowling_Match', '')))
    if match:
        player['Best_Bowling_Wickets'] = int(match.group(1))
    else:
        player['Best_Bowling_Wickets'] = 0
    
    # Convert numeric fields safely
    for key in ['Batting_Average', 'Bowling_Average', 'Economy_Rate']:
        try:
            player[key] = float(player[key])
        except:
            player[key] = 0.0
    return player

# Load and preprocess player data
BASE_DIR = Path(__file__).resolve().parent

# Load players.json
with open(BASE_DIR / 'players.json') as f:
    players = json.load(f)
processed_players = [preprocess_player(p) for p in players]
df = pd.DataFrame(processed_players)

# Define stats columns
STATS = [
    'Runs_Scored', 'Centuries', 'Half_Centuries', 'Sixes',
    'Highest_Score', 'Batting_Strike_Rate', 'Wickets_Taken',
    'Four_Wicket_Hauls', 'Five_Wicket_Hauls', 'Catches_Taken',
    'Stumpings', 'Best_Bowling_Wickets'
]

# Scale stats
df[STATS] = df[STATS].replace('', 0).astype(float)
scaler = MinMaxScaler()
df[STATS] = scaler.fit_transform(df[STATS])

# Define the neural network model with CORRECT input size (12 features)
class CricketNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_stats = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU()
        )
        self.role_embed = nn.Embedding(4, 8)
        self.fc_combined = nn.Sequential(
            nn.Linear(64 + 8 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.head_player = nn.Linear(128, 11)
        self.head_stat = nn.Linear(128, 12)

    def forward(self, state):
        stats_out = self.fc_stats(state['stats_vector'])
        role_out = self.role_embed(state['role_idx'])
        combined = torch.cat([stats_out, role_out, state['round_vector']], dim=1)
        features = self.fc_combined(combined)
        return self.head_player(features), self.head_stat(features)

# Initialize and load the trained model
model = CricketNN()

# Load with weights_only=True to avoid security warning
model.load_state_dict(
    torch.load(
        BASE_DIR / 'cricket_model.pth',
        map_location=torch.device('cpu'),
        weights_only=True
    )
)
model.eval()

# Team Generator class
class TeamGenerator:
    def __init__(self, players_df):
        self.players_df = players_df
        self.ai_team = []
        self.opponent_team = []
        self.remaining_players = list(players_df.index)
        random.shuffle(self.remaining_players)
        self.round_number = 0

    def generate_teams(self, difficulty):
        # First generate teams without role constraints
        for _ in range(11):
            # AI selects a player
            ai_player_idx = self._ai_select()
            if ai_player_idx is not None:
                self.ai_team.append(ai_player_idx)
                if ai_player_idx in self.remaining_players:
                    self.remaining_players.remove(ai_player_idx)

            # Opponent selects a player
            if self.remaining_players:
                opponent_player_idx = self._opponent_select(difficulty)
                if opponent_player_idx is not None:
                    self.opponent_team.append(opponent_player_idx)
                    if opponent_player_idx in self.remaining_players:
                        self.remaining_players.remove(opponent_player_idx)
            
            self.round_number += 1
        
        # Now adjust teams to meet composition requirements
        self.ai_team = self.adjust_team_composition(self.ai_team)
        self.opponent_team = self.adjust_team_composition(self.opponent_team)
        
        return self.ai_team, self.opponent_team

    def adjust_team_composition(self, team_indices):
        """Adjust team to have 5 batsmen (including 1 wicketkeeper), 2 all-rounders, 4 bowlers"""
        # Categorize players
        players = self.players_df.loc[team_indices]
        wicketkeepers = players[players['type_of_player'].str.contains('Wicketkeeper', case=False, na=False)].index.tolist()
        batsmen = players[players['type_of_player'].str.contains('Batsman', case=False, na=False)].index.tolist()
        allrounders = players[players['type_of_player'].str.contains('All-Rounder', case=False, na=False)].index.tolist()
        bowlers = players[players['type_of_player'].str.contains('Bowler', case=False, na=False)].index.tolist()
        
        # Ensure we have at least 1 wicketkeeper
        if not wicketkeepers:
            # If no wicketkeeper, convert a batsman to wicketkeeper
            if batsmen:
                wicketkeepers = [batsmen.pop(0)]
            elif allrounders:
                wicketkeepers = [allrounders.pop(0)]
            elif bowlers:
                wicketkeepers = [bowlers.pop(0)]
        
        # Ensure we have exactly 1 wicketkeeper
        final_wicketkeeper = wicketkeepers[0] if wicketkeepers else None
        
        # Select 4 batsmen (excluding the wicketkeeper if they were a batsman)
        final_batsmen = []
        if final_wicketkeeper in batsmen:
            batsmen.remove(final_wicketkeeper)
        final_batsmen = batsmen[:4]
        
        # Select 2 all-rounders
        final_allrounders = allrounders[:2]
        
        # Select 4 bowlers (to make total 11 players)
        final_bowlers = bowlers[:4]
        
        # Combine all players
        final_team = [final_wicketkeeper] + final_batsmen + final_allrounders + final_bowlers
        
        # Remove None values if any
        final_team = [idx for idx in final_team if idx is not None]
        
        # If we still don't have 11 players, fill with the best remaining
        if len(final_team) < 11:
            remaining = [idx for idx in team_indices if idx not in final_team]
            final_team += remaining[:11 - len(final_team)]
        
        return final_team[:11]  # Ensure only 11 players

    def _ai_select(self):
        if not self.remaining_players:
            return None
            
        # Build state with zeros (12 features)
        state = {
            'round_vector': torch.tensor([[self.round_number / 11, 1.0]], dtype=torch.float32),
            'stats_vector': torch.zeros((1, 12), dtype=torch.float32),
            'role_idx': torch.tensor([0], dtype=torch.long)
        }
        with torch.no_grad():
            player_logits, _ = model(state)
        
        # Select from available players
        action = player_logits.argmax().item()
        return self.remaining_players[action % len(self.remaining_players)]

    def _opponent_select(self, difficulty):
        if not self.remaining_players:
            return None
            
        if difficulty == 'easy':
            return random.choice(self.remaining_players)
        elif difficulty == 'medium':
            if random.random() < 0.8:
                return random.choice(self.remaining_players)
            else:
                ratings = self.players_df.loc[self.remaining_players, 'Overall_Rating']
                return ratings.idxmax()
        elif difficulty == 'hard':
            state = {
                'round_vector': torch.tensor([[self.round_number / 11, 0.0]], dtype=torch.float32),
                'stats_vector': torch.zeros((1, 12), dtype=torch.float32),
                'role_idx': torch.tensor([0], dtype=torch.long)
            }
            with torch.no_grad():
                player_logits, _ = model(state)
            action = player_logits.argmax().item()
            return self.remaining_players[action % len(self.remaining_players)]

# Flask API setup
app = Flask(__name__)

@app.route('/generate_teams', methods=['POST'])
def generate_teams():
    data = request.json
    difficulty = data.get('difficulty', 'medium')  # Default to medium
    generator = TeamGenerator(df)
    ai_team_indices, opponent_team_indices = generator.generate_teams(difficulty)
    
    # Convert indices to player details with roles
    ai_team = []
    for idx in ai_team_indices:
        # Ensure idx is integer
        if isinstance(idx, int) and idx in df.index:
            player = df.loc[idx]
            ai_team.append({
                'name': player['Player_Name'],
                'role': player['type_of_player'],
                'batting_rating': float(player['Batting_Rating']),
                'bowling_rating': float(player['Bowling_Rating']),
                'overall_rating': float(player['Overall_Rating'])
            })
    
    opponent_team = []
    for idx in opponent_team_indices:
        if isinstance(idx, int) and idx in df.index:
            player = df.loc[idx]
            opponent_team.append({
                'name': player['Player_Name'],
                'role': player['type_of_player'],
                'batting_rating': float(player['Batting_Rating']),
                'bowling_rating': float(player['Bowling_Rating']),
                'overall_rating': float(player['Overall_Rating'])
            })
    
    # Verify team composition
    ai_composition = analyze_composition(ai_team)
    opponent_composition = analyze_composition(opponent_team)
    
    return jsonify({
        'ai_team': ai_team,
        'opponent_team': opponent_team,
        'difficulty': difficulty,
        'ai_composition': ai_composition,
        'opponent_composition': opponent_composition
    })

def analyze_composition(team):
    """Analyze team composition and return counts"""
    batsmen = 0
    wicketkeepers = 0
    allrounders = 0
    bowlers = 0
    
    for player in team:
        role = str(player['role']).lower()
        if 'wicketkeeper' in role:
            wicketkeepers += 1
            batsmen += 1  # Wicketkeeper is also a batsman
        elif 'batsman' in role:
            batsmen += 1
        elif 'all-rounder' in role:
            allrounders += 1
        elif 'bowler' in role:
            bowlers += 1
    
    return {
        'batsmen': batsmen,
        'wicketkeepers': wicketkeepers,
        'allrounders': allrounders,
        'bowlers': bowlers
    }

@app.route('/')
def home():
    return "Cricket Team Generator API - POST to /generate_teams with {'difficulty': 'easy|medium|hard'}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
