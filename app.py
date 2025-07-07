import torch
import numpy as np
import random
from flask import Flask, request, jsonify
import json
import pandas as pd
import re
import torch.nn as nn
import os
from pathlib import Path

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
    
    # Normalize role names
    role = str(player.get('type_of_player', '')).lower()
    if 'wicket' in role:
        player['type_of_player'] = 'Wicketkeeper'
    elif 'bat' in role:
        player['type_of_player'] = 'Batsman'
    elif 'all' in role or 'round' in role:
        player['type_of_player'] = 'All-Rounder'
    elif 'bowl' in role:
        player['type_of_player'] = 'Bowler'
    else:
        player['type_of_player'] = 'Batsman'  # Default
    
    return player

# Get current script directory
BASE_DIR = Path(__file__).resolve().parent

# Load and preprocess player data
with open(BASE_DIR / 'players.json') as f:
    players = json.load(f)
    
processed_players = [preprocess_player(p) for p in players]
df = pd.DataFrame(processed_players)
df.reset_index(drop=True, inplace=True)  # Ensure consistent integer indices

# Define stats columns
STATS = [
    'Runs_Scored', 'Centuries', 'Half_Centuries', 'Sixes',
    'Highest_Score', 'Batting_Strike_Rate', 'Wickets_Taken',
    'Four_Wicket_Hauls', 'Five_Wicket_Hauls', 'Catches_Taken',
    'Stumpings', 'Best_Bowling_Wickets'
]

# Convert stats to float
df[STATS] = df[STATS].replace('', 0).astype(float)

# Define the neural network model
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

# Load model with absolute path
model.load_state_dict(
    torch.load(
        BASE_DIR / 'cricket_model.pth',
        map_location=torch.device('cpu'),
        weights_only=True
    )
)
model.eval()

# Team Generator class with improved composition logic
class TeamGenerator:
    def __init__(self, players_df):
        self.players_df = players_df
        self.ai_team = []
        self.opponent_team = []
        self.remaining_players = list(players_df.index)
        random.shuffle(self.remaining_players)
        self.round_number = 0

    def generate_teams(self, difficulty):
        # Convert difficulty to lowercase for consistent handling
        difficulty = difficulty.lower()
        
        # First generate teams without role constraints
        for _ in range(11):
            # AI selects a player
            ai_player_idx = self._ai_select()
            if ai_player_idx is not None and ai_player_idx in self.remaining_players:
                self.ai_team.append(ai_player_idx)
                self.remaining_players.remove(ai_player_idx)

            # Opponent selects a player
            if self.remaining_players:
                opponent_player_idx = self._opponent_select(difficulty)
                if opponent_player_idx is not None and opponent_player_idx in self.remaining_players:
                    self.opponent_team.append(opponent_player_idx)
                    self.remaining_players.remove(opponent_player_idx)
            
            self.round_number += 1
        
        # Now adjust teams to meet composition requirements
        self.ai_team, ai_roles = self.adjust_team_composition(self.ai_team)
        self.opponent_team, opponent_roles = self.adjust_team_composition(self.opponent_team)
        
        return (self.ai_team, ai_roles), (self.opponent_team, opponent_roles)

    def adjust_team_composition(self, team_indices):
        """Adjust team to have 5 batsmen (including 1 wicketkeeper), 2 all-rounders, 4 bowlers"""
        # Categorize players
        players = self.players_df.loc[team_indices]
        wicketkeepers = players[players['type_of_player'] == 'Wicketkeeper'].index.tolist()
        batsmen = players[players['type_of_player'] == 'Batsman'].index.tolist()
        allrounders = players[players['type_of_player'] == 'All-Rounder'].index.tolist()
        bowlers = players[players['type_of_player'] == 'Bowler'].index.tolist()
        others = [idx for idx in team_indices if idx not in wicketkeepers + batsmen + allrounders + bowlers]
        
        # Assign roles to "others" based on their ratings
        for idx in others:
            player = self.players_df.loc[idx]
            bowling_strength = player['Bowling_Rating'] > player['Batting_Rating']
            
            if bowling_strength and player['Bowling_Rating'] > 30:
                bowlers.append(idx)
            elif player['Batting_Rating'] > 30:
                batsmen.append(idx)
            elif player['Bowling_Rating'] > 20 and player['Batting_Rating'] > 20:
                allrounders.append(idx)
            else:
                # Assign to largest category
                cats = [('Batsman', batsmen), ('Bowler', bowlers), 
                       ('All-Rounder', allrounders), ('Wicketkeeper', wicketkeepers)]
                cats.sort(key=lambda x: len(x[1]))
                cats[0][1].append(idx)
        
        # Ensure we have at least 1 wicketkeeper
        if not wicketkeepers:
            # Find best candidate to convert to wicketkeeper from batsmen or all-rounders
            candidates = batsmen + allrounders
            if candidates:
                # Select player with best batting skills
                best_idx = max(candidates, key=lambda idx: self.players_df.loc[idx, 'Batting_Rating'])
                if best_idx in batsmen: batsmen.remove(best_idx)
                if best_idx in allrounders: allrounders.remove(best_idx)
                wicketkeepers.append(best_idx)
            else:
                # Fallback: use the best batsman if no candidates
                if batsmen:
                    best_idx = max(batsmen, key=lambda idx: self.players_df.loc[idx, 'Batting_Rating'])
                    batsmen.remove(best_idx)
                    wicketkeepers.append(best_idx)
                elif allrounders:
                    best_idx = max(allrounders, key=lambda idx: self.players_df.loc[idx, 'Batting_Rating'])
                    allrounders.remove(best_idx)
                    wicketkeepers.append(best_idx)
                elif bowlers:
                    best_idx = max(bowlers, key=lambda idx: self.players_df.loc[idx, 'Batting_Rating'])
                    bowlers.remove(best_idx)
                    wicketkeepers.append(best_idx)
        
        # Ensure we have exactly 1 wicketkeeper
        final_wicketkeeper = wicketkeepers[0] if wicketkeepers else None
        if len(wicketkeepers) > 1:
            # Keep the best wicketkeeper, convert others to batsmen
            best_wk = max(wicketkeepers, key=lambda idx: self.players_df.loc[idx, 'Batting_Rating'])
            wicketkeepers = [best_wk]
            for idx in wicketkeepers[1:]:
                batsmen.append(idx)
        
        # Remove wicketkeeper from batsmen if present
        if final_wicketkeeper in batsmen:
            batsmen.remove(final_wicketkeeper)
        
        # Ensure we have 4 batsmen (total batsmen will be 5 including wicketkeeper)
        if len(batsmen) < 4:
            # Convert all-rounders to batsmen
            needed = 4 - len(batsmen)
            if needed > 0 and allrounders:
                convert = min(needed, len(allrounders))
                batsmen.extend(allrounders[:convert])
                allrounders = allrounders[convert:]
                needed -= convert
            
            if needed > 0 and bowlers:
                # Convert bowlers with batting skills
                bowling_batsmen = sorted(
                    bowlers, 
                    key=lambda idx: self.players_df.loc[idx, 'Batting_Rating'], 
                    reverse=True
                )[:needed]
                batsmen.extend(bowling_batsmen)
                for idx in bowling_batsmen:
                    if idx in bowlers: bowlers.remove(idx)
        
        # Ensure we have 2 all-rounders
        if len(allrounders) < 2:
            needed = 2 - len(allrounders)
            if needed > 0 and batsmen:
                # Convert batsmen with bowling skills
                batting_allrounders = sorted(
                    batsmen, 
                    key=lambda idx: self.players_df.loc[idx, 'Bowling_Rating'], 
                    reverse=True
                )[:needed]
                allrounders.extend(batting_allrounders)
                for idx in batting_allrounders:
                    if idx in batsmen: batsmen.remove(idx)
                needed -= len(batting_allrounders)
            
            if needed > 0 and bowlers:
                # Convert bowlers with batting skills
                bowling_allrounders = sorted(
                    bowlers, 
                    key=lambda idx: self.players_df.loc[idx, 'Batting_Rating'], 
                    reverse=True
                )[:needed]
                allrounders.extend(bowling_allrounders)
                for idx in bowling_allrounders:
                    if idx in bowlers: bowlers.remove(idx)
        
        # Ensure we have 4 bowlers
        if len(bowlers) < 4:
            needed = 4 - len(bowlers)
            if needed > 0 and allrounders:
                convert = min(needed, len(allrounders))
                bowlers.extend(allrounders[:convert])
                allrounders = allrounders[convert:]
                needed -= convert
            
            if needed > 0 and batsmen:
                # Convert batsmen with bowling skills
                batting_bowlers = sorted(
                    batsmen, 
                    key=lambda idx: self.players_df.loc[idx, 'Bowling_Rating'], 
                    reverse=True
                )[:needed]
                bowlers.extend(batting_bowlers)
                for idx in batting_bowlers:
                    if idx in batsmen: batsmen.remove(idx)
        
        # Final team composition
        final_team = [final_wicketkeeper] + batsmen[:4] + allrounders[:2] + bowlers[:4]
        # Ensure exactly 11 players
        final_team = final_team[:11]
        
        # Create role assignments
        role_assignments = {}
        role_assignments[final_wicketkeeper] = "Wicketkeeper"
        for idx in batsmen[:4]:
            role_assignments[idx] = "Batsman"
        for idx in allrounders[:2]:
            role_assignments[idx] = "All-Rounder"
        for idx in bowlers[:4]:
            role_assignments[idx] = "Bowler"
        
        # Handle any remaining players
        for idx in final_team:
            if idx not in role_assignments:
                # Assign based on original role
                role_assignments[idx] = self.players_df.loc[idx, 'type_of_player']
        
        return final_team, [role_assignments[idx] for idx in final_team]

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
            
        difficulty = difficulty.lower()  # Ensure consistent case handling
        
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
        else:  # Handle unknown difficulties
            return random.choice(self.remaining_players)

# Flask API setup
app = Flask(__name__)

def analyze_composition(team):
    """Analyze team composition and return counts"""
    batsmen = 0
    wicketkeepers = 0
    allrounders = 0
    bowlers = 0
    
    for player in team:
        role = str(player.get('assigned_role', player.get('type_of_player', ''))).lower()
        if 'wicket' in role:
            wicketkeepers += 1
            batsmen += 1  # Wicketkeeper is also a batsman
        elif 'bat' in role:
            batsmen += 1
        elif 'all' in role or 'round' in role:
            allrounders += 1
        elif 'bowl' in role:
            bowlers += 1
        else:
            # Default to batsman if role not recognized
            batsmen += 1
    
    return {
        'batsmen': batsmen,
        'wicketkeepers': wicketkeepers,
        'allrounders': allrounders,
        'bowlers': bowlers
    }

@app.route('/generate_teams', methods=['POST'])
def generate_teams():
    data = request.json
    difficulty = data.get('difficulty', 'medium')  # Default to medium
    
    # Convert difficulty to lowercase for consistent handling
    if difficulty:
        difficulty = difficulty.lower()
    
    generator = TeamGenerator(df)
    (ai_team_indices, ai_roles), (opponent_team_indices, opponent_roles) = generator.generate_teams(difficulty)
    
    # Convert indices to complete player details
    ai_team = []
    for idx, role in zip(ai_team_indices, ai_roles):
        if isinstance(idx, int) and idx in df.index:
            player = df.loc[idx]
            player_dict = player.to_dict()
            player_dict['assigned_role'] = role  # Add assigned role
            ai_team.append(player_dict)
    
    opponent_team = []
    for idx, role in zip(opponent_team_indices, opponent_roles):
        if isinstance(idx, int) and idx in df.index:
            player = df.loc[idx]
            player_dict = player.to_dict()
            player_dict['assigned_role'] = role  # Add assigned role
            opponent_team.append(player_dict)
    
    # Verify team composition
    ai_composition = analyze_composition(ai_team)
    opponent_composition = analyze_composition(opponent_team)
    
    return jsonify({
        'ai_team': ai_team[:11],  # Ensure exactly 11 players
        'opponent_team': opponent_team[:11],  # Ensure exactly 11 players
        'difficulty': difficulty,
        'ai_composition': ai_composition,
        'opponent_composition': opponent_composition
    })

@app.route('/')
def home():
    return "Cricket Team Generator API - POST to /generate_teams with {'difficulty': 'easy|medium|hard'}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
