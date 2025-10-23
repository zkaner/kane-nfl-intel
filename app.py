from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os
import math

# Try to load data and model, but fall back to simple ratings if files are missing
stats_df = None
model = None
if os.path.exists('2025_team_aggregated_stats.csv'):
    try:
        stats_df = pd.read_csv('2025_team_aggregated_stats.csv', index_col='team')
    except Exception:
        stats_df = None

if os.path.exists('model.pkl'):
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    except Exception:
        model = None

# Fallback simple team ratings for demonstration (add other teams as needed)
team_ratings = {
    "BUF": 5.2,
    "KC": 4.8,
    "PHI": 4.5,
    "SF": 4.3,
    "DAL": 4.0,
    "BAL": 4.1,
    "GB": 3.5,
    "NYJ": 3.0
    # Add remaining team ratings here
}

# Extended feature list used by the model (only needed if stats_df and model are present)
FEATURE_COLS = [
    'turnover_ratio', 'yards_per_play', 'pass_success_rate', 'rush_success_rate',
    'sack_rate', 'takeaways', 'giveaways', 'offensive_epa', 'defensive_epa',
    'third_down_success_rate', 'third_down_defense_rate',
    'red_zone_td_rate', 'red_zone_defense_rate', 'penalty_yards_per_game',
    'average_starting_field_position', 'time_of_possession',
    'points_per_game', 'points_allowed_per_game', 'rushing_yards_per_game',
    'passing_yards_per_game', 'completion_rate', 'sacks_taken',
    'sacks_made', 'interceptions_thrown', 'interceptions_caught',
    'fumbles_lost', 'fatigue_index', 'travel_distance', 'rest_days',
    'time_zone_change', 'altitude_difference', 'weather_temperature',
    'weather_wind_speed', 'weather_precipitation', 'injury_index',
    'point_differential', 'drive_success_rate', 'field_goal_success_rate',
    'punt_efficiency', 'kick_return_yards', 'penalty_first_downs',
    'early_down_pass_rate', 'offensive_line_pass_block_win_rate',
    'offensive_line_run_block_win_rate', 'defensive_pressure_rate',
    'explosive_play_rate'
]

app = Flask(__name__)

def predict_game(team_a, team_b):
    """Predict winner and probability between two teams."""
    # If we have stats and a trained model, use it
    if stats_df is not None and model is not None:
        try:
            a_stats = stats_df.loc[team_a][FEATURE_COLS]
            b_stats = stats_df.loc[team_b][FEATURE_COLS]
            input_vector = (a_stats - b_stats).values.reshape(1, -1)
            prob = model.predict_proba(input_vector)[0][1]
            winner = team_a if prob >= 0.5 else team_b
            prob_win = prob if prob >= 0.5 else 1 - prob
            return winner, prob_win
        except Exception:
            pass
    # Fallback using simple ratings and logistic function
    rating_a = team_ratings.get(team_a, 0)
    rating_b = team_ratings.get(team_b, 0)
    diff = rating_a - rating_b
    prob = 1 / (1 + math.exp(-diff))
    winner = team_a if prob >= 0.5 else team_b
    prob_win = prob if prob >= 0.5 else 1 - prob
    return winner, prob_win

@app.route('/')
def index():
    # Provide team list for dropdown; use stats_df index if available, else fallback to rating keys
    teams = stats_df.index.tolist() if stats_df is not None else list(team_ratings.keys())
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    team_a = request.form['teamA']
    team_b = request.form['teamB']
    winner, probability = predict_game(team_a, team_b)
    return render_template('result.html', team_a=team_a, team_b=team_b,
                           winner=winner, probability=f"{probability*100:.1f}%")

if __name__ == '__main__':
    # Run on all interfaces to allow hosting platforms to bind to the port
    app.run(host='0.0.0.0', port=10000)
