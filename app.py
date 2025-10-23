from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

# load data and model
stats_df = pd.read_csv('2025_team_aggregated_stats.csv', index_col='team')
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# list of features used in model, extended features
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
    # get stats for each team
    a_stats = stats_df.loc[team_a][FEATURE_COLS]
    b_stats = stats_df.loc[team_b][FEATURE_COLS]
    # create difference feature (team A minus team B)
    input_vector = (a_stats - b_stats).values.reshape(1, -1)
    # get probability using model (assumes logistic regression)
    prob = model.predict_proba(input_vector)[0][1]
    winner = team_a if prob >= 0.5 else team_b
    prob_win = prob if prob >= 0.5 else 1 - prob
    return winner, prob_win

@app.route('/')
def index():
    # Provide list of teams from stats dataframe index for dropdown
    teams = stats_df.index.tolist()
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    team_a = request.form['teamA']
    team_b = request.form['teamB']
    winner, probability = predict_game(team_a, team_b)
    return render_template('result.html', team_a=team_a, team_b=team_b,
                           winner=winner, probability=f"{probability*100:.1f}%")

if __name__ == '__main__':
    app.run(debug=True)
