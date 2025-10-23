from flask import Flask, render_template, request
import math

app = Flask(__name__)

# Predefined team ratings (placeholder values; update with real stats as needed)
team_ratings = {
    "ARI": -2.0,
    "ATL": -0.5,
    "BAL": 3.5,
    "BUF": 4.0,
    "CAR": -1.0,
    "CHI": -0.2,
    "CIN": 2.5,
    "CLE": 1.5,
    "DAL": 3.0,
    "DEN": -0.5,
    "DET": 2.0,
    "GB": 1.0,
    "HOU": 1.0,
    "IND": 0.0,
    "JAX": 1.2,
    "KC": 4.5,
    "LV": -0.8,
    "LAC": 0.2,
    "LAR": 0.5,
    "MIA": 2.8,
    "MIN": 0.7,
    "NE": -1.0,
    "NO": 0.8,
    "NYG": -1.2,
    "NYJ": 0.5,
    "PHI": 3.0,
    "PIT": 1.2,
    "SEA": 1.0,
    "SF": 4.2,
    "TB": 0.3,
    "TEN": -0.5,
    "WAS": -1.3,
}

def predict_winner(team_a, team_b):
    rating_a = team_ratings.get(team_a.upper(), 0.0)
    rating_b = team_ratings.get(team_b.upper(), 0.0)
    diff = rating_a - rating_b
    # logistic transform to convert rating difference into a probability between 0 and 1
    prob_a = 1 / (1 + math.exp(-diff))
    if prob_a >= 0.5:
        winner = team_a
        probability = prob_a
    else:
        winner = team_b
        probability = 1 - prob_a
    return winner.upper(), probability

@app.route("/")
def index():
    teams = list(team_ratings.keys())
    return render_template("index.html", teams=teams)

@app.route("/predict", methods=["POST"])
def predict():
    team_a = request.form["teamA"]
    team_b = request.form["teamB"]
    winner, prob = predict_winner(team_a, team_b)
    return render_template("result.html",
                           team_a=team_a.upper(),
                           team_b=team_b.upper(),
                           winner=winner,
                           probability=f"{prob*100:.1f}%")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
