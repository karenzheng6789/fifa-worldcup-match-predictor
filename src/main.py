from data_loader import load_data
from feature_engineering import build_features, get_team_features, FEATURE_COLS
from model import train_outcome_model
from poisson_model import train_poisson_regression
import pandas as pd
from pathlib import Path
team_apps, matches, hosts = load_data()
df = build_features(team_apps)
outcome_model = train_outcome_model(df)
goals_model = train_poisson_regression(df)

print("\n--- 2026 World Cup Predictions ---")

BASE_DIR = Path(__file__).resolve().parent.parent
fixtures_path = BASE_DIR/"data"/"2026_wc_matches.csv"
fixtures = pd.read_csv(fixtures_path)
#fixtures = pd.read_csv('../data/2026_wc_matches.csv')
fixtures['match_date'] = pd.to_datetime(fixtures['match_date'])
team_apps['match_date'] = pd.to_datetime(team_apps['match_date'])

result_map = {0: 'lose', 1: 'draw', 2: 'win'}

for _, row in fixtures.iterrows():
    team_feats = get_team_features(row['team_id'], row['match_date'], team_apps)
    opp_feats = get_team_features(row['opponent_id'], row['match_date'], team_apps)

    match_row = {
        't_win_rate': team_feats['win_rate'],
        't_draw_rate': team_feats['draw_rate'],
        't_avg_goals_for': team_feats['avg_goals_for'],
        't_avg_goals_against': team_feats['avg_goals_against'],
        't_avg_goal_diff': team_feats['avg_goal_diff'],
        't_knockout_win_rate': team_feats['knockout_win_rate'],
        't_matches_played': team_feats['matches_played'],
        'o_win_rate': opp_feats['win_rate'],
        'o_draw_rate': opp_feats['draw_rate'],
        'o_avg_goals_for': opp_feats['avg_goals_for'],
        'o_avg_goals_against': opp_feats['avg_goals_against'],
        'o_avg_goal_diff': opp_feats['avg_goal_diff'],
        'o_knockout_win_rate': opp_feats['knockout_win_rate'],
        'o_matches_played': opp_feats['matches_played'],
        'win_rate_diff': team_feats['win_rate'] - opp_feats['win_rate'],
        'goal_diff_diff': team_feats['avg_goal_diff'] - opp_feats['avg_goal_diff'],
        'attack_vs_defense': team_feats['avg_goals_for'] - opp_feats['avg_goals_against'],
        'experience_diff': team_feats['matches_played'] - opp_feats['matches_played'],
        'is_knockout': row['knockout_stage'],
    }

    X = pd.DataFrame([match_row])[FEATURE_COLS]
    prediction = result_map[outcome_model.predict(X)[0]]
    goals = goals_model.predict(X)[0]

    print(f"{row['match_id']} | {row['team_name']} vs {row['opponent_name']}")
    if prediction == 'win':
        print(f"Predicted Winner: {row['team_name']} | Est. Goals: {goals:.1f}")
    elif prediction == 'lose':
        print(f"Predicted Winner: {row['opponent_name']}")
    else:
        print(f"Predicted: Draw")
    print("-" * 50)

# ---- TOURNAMENT SIMULATION ----

def predict_winner(team1_id, team2_id, team1_name, team2_name, match_date, ta, outcome_model, goals_model):
    team_feats = get_team_features(team1_id, match_date, ta)
    opp_feats = get_team_features(team2_id, match_date, ta)

    match_row = {
        't_win_rate': team_feats['win_rate'],
        't_draw_rate': team_feats['draw_rate'],
        't_avg_goals_for': team_feats['avg_goals_for'],
        't_avg_goals_against': team_feats['avg_goals_against'],
        't_avg_goal_diff': team_feats['avg_goal_diff'],
        't_knockout_win_rate': team_feats['knockout_win_rate'],
        't_matches_played': team_feats['matches_played'],
        'o_win_rate': opp_feats['win_rate'],
        'o_draw_rate': opp_feats['draw_rate'],
        'o_avg_goals_for': opp_feats['avg_goals_for'],
        'o_avg_goals_against': opp_feats['avg_goals_against'],
        'o_avg_goal_diff': opp_feats['avg_goal_diff'],
        'o_knockout_win_rate': opp_feats['knockout_win_rate'],
        'o_matches_played': opp_feats['matches_played'],
        'win_rate_diff': team_feats['win_rate'] - opp_feats['win_rate'],
        'goal_diff_diff': team_feats['avg_goal_diff'] - opp_feats['avg_goal_diff'],
        'attack_vs_defense': team_feats['avg_goals_for'] - opp_feats['avg_goals_against'],
        'experience_diff': team_feats['matches_played'] - opp_feats['matches_played'],
        'is_knockout': 1,
    }

    X = pd.DataFrame([match_row])[FEATURE_COLS]
    prediction = outcome_model.predict(X)[0]

    # in knockout rounds draws go to team1 winning (simplification)
    if prediction == 2 or prediction == 1:
        return team1_id, team1_name
    else:
        return team2_id, team2_name

knockout_date = pd.Timestamp('2026-07-01')

# define the bracket - top 2 from each group (simplified based on group stage predictions)
# format: (team_id, team_name)
bracket = [
    ('T-46', 'Mexico'),       ('T-71', 'Korea Republic'),
    ('T-12', 'Canada'),       ('T-75', 'Switzerland'),
    ('T-09', 'Brazil'),       ('T-47', 'Morocco'),
    ('T-83', 'United States'),('T-55', 'Paraguay'),
    ('T-73', 'Spain'),        ('T-48', 'Netherlands'),
    ('T-06', 'Belgium'),      ('T-03', 'Argentina'),
    ('T-58', 'Portugal'),     ('T-30', 'France'),
    ('T-31', 'Germany'),      ('T-28', 'England'),
]

print("\n--- TOURNAMENT SIMULATION ---")

# Round of 16
print("\nRound of 16:")
r16_winners = []
for i in range(0, len(bracket), 2):
    t1_id, t1_name = bracket[i]
    t2_id, t2_name = bracket[i+1]
    winner_id, winner_name = predict_winner(t1_id, t2_id, t1_name, t2_name, knockout_date, team_apps, outcome_model, goals_model)
    r16_winners.append((winner_id, winner_name))
    print(f"  {t1_name} vs {t2_name} → {winner_name}")

# Quarterfinals
print("\nQuarterfinals:")
qf_winners = []
for i in range(0, len(r16_winners), 2):
    t1_id, t1_name = r16_winners[i]
    t2_id, t2_name = r16_winners[i+1]
    winner_id, winner_name = predict_winner(t1_id, t2_id, t1_name, t2_name, knockout_date, team_apps, outcome_model, goals_model)
    qf_winners.append((winner_id, winner_name))
    print(f"  {t1_name} vs {t2_name} → {winner_name}")

# Semifinals
print("\nSemifinals:")
sf_winners = []
for i in range(0, len(qf_winners), 2):
    t1_id, t1_name = qf_winners[i]
    t2_id, t2_name = qf_winners[i+1]
    winner_id, winner_name = predict_winner(t1_id, t2_id, t1_name, t2_name, knockout_date, team_apps, outcome_model, goals_model)
    sf_winners.append((winner_id, winner_name))
    print(f"  {t1_name} vs {t2_name} → {winner_name}")

# Final
print("\nFinal:")
t1_id, t1_name = sf_winners[0]
t2_id, t2_name = sf_winners[1]
winner_id, winner_name = predict_winner(t1_id, t2_id, t1_name, t2_name, knockout_date, team_apps, outcome_model, goals_model)
print(f"  {t1_name} vs {t2_name} → {winner_name}")

print(f"\n🏆 PREDICTED 2026 WORLD CUP WINNER: {winner_name} 🏆")
