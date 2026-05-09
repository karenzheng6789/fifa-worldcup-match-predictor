from data_loader import load_data
from feature_engineering import build_features, get_team_features, FEATURE_COLS
from model import train_outcome_model
from model2 import train_xgboost_outcome_model
from poisson_model import train_poisson_regression
import pandas as pd
from pathlib import Path
from linear_model import train_linear_regression


team_apps, matches, hosts = load_data()
df = build_features(team_apps)
df["match_date"] = pd.to_datetime(df["match_date"])
df = df[df["match_date"] >= "2010-01-01"]

print("\n================ RANDOM FOREST TRAINING ================")
outcome_model = train_outcome_model(df)

print("\n================ XGBOOST TRAINING ================")
xgb_model = train_xgboost_outcome_model(df)

print("\n================ POISSON GOALS MODEL ================")
goals_model = train_poisson_regression(df)

print("\n================ LINEAR REGRESSION GOALS MODEL ================")
linear_model, linear_preds = train_linear_regression(df)


BASE_DIR = Path(__file__).resolve().parent.parent
fixtures_path = BASE_DIR / "data" / "2026_wc_matches.csv"

fixtures = pd.read_csv(fixtures_path)
fixtures["match_date"] = pd.to_datetime(fixtures["match_date"])
team_apps["match_date"] = pd.to_datetime(team_apps["match_date"])

result_map = {0: "lose", 1: "draw", 2: "win"}


def build_match_row(row, team_apps):
    team_feats = get_team_features(row["team_id"], row["match_date"], team_apps)
    opp_feats = get_team_features(row["opponent_id"], row["match_date"], team_apps)

    match_row = {
        "t_win_rate": team_feats["win_rate"],
        "t_draw_rate": team_feats["draw_rate"],
        "t_avg_goals_for": team_feats["avg_goals_for"],
        "t_avg_goals_against": team_feats["avg_goals_against"],
        "t_avg_goal_diff": team_feats["avg_goal_diff"],
        "t_knockout_win_rate": team_feats["knockout_win_rate"],
        "t_matches_played": team_feats["matches_played"],

        "o_win_rate": opp_feats["win_rate"],
        "o_draw_rate": opp_feats["draw_rate"],
        "o_avg_goals_for": opp_feats["avg_goals_for"],
        "o_avg_goals_against": opp_feats["avg_goals_against"],
        "o_avg_goal_diff": opp_feats["avg_goal_diff"],
        "o_knockout_win_rate": opp_feats["knockout_win_rate"],
        "o_matches_played": opp_feats["matches_played"],

        "win_rate_diff": team_feats["win_rate"] - opp_feats["win_rate"],
        "goal_diff_diff": team_feats["avg_goal_diff"] - opp_feats["avg_goal_diff"],
        "attack_vs_defense": team_feats["avg_goals_for"] - opp_feats["avg_goals_against"],
        "experience_diff": team_feats["matches_played"] - opp_feats["matches_played"],
        "is_knockout": row["knockout_stage"],
    }

    return pd.DataFrame([match_row])[FEATURE_COLS]


def print_fixture_predictions(model, model_name):
    print(f"\n--- 2026 World Cup Predictions Using {model_name} ---")

    for _, row in fixtures.iterrows():
        # Build features from team's perspective
        X = build_match_row(row, team_apps)
        prediction = result_map[model.predict(X)[0]]
        team_goals = goals_model.predict(X)[0]

        # Flip features to get opponent's goals (model only uses team_id)
        flipped_row = {
            'team_id': row['opponent_id'],
            'opponent_id': row['team_id'],
            'team_name': row['opponent_name'],
            'opponent_name': row['team_name'],
            'match_date': row['match_date'],
            'knockout_stage': row['knockout_stage'],
        }
        X_opp = build_match_row(pd.Series(flipped_row), team_apps)
        opp_goals = goals_model.predict(X_opp)[0]

        print(f"{row['match_id']} | {row['team_name']} vs {row['opponent_name']}")

        if prediction == "win":
            print(f"Predicted Winner: {row['team_name']}")
            print(f"Est. Score: {row['team_name']} {team_goals:.1f} - {opp_goals:.1f} {row['opponent_name']}")
        elif prediction == "lose":
            print(f"Predicted Winner: {row['opponent_name']}")
            print(f"Est. Score: {row['team_name']} {team_goals:.1f} - {opp_goals:.1f} {row['opponent_name']}")
        else:
            print(f"Predicted: Draw")
            print(f"Est. Score: {row['team_name']} {team_goals:.1f} - {opp_goals:.1f} {row['opponent_name']}")

        print("-" * 50)


print_fixture_predictions(outcome_model, "Random Forest")
print_fixture_predictions(xgb_model, "XGBoost")


# ---- TOURNAMENT SIMULATION ----

def make_knockout_X(team1_id, team2_id, match_date, team_apps):
    team_feats = get_team_features(team1_id, match_date, team_apps)
    opp_feats = get_team_features(team2_id, match_date, team_apps)

    match_row = {
        "t_win_rate": team_feats["win_rate"],
        "t_draw_rate": team_feats["draw_rate"],
        "t_avg_goals_for": team_feats["avg_goals_for"],
        "t_avg_goals_against": team_feats["avg_goals_against"],
        "t_avg_goal_diff": team_feats["avg_goal_diff"],
        "t_knockout_win_rate": team_feats["knockout_win_rate"],
        "t_matches_played": team_feats["matches_played"],

        "o_win_rate": opp_feats["win_rate"],
        "o_draw_rate": opp_feats["draw_rate"],
        "o_avg_goals_for": opp_feats["avg_goals_for"],
        "o_avg_goals_against": opp_feats["avg_goals_against"],
        "o_avg_goal_diff": opp_feats["avg_goal_diff"],
        "o_knockout_win_rate": opp_feats["knockout_win_rate"],
        "o_matches_played": opp_feats["matches_played"],

        "win_rate_diff": team_feats["win_rate"] - opp_feats["win_rate"],
        "goal_diff_diff": team_feats["avg_goal_diff"] - opp_feats["avg_goal_diff"],
        "attack_vs_defense": team_feats["avg_goals_for"] - opp_feats["avg_goals_against"],
        "experience_diff": team_feats["matches_played"] - opp_feats["matches_played"],
        "is_knockout": 1,
    }

    X = pd.DataFrame([match_row])[FEATURE_COLS]
    return X, team_feats, opp_feats


def predict_winner(team1_id, team2_id, team1_name, team2_name, match_date, team_apps, model):
    X, team_feats, opp_feats = make_knockout_X(team1_id, team2_id, match_date, team_apps)

    prediction = model.predict(X)[0]

    if prediction == 2:
        return team1_id, team1_name

    if prediction == 0:
        return team2_id, team2_name

    team1_score = (
        team_feats["win_rate"]
        + team_feats["avg_goal_diff"]
        + team_feats["knockout_win_rate"]
    )

    team2_score = (
        opp_feats["win_rate"]
        + opp_feats["avg_goal_diff"]
        + opp_feats["knockout_win_rate"]
    )

    if team1_score >= team2_score:
        return team1_id, team1_name
    else:
        return team2_id, team2_name


knockout_date = pd.Timestamp("2026-07-01")

bracket = [
    ("T-46", "Mexico"),         ("T-71", "Korea Republic"),
    ("T-12", "Canada"),        ("T-75", "Switzerland"),
    ("T-09", "Brazil"),        ("T-47", "Morocco"),
    ("T-83", "United States"), ("T-55", "Paraguay"),
    ("T-73", "Spain"),         ("T-48", "Netherlands"),
    ("T-06", "Belgium"),       ("T-03", "Argentina"),
    ("T-58", "Portugal"),      ("T-30", "France"),
    ("T-31", "Germany"),       ("T-28", "England"),
]


def run_tournament_simulation(model, model_name):
    print(f"\n--- TOURNAMENT SIMULATION USING {model_name} ---")

    print("\nRound of 16:")
    r16_winners = []

    for i in range(0, len(bracket), 2):
        t1_id, t1_name = bracket[i]
        t2_id, t2_name = bracket[i + 1]

        winner_id, winner_name = predict_winner(
            t1_id, t2_id, t1_name, t2_name,
            knockout_date, team_apps, model
        )

        r16_winners.append((winner_id, winner_name))
        print(f"  {t1_name} vs {t2_name} → {winner_name}")

    print("\nQuarterfinals:")
    qf_winners = []

    for i in range(0, len(r16_winners), 2):
        t1_id, t1_name = r16_winners[i]
        t2_id, t2_name = r16_winners[i + 1]

        winner_id, winner_name = predict_winner(
            t1_id, t2_id, t1_name, t2_name,
            knockout_date, team_apps, model
        )

        qf_winners.append((winner_id, winner_name))
        print(f"  {t1_name} vs {t2_name} → {winner_name}")

    print("\nSemifinals:")
    sf_winners = []

    for i in range(0, len(qf_winners), 2):
        t1_id, t1_name = qf_winners[i]
        t2_id, t2_name = qf_winners[i + 1]

        winner_id, winner_name = predict_winner(
            t1_id, t2_id, t1_name, t2_name,
            knockout_date, team_apps, model
        )

        sf_winners.append((winner_id, winner_name))
        print(f"  {t1_name} vs {t2_name} → {winner_name}")

    print("\nFinal:")
    t1_id, t1_name = sf_winners[0]
    t2_id, t2_name = sf_winners[1]

    winner_id, winner_name = predict_winner(
        t1_id, t2_id, t1_name, t2_name,
        knockout_date, team_apps, model
    )

    print(f"  {t1_name} vs {t2_name} → {winner_name}")
    print(f"\nPREDICTED 2026 WORLD CUP WINNER USING {model_name}: {winner_name}")

def run_tournament_goals(model_name):
    print(f"\n--- 2026 TOURNAMENT GOAL PREDICTIONS ---")
    print(f"{'Match':<12} {'Team':<25} {'Poisson Goals':>15} {'Linear Goals':>15}")
    print("-" * 70)

    for _, row in fixtures.iterrows():
        # get team goals
        X = build_match_row(row, team_apps)
        poisson_goals = goals_model.predict(X)[0]
        linear_goals = max(0, linear_model.predict(X)[0])

        # flip to get opponent goals
        flipped_row = {
            'team_id': row['opponent_id'],
            'opponent_id': row['team_id'],
            'team_name': row['opponent_name'],
            'opponent_name': row['team_name'],
            'match_date': row['match_date'],
            'knockout_stage': row['knockout_stage'],
        }
        X_opp = build_match_row(pd.Series(flipped_row), team_apps)
        poisson_goals_opp = goals_model.predict(X_opp)[0]
        linear_goals_opp = max(0, linear_model.predict(X_opp)[0])

        print(f"{row['match_id']:<12} {row['team_name']:<25} {poisson_goals:>15.1f} {linear_goals:>15.1f}")
        print(f"{'':12} {row['opponent_name']:<25} {poisson_goals_opp:>15.1f} {linear_goals_opp:>15.1f}")
        print("-" * 70)

    # total goals per team across all group stage matches
    print(f"\n--- TOTAL PREDICTED GROUP STAGE GOALS PER TEAM ---")
    print(f"{'Team':<25} {'Poisson Total':>15} {'Linear Total':>15}")
    print("-" * 55)

    team_goals_poisson = {}
    team_goals_linear = {}

    for _, row in fixtures.iterrows():
        X = build_match_row(row, team_apps)
        p = goals_model.predict(X)[0]
        l = max(0, linear_model.predict(X)[0])

        flipped_row = {
            'team_id': row['opponent_id'],
            'opponent_id': row['team_id'],
            'team_name': row['opponent_name'],
            'opponent_name': row['team_name'],
            'match_date': row['match_date'],
            'knockout_stage': row['knockout_stage'],
        }
        X_opp = build_match_row(pd.Series(flipped_row), team_apps)
        p_opp = goals_model.predict(X_opp)[0]
        l_opp = max(0, linear_model.predict(X_opp)[0])

        team = row['team_name']
        opp = row['opponent_name']

        team_goals_poisson[team] = team_goals_poisson.get(team, 0) + p
        team_goals_poisson[opp] = team_goals_poisson.get(opp, 0) + p_opp
        team_goals_linear[team] = team_goals_linear.get(team, 0) + l
        team_goals_linear[opp] = team_goals_linear.get(opp, 0) + l_opp

    # sort highest to lowest
    sorted_teams = sorted(team_goals_poisson.items(), key=lambda x: x[1], reverse=True)

    for team, poisson_total in sorted_teams:
        linear_total = team_goals_linear[team]
        print(f"{team:<25} {poisson_total:>15.1f} {linear_total:>15.1f}")


run_tournament_simulation(outcome_model, "Random Forest")
run_tournament_simulation(xgb_model, "XGBoost")
run_tournament_goals("Both Models")
