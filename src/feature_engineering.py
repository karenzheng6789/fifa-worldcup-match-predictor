import pandas as pd
import numpy as np

FEATURE_COLS = [
    't_win_rate',
    't_draw_rate',
    't_avg_goals_for',
    't_avg_goals_against',
    't_avg_goal_diff',
    't_knockout_win_rate',
    't_matches_played',
    'o_win_rate',
    'o_draw_rate',
    'o_avg_goals_for',
    'o_avg_goals_against',
    'o_avg_goal_diff',
    'o_knockout_win_rate',
    'o_matches_played',
    'win_rate_diff',
    'goal_diff_diff',
    'attack_vs_defense',
    'experience_diff',
    'is_knockout'
]

def get_team_features(team_id, before_date, df, window=10):
    """ For a given team, we look back at their past matches and calculate basic stats.
        We only use matches before the current match date."""
    # only look at matches before the current match
    past_matches = df[(df['team_id'] == team_id) & (df['match_date'] < before_date)]

    past_matches = past_matches.tail(window) # only look at last 10 matches

    # If it's a team's first time playing, return neutral/average values
    if len(past_matches) == 0:
        return{
            'win_rate': 0.5,
            'draw_rate': 0.2,
            'avg_goals_for': 1.5,
            'avg_goals_against': 1.5,
            'avg_goal_diff': 0.0,
            'knockout_win_rate': 0.5,
            'matches_played': 0
        }

    total_matches = len(past_matches)
    win_rate = past_matches['win'].mean()
    draw_rate = past_matches['draw'].mean()
    avg_goals_for = past_matches['goals_for'].mean()
    avg_goals_against = past_matches['goals_against'].mean()
    avg_goal_diff = past_matches['goal_differential'].mean()

    # Calculate knockout win rates
    knockout_matches = past_matches[past_matches['knockout_stage'] == 1]
    if len(knockout_matches) == 0:
        knockout_win_rate = 0.5   # use neutral value if team has no knockout history
    else:
        knockout_win_rate = knockout_matches['win'].mean()   # return avg of knockout history

    return {
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'avg_goals_for': avg_goals_for,
        'avg_goals_against': avg_goals_against,
        'avg_goal_diff': avg_goal_diff,
        'knockout_win_rate': knockout_win_rate,
        'matches_played': total_matches
    }

def build_features(ta):
    """ Loop through each match and build a row of features per match"""
    ta['match_date'] = pd.to_datetime(ta['match_date']) # converts date from string to actual date
    ta = ta.sort_values('match_date').reset_index(drop=True) # sort date from oldest to most recent

    # since each match appears twice, deduplicate it using the match_id
    unique_matches = ta.drop_duplicates(subset=['match_id'], keep='first')

    all_rows = []
    for index, row in unique_matches.iterrows():
        match_id = row['match_id']
        match_date = row['match_date']
        team_id = row['team_id']
        opp_id = row['opponent_id']

        # split both rows into the team row and opponent row
        both_rows = ta[ta['match_id'] == match_id]
        team_row = both_rows[both_rows['team_id'] == team_id].iloc[0]
        opp_row = both_rows[both_rows['team_id'] == opp_id].iloc[0]

        # calls earlier function to calculate historical stats
        team_feats = get_team_features(team_id, match_date, ta)
        opp_feats = get_team_features(opp_id, match_date, ta)

        # calculates the difference between the team and its opponent
        # positive number = team is better, negative = opp is better, 0 = equal
        win_rate_diff = team_feats['win_rate'] - opp_feats['win_rate']
        goal_diff_diff = team_feats['avg_goal_diff'] - opp_feats['avg_goal_diff']
        attack_vs_defense = team_feats['avg_goals_for'] - opp_feats['avg_goals_against']
        experience_diff = team_feats['matches_played'] - opp_feats['matches_played']

        # build one row combining everything we know about this match
        match_row = {
            # Basic info
            'match_id': match_id,
            'match_date': match_date,
            'team_name': team_row['team_name'],
            'opp_name': opp_row['team_name'],
            'stage': row['stage_name'],

            # Team historical features
            't_win_rate': team_feats['win_rate'],
            't_draw_rate': team_feats['draw_rate'],
            't_avg_goals_for': team_feats['avg_goals_for'],
            't_avg_goals_against': team_feats['avg_goals_against'],
            't_avg_goal_diff': team_feats['avg_goal_diff'],
            't_knockout_win_rate': team_feats['knockout_win_rate'],
            't_matches_played': team_feats['matches_played'],

            # Opponent historical features
            'o_win_rate': opp_feats['win_rate'],
            'o_draw_rate': opp_feats['draw_rate'],
            'o_avg_goals_for': opp_feats['avg_goals_for'],
            'o_avg_goals_against': opp_feats['avg_goals_against'],
            'o_avg_goal_diff': opp_feats['avg_goal_diff'],
            'o_knockout_win_rate': opp_feats['knockout_win_rate'],
            'o_matches_played': opp_feats['matches_played'],

            # Differential features (relative strength)
            'win_rate_diff': win_rate_diff,
            'goal_diff_diff': goal_diff_diff,
            'attack_vs_defense': attack_vs_defense,
            'experience_diff': experience_diff,

            # Context features
            'is_knockout': row['knockout_stage'],  # 1 if knockout, 0 if group stage

            # Target variables (what we want to predict)
            'result': team_row['result'],  # win / draw / lose
            'goals_for': team_row['goals_for'],  # how many goals they scored
            'goals_against': opp_row['goals_for'],  # how many goals opponent scored
        }

        all_rows.append(match_row)

    df = pd.DataFrame.from_records(all_rows) # turns list into dataframe
    #convert result into numbers (win = 2, draw = 1, lose = 0)
    df['result_encoded'] = df['result'].map({'win': 2, 'draw': 1, 'lose': 0})

    # Summary
    print(f"Feature matrix built successfully!")
    print(f"Total matches: {len(df)}")
    print(f"Total features: {len(FEATURE_COLS)}")
    print(f"\nResult breakdown:")
    print(df['result'].value_counts())

    return df

