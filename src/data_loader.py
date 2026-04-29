import pandas as pd
import os

def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    ta      = pd.read_csv(os.path.join(data_dir, 'team_appearances.csv'))
    matches = pd.read_csv(os.path.join(data_dir, 'matches.csv'))
    hosts   = pd.read_csv(os.path.join(data_dir, 'host_countries.csv'))

    return ta, matches, hosts