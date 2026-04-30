from data_loader import load_data
from feature_engineering import build_features
from model import train_outcome_model

team_apps, matches, hosts = load_data()

df = build_features(team_apps)

outcome_model = train_outcome_model(df)