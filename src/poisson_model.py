from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from feature_engineering import FEATURE_COLS

def train_poisson_regression(df):
    """ Predicts how many goals a team will score in a match using Poisson Regression"""
    X = df[FEATURE_COLS]
    y = df['goals_for'] # number of goals scored

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # train model
    model = PoissonRegressor()
    print("\nTraining Poisson Regression model...")
    model.fit(X_train, y_train)

    # make predictions on matches the model has never seen
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"Poisson Model Results")
    print(f"MAE:  {mae:.3f} goals")
    print(f"RMSE: {rmse:.3f} goals")

    return model