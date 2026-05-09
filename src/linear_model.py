from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from feature_engineering import FEATURE_COLS

def train_linear_regression(df):
    """ Predicts how many goals a team will score using Linear Regression"""
    X = df[FEATURE_COLS]
    y = df['goals_for']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    print("\nTraining Linear Regression model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"\nLinear Regression Results")
    print(f"MAE:  {mae:.3f} goals")
    print(f"RMSE: {rmse:.3f} goals")

    # print example matches
    print("\nExample Goal Predictions:")
    print("-" * 50)

    # grab team names and dates for the test matches
    examples = X_test.copy()
    examples['actual_goals'] = y_test.values
    examples['predicted_goals'] = predictions
    examples['team_name'] = df.loc[X_test.index, 'team_name'].values
    examples['match_date'] = df.loc[X_test.index, 'match_date'].values

    for _, row in examples.head(5).iterrows():
        print(f"{row['match_date']} | Team: {row['team_name']}")
        print(f"Actual Goals: {row['actual_goals']} | Predicted Goals: {row['predicted_goals']:.1f}")
        print("-" * 50)

    return model, predictions