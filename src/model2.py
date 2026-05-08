from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from feature_engineering import FEATURE_COLS


def train_xgboost_outcome_model(df):
    """
    Train an XGBoost model to predict whether a team will win, draw, or lose.
    """

    X = df[FEATURE_COLS]
    y = df["result_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4, #prevent overfitting
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8, #each tree just use 80% of the features. 
        objective="multi:softprob", #model calculate the prob for each class
        num_class=3, #draw, win, loss
        eval_metric="mlogloss", #penalty if predict wrong
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("XGBoost Outcome Model Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred, target_names=["lose", "draw", "win"]))

    result_map = {
        0: "lose",
        1: "draw",
        2: "win"
    }

    print("\nExample Match Predictions:")

    examples = X_test.copy()
    examples["actual_result"] = y_test
    examples["predicted_result"] = y_pred

    examples["team_name"] = df.loc[X_test.index, "team_name"]
    examples["match_date"] = df.loc[X_test.index, "match_date"]

    for _, row in examples.head(10).iterrows():
        team = row["team_name"]
        actual = result_map[int(row["actual_result"])]
        predicted = result_map[int(row["predicted_result"])]

        print(f"{row['match_date']} | Team: {team}")
        print(f"Actual: {actual} | Predicted: {predicted}")
        print("-" * 50)

    return model